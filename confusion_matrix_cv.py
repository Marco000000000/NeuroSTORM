import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pytorch_lightning as pl
import gc
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, roc_curve, classification_report
import warnings

# Sopprimi warning
warnings.filterwarnings("ignore", category=UserWarning, module="lightning_fabric")

# Import tuoi
from utils.cv_data_module import fMRIDataModule
from models.lightning_model import LightningModel
import pytorch_lightning as pl

def get_preds(model, loader, num_classes):
    """Estrae predizioni (y_true, y_probs) da un DataLoader"""
    model.eval()
    y_true = []
    y_probs = []
    
    with torch.no_grad():
        for batch in loader:
            # Sposta su GPU
            if isinstance(batch, dict):
                batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            else:
                batch = [b.cuda() if isinstance(b, torch.Tensor) else b for b in batch]
                
            _, logits, t, _ = model._compute_logits(batch)
            
            # Calcolo Probabilità
            if num_classes == 2:
                if logits.ndim == 1 or logits.shape[1] == 1:
                    probs = torch.sigmoid(logits).squeeze()
                else:
                    probs = torch.softmax(logits, dim=1)[:, 1]
            else:
                probs = torch.softmax(logits, dim=1)
            
            y_true.extend(t.cpu().numpy())
            y_probs.extend(probs.cpu().numpy())
            
    return np.array(y_true), np.array(y_probs)

def save_confusion_matrix(y_true, y_pred, save_path, title, labels):
    """Helper per plottare e salvare la matrice di confusione"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def cli_main():
    # 1. Inizializza Parser
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter, add_help=False)
    parser.add_argument("--downstream_task_id", type=int, default="1", help="downstream task id")
    parser.add_argument("--task_name", type=str, default="d", help="specify the task name")
    parser.add_argument("--downstream_task_type", type=str, default="classification", help="select either classification or regression")

    # 2. Argomenti ESCLUSIVI script
    parser.add_argument("--project_name", type=str, required=True, help="Nome cartella output")
    parser.add_argument("--num_folds", type=int, default=5, help="Numero di fold CV")
    parser.add_argument("--seed", type=int, default=1234, help="Seed globale")
    parser.add_argument("--dataset_name", type=str, default="UCLA")
    
    # 3. Import argomenti
    parser = LightningModel.add_model_specific_args(parser) 
    parser = pl.Trainer.add_argparse_args(parser)
    parser = fMRIDataModule.add_data_specific_args(parser)

    # 5. Parsing
    args = parser.parse_args()
    
    if args.image_path is None:
        parser.error("L'argomento --image_path è obbligatorio.")

    # Setup Root Directory
    if args.model == "neurostorm": category_dir = "neurostorm"
    else: category_dir = "other"
    base_dir = os.path.join('output', category_dir, args.project_name)
    
    print(f"📂 Ricerca risultati in: {base_dir}")
    if not os.path.exists(base_dir):
        raise ValueError(f"La cartella {base_dir} non esiste! Controlla il project_name.")

    pl.seed_everything(args.seed)

    # Accumulatori Globali
    global_true = []
    global_probs = []
    
    # Nomi Classi per i Plot
    if args.num_classes == 2:
        class_labels = ['Control', 'Schizo']
    else:
        class_labels = [f'Class {i}' for i in range(args.num_classes)]

    # --- LOOP FOLDS ---
    for fold_idx in range(args.num_folds):
        print(f"\n⚡ ANALISI FOLD {fold_idx+1}/{args.num_folds}")
        
        # Cartella specifica del fold
        fold_dir = os.path.join(base_dir, f"fold_{fold_idx}")
        ckpt_path = os.path.join(fold_dir, "checkpoints", "best_fold.ckpt")
        
        # Fallback ricerca file
        if not os.path.exists(ckpt_path):
            ckpt_dir = os.path.join(fold_dir, "checkpoints")
            if os.path.exists(ckpt_dir):
                files = [f for f in os.listdir(ckpt_dir) if f.endswith('.ckpt')]
                if len(files) > 0:
                    ckpt_path = os.path.join(ckpt_dir, files[0])
                    print(f"   ⚠️ 'best_fold.ckpt' non trovato, uso: {files[0]}")
                else:
                    print(f"   ❌ Nessun checkpoint in {ckpt_dir}. Salto.")
                    continue
            else:
                print(f"   ❌ Cartella {ckpt_dir} non trovata.")
                continue
                
        print(f"   📥 Caricamento pesi: {ckpt_path}")

        # Setup DataModule
        dm = fMRIDataModule(
            use_cv=True,
            fold_index=fold_idx,
            **vars(args)
        )
        dm.setup()
        
        # Carica Modello
        model = LightningModel.load_from_checkpoint(ckpt_path, data_module=dm)
        model.cuda()
        model.eval()
        
        # --- VALIDATION (Calibrazione Soglia) ---
        val_loaders = dm.val_dataloader()
        if isinstance(val_loaders, list): val_loader = val_loaders[0]
        else: val_loader = val_loaders
        
        y_val_true, y_val_probs = get_preds(model, val_loader, args.num_classes)
        
        best_thresh = 0.5
        if args.num_classes == 2:
            fpr, tpr, thresholds = roc_curve(y_val_true, y_val_probs)
            best_thresh = thresholds[np.argmax(tpr - fpr)]
            print(f"   🎯 Soglia Calibrata (Val): {best_thresh:.4f}")
            
        # --- TEST (Predizione e Report) ---
        test_loader = dm.test_dataloader()
        y_test_true, y_test_probs = get_preds(model, test_loader, args.num_classes)
        
        if args.num_classes == 2:
            y_test_pred = (y_test_probs >= best_thresh).astype(int)
            acc = accuracy_score(y_test_true, y_test_pred)
            try: auc = roc_auc_score(y_test_true, y_test_probs)
            except: auc = 0.5
        else:
            y_test_pred = np.argmax(y_test_probs, axis=1)
            acc = accuracy_score(y_test_true, y_test_pred)
            try: auc = roc_auc_score(y_test_true, y_test_probs, multi_class='ovr')
            except: auc = 0.5
            
        print(f"   📊 Fold Score -> AUC: {auc:.4f}, ACC: {acc:.4f}")
        
        # --- SALVATAGGIO REPORT FOLD ---
        report_txt = classification_report(y_test_true, y_test_pred, target_names=class_labels, digits=4)
        report_path = os.path.join(fold_dir, f"report_fold_{fold_idx}.txt")
        with open(report_path, "w") as f:
            f.write(f"Fold {fold_idx} Analysis\n")
            f.write("="*30 + "\n")
            f.write(f"Model: {ckpt_path}\n")
            f.write(f"Threshold (from Val): {best_thresh:.4f}\n")
            f.write(f"AUC: {auc:.4f}\n")
            f.write(f"Accuracy: {acc:.4f}\n\n")
            f.write("Classification Report:\n")
            f.write(report_txt)
        print(f"   📝 Report salvato in: {report_path}")

        # --- SALVATAGGIO CM FOLD ---
        cm_path = os.path.join(fold_dir, f"confusion_matrix_fold_{fold_idx}.png")
        save_confusion_matrix(y_test_true, y_test_pred, cm_path, f"Fold {fold_idx} CM (Thresh: {best_thresh:.2f})", class_labels)
        print(f"   🖼️  CM salvata in: {cm_path}")

        # Accumulo per globale
        global_true.extend(y_test_true)
        global_probs.extend(y_test_probs)
        
        # Pulizia
        print("   🧹 Pulizia memoria GPU...")
        del model
        del dm
        gc.collect()
        torch.cuda.empty_cache()

    # ==========================================
    # REPORT GLOBALE
    # ==========================================
    if len(global_true) == 0:
        print("❌ Nessun risultato raccolto. Controlla i path.")
        return

    global_true = np.array(global_true)
    global_probs = np.array(global_probs)
    
    # Recalibrazione soglia globale
    if args.num_classes == 2:
        fpr, tpr, thresholds = roc_curve(global_true, global_probs)
        best_thresh_global = thresholds[np.argmax(tpr - fpr)]
        global_preds = (global_probs >= best_thresh_global).astype(int)
        print(f"\n🌍 Global Threshold: {best_thresh_global:.4f}")
    else:
        global_preds = np.argmax(global_probs, axis=1)
    
    # Plot Matrice Globale
    save_path = os.path.join(base_dir, "final_confusion_matrix_aggregated.png")
    save_confusion_matrix(global_true, global_preds, save_path, f"Aggregated CM ({args.num_folds} Folds)", class_labels)
    print(f"\n✅ Grafico globale salvato in: {save_path}")
    
    # Report Globale
    global_report_txt = classification_report(global_true, global_preds, target_names=class_labels, digits=4)
    global_report_path = os.path.join(base_dir, "final_report_aggregated.txt")
    with open(global_report_path, "w") as f:
        f.write(f"AGGREGATED ANALYSIS ({args.num_folds} Folds)\n")
        f.write("="*40 + "\n")
        f.write(f"Global Threshold: {best_thresh_global:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(global_report_txt)
    print(f"✅ Report globale salvato in: {global_report_path}")

if __name__ == "__main__":
    cli_main()