import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import gc
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, roc_curve, classification_report
import warnings

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

warnings.filterwarnings("ignore")

from utils.cv_data_module import fMRIDataModule
from models.lightning_model import LightningModel
import pytorch_lightning as pl

def get_preds_safe(model, loader, num_classes):
    """Estrae predizioni in modo efficiente per la memoria"""
    model.eval()
    y_true = []
    y_probs = []
    subjects = []
    
    with torch.inference_mode():
        for batch in loader:
            # Estrai ID
            if isinstance(batch, dict):
                subjs = batch.get('subject', batch.get('subject_name', ["unknown"]*len(batch['target'])))
                batch_gpu = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            else:
                subjs = ["unknown"] * len(batch[1])
                batch_gpu = [b.cuda() if isinstance(b, torch.Tensor) else b for b in batch]

            subjects.extend(subjs)

            # Forward
            _, logits, t, _ = model._compute_logits(batch_gpu)
            
            # Prob
            if num_classes == 2:
                probs = torch.sigmoid(logits).squeeze() if logits.ndim==1 or logits.shape[1]==1 else torch.softmax(logits, dim=1)[:, 1]
            else:
                probs = torch.softmax(logits, dim=1)
            
            y_true.extend(t.cpu().numpy().tolist())
            y_probs.extend(probs.cpu().numpy().tolist())
            
    return np.array(y_true), np.array(y_probs), np.array(subjects)

def cli_main():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter, add_help=False)
    parser.add_argument("--downstream_task_id", type=int, default="1", help="downstream task id")
    parser.add_argument("--task_name", type=str, default="d", help="specify the task name")
    parser.add_argument("--downstream_task_type", type=str, default="classification", help="select either classification or regression")
    parser.add_argument("--single_fold_index", type=int, default=-1, help="Esegui solo questo fold (per risparmiare RAM)")
    parser.add_argument("--project_name", type=str, required=True, help="Nome cartella output")
    parser.add_argument("--num_folds", type=int, default=5, help="Numero di fold CV")
    parser.add_argument("--seed", type=int, default=1234, help="Seed globale")
    parser.add_argument("--dataset_name", type=str, default="UCLA")
    
    parser = LightningModel.add_model_specific_args(parser) 
    parser = pl.Trainer.add_argparse_args(parser)
    parser = fMRIDataModule.add_data_specific_args(parser)

    args = parser.parse_args()
    
    if args.model == "neurostorm": category_dir = "neurostorm"
    else: category_dir = "other"
    base_dir = os.path.join('output', category_dir, args.project_name)
    
    pl.seed_everything(args.seed)

    res_file = os.path.join(base_dir, "results_progressive.csv")
    if os.path.exists(res_file): os.remove(res_file)
    
    print(f"📂 Output Dir: {base_dir}")
    # Logica Loop
    if args.single_fold_index != -1:
        # Esegui SOLO il fold specificato
        folds_to_run = [args.single_fold_index]
        print(f"🔒 Modalità Processo Singolo: Eseguo solo Fold {args.single_fold_index}")
    else:
        # Esegui tutti i fold (comportamento vecchio)
        folds_to_run = range(args.num_folds)
    for fold_idx in folds_to_run:
        print(f"\n⚡ FOLD {fold_idx+1}/{args.num_folds}")
        
        # Checkpoint
        ckpt_dir = os.path.join(base_dir, f"fold_{fold_idx}", "checkpoints")
        if not os.path.exists(ckpt_dir): 
            print("❌ No Checkpoint Dir"); continue
            
        files = [f for f in os.listdir(ckpt_dir) if f.endswith('.ckpt')]
        if not files: 
            print("❌ No Checkpoint File"); continue
            
        ckpt_path = os.path.join(ckpt_dir, files[0])
        print(f"   📥 Load: {files[0]}")

        # Data & Model
        dm = fMRIDataModule(use_cv=True, fold_index=fold_idx, **vars(args))
        dm.setup()
        
        model = LightningModel.load_from_checkpoint(ckpt_path, data_module=dm)
        model.cuda().eval()
        
        # --- FIX 1: Usa val_dataloader invece di train_dataloader ---
        val_loaders = dm.train_dataloader()
        if isinstance(val_loaders, list): 
            val_loader = val_loaders[0] # Il primo è il validation set
        else: 
            val_loader = val_loaders
            
        y_val_true, y_val_probs, _ = get_preds_safe(model, val_loader, args.num_classes)
        
        best_thresh = 0.5
        if args.num_classes == 2:
            fpr, tpr, thresholds = roc_curve(y_val_true, y_val_probs)
            best_thresh = thresholds[np.argmax(tpr - fpr)]
            print(f"   🎯 Thresh (Val): {best_thresh:.4f}")
            
        # 2. Test
        test_loader = dm.test_dataloader()
        y_test_true, y_test_probs, test_subjects = get_preds_safe(model, test_loader, args.num_classes)
        
        y_test_pred = (y_test_probs >= best_thresh).astype(int) if args.num_classes == 2 else np.argmax(y_test_probs, axis=1)
        
        acc = accuracy_score(y_test_true, y_test_pred)
        try: auc = roc_auc_score(y_test_true, y_test_probs)
        except: auc = 0.5
        
        # Salva predizioni
        df_fold = pd.DataFrame({
            'fold': fold_idx,
            'subject': test_subjects,
            'true': y_test_true,
            'prob': y_test_probs,
            'thresh': best_thresh
        })
        df_fold.to_csv(res_file, mode='a', header=not os.path.exists(res_file), index=False)
        print(f"   💾 Dati salvati in {res_file}")
        
        # --- FIX 3: Crea cartella fold se non esiste ---
        fold_dir = os.path.join(base_dir, f"fold_{fold_idx}")
        os.makedirs(fold_dir, exist_ok=True) 

        # --- FIX 2: Passa y_test_pred (classi) invece di y_test_probs (probabilità) ---
        if args.num_classes == 2:
            class_labels = ['Control', 'Schizo']
        else:
            class_labels = [str(i) for i in range(args.num_classes)]
            
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

        cm_path = os.path.join(fold_dir, f"confusion_matrix_fold_{fold_idx}.png")
        save_confusion_matrix(y_test_true, y_test_pred, cm_path, f"Fold {fold_idx} CM (Thresh: {best_thresh:.2f})", class_labels)
        print(f"   🖼️  CM salvata in: {cm_path}")
        
        # Clean
        del model, dm, y_test_true, y_test_probs, y_val_true, y_val_probs
        gc.collect()
        torch.cuda.empty_cache()

    print("\n✅ Finito! Ora calcolo metriche aggregate...")
    analyze_results(res_file, args.num_classes, base_dir)

def analyze_results(csv_path, num_classes, output_dir):
    """Legge il CSV accumulato e calcola metriche finali"""
    df = pd.read_csv(csv_path)
    
    # --- EVENT LEVEL ---
    print("\n🌍 GLOBAL EVENT LEVEL")
    df['pred'] = df.apply(lambda row: 1 if row['prob'] >= row['thresh'] else 0, axis=1)
    
    acc = accuracy_score(df['true'], df['pred'])
    auc = roc_auc_score(df['true'], df['prob'])
    print(f"   AUC: {auc:.4f} | ACC: {acc:.4f}")
    print(classification_report(df['true'], df['pred'], digits=4))

    # --- SUBJECT LEVEL ---
    print("\n👤 GLOBAL SUBJECT LEVEL")
    subj_df = df.groupby('subject').agg({'true':'first', 'prob':'mean', 'thresh':'mean'}).reset_index()
    subj_df['pred'] = subj_df.apply(lambda row: 1 if row['prob'] >= row['thresh'] else 0, axis=1)
    
    acc_subj = accuracy_score(subj_df['true'], subj_df['pred'])
    auc_subj = roc_auc_score(subj_df['true'], subj_df['prob'])
    
    print(f"   AUC: {auc_subj:.4f} | ACC: {acc_subj:.4f}")
    print(classification_report(subj_df['true'], subj_df['pred'], digits=4))
    
    # Salva report
    with open(os.path.join(output_dir, "final_metrics.txt"), "w") as f:
        f.write(f"Event Level AUC: {auc:.4f}, ACC: {acc:.4f}\n")
        f.write(f"Subject Level AUC: {auc_subj:.4f}, ACC: {acc_subj:.4f}\n")

if __name__ == "__main__":
    cli_main()