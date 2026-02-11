import os
import torch
import numpy as np
import pandas as pd
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from sklearn.metrics import roc_curve
import warnings
import pytorch_lightning as pl

# --- IMPORTS MICCAI ---
from utils.cv_data_module_miccai import fMRIDataModuleMICCAI
from models.lightning_model_miccai import LightningModelMICCAI

warnings.filterwarnings("ignore")

def get_preds_safe(model, loader, num_classes):
    model.eval()
    y_true, y_probs, subjects, events = [], [], [], [] # <--- Aggiunta lista events
    
    with torch.inference_mode():
        for batch in loader:
            # --- Estrazione Dati dal Batch ---
            if isinstance(batch, dict):
                # Estrarre Subject
                subjs = batch.get('subject', batch.get('subject_name', ["unknown"]*len(batch.get('target', []))))
                
                # Estrarre Evento (Se disponibile, altrimenti 'unknown')
                evts = batch.get('event', ["unknown"]*len(subjs))
                
                # Spostare su GPU solo i tensori
                batch_gpu = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            else:
                # Fallback per loader vecchi (tuple)
                subjs = ["unknown"] * len(batch[1])
                evts = ["unknown"] * len(batch[1])
                batch_gpu = [b.cuda() if isinstance(b, torch.Tensor) else b for b in batch]
            
            subjects.extend(subjs)
            events.extend(evts) # <--- Salviamo gli eventi
            
            # --- Forward ---
            _, logits, t, _, _ = model._compute_logits(batch_gpu) 
            
            if num_classes == 2:
                # Binary
                probs = torch.sigmoid(logits).squeeze() if logits.ndim==1 or logits.shape[1]==1 else torch.softmax(logits, dim=1)[:, 1]
            else:
                # Multiclass
                probs = torch.softmax(logits, dim=1)
            
            y_true.extend(t.cpu().numpy().tolist())
            
            if isinstance(probs, torch.Tensor):
                probs = probs.cpu().numpy()
            
            if num_classes > 2:
                y_probs.extend(list(probs))
            else:
                y_probs.extend(probs.tolist())
            
    return np.array(y_true), np.array(y_probs), np.array(subjects), np.array(events)

def cli_main():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter, add_help=False)
    
    # --- ARGOMENTI PRINCIPALI ---
    parser.add_argument("--project_name", type=str, required=True)
    parser.add_argument("--fold_index", type=int, required=True)
    parser.add_argument("--num_folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1234)
    
    # --- ARGOMENTI DATASET/TASK ---
    parser.add_argument("--dataset_name", type=str, default="UCLA")
    parser.add_argument("--image_path", type=str, required=True) 
    parser.add_argument("--downstream_task_type", type=str, default="classification")
    parser.add_argument("--task_name", type=str, default="diagnosis")
    parser.add_argument("--downstream_task_id", type=int, default=1)
    
    # --- AGGIUNTA ARGOMENTI DAI MODULI ---
    parser = LightningModelMICCAI.add_model_specific_args(parser) 
    parser = fMRIDataModuleMICCAI.add_data_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    
    args = parser.parse_args()
    
    if args.model == "neurostorm": category_dir = "neurostorm"
    else: category_dir = "other"
    base_dir = os.path.join('output', category_dir, args.project_name)
    
    pl.seed_everything(args.seed)
    
    res_file = os.path.join(base_dir, f"results_fold_{args.fold_index}.csv")
    print(f"\n⚡ INFERENCE FOLD {args.fold_index}")
    
    # Trova Checkpoint
    ckpt_path = os.path.join(base_dir, f"fold_{args.fold_index}", "checkpoints", "best_fold.ckpt")
    if not os.path.exists(ckpt_path):
        ckpt_dir = os.path.join(base_dir, f"fold_{args.fold_index}", "checkpoints")
        if os.path.exists(ckpt_dir) and os.listdir(ckpt_dir):
            ckpt_path = os.path.join(ckpt_dir, os.listdir(ckpt_dir)[0])
        else: 
            print(f"❌ Checkpoint non trovato per fold {args.fold_index} in {ckpt_dir}. Salto.")
            return

    # Init DataModule (Rimuovendo fold_index dai kwargs per evitare duplicati)
    args_dict = vars(args).copy()
    if 'fold_index' in args_dict:
        del args_dict['fold_index']
        
    dm = fMRIDataModuleMICCAI(use_cv=True, fold_index=args.fold_index, **args_dict)
    dm.setup()
    
    # Load Model
    print(f"   📥 Loading: {ckpt_path}")
    model = LightningModelMICCAI.load_from_checkpoint(ckpt_path, data_module=dm, strict=False)
    model.cuda().eval()
    
    # --- TEST SET INFERENCE ---
    test_loader = dm.test_dataloader()
    # Ora recuperiamo anche gli eventi
    y_test_true, y_test_probs, test_subjects, test_events = get_preds_safe(model, test_loader, args.num_classes)
    
    best_thresh = 0.5
    
    # --- CALCOLO SOGLIA (Solo Binary) ---
    if args.num_classes == 2:
        val_loader = dm.val_dataloader()
        if isinstance(val_loader, list): val_loader = val_loader[0]
        
        # Ignoriamo gli eventi del validation set (useremo _ come placeholder)
        y_val_true, y_val_probs, _, _ = get_preds_safe(model, val_loader, args.num_classes)
        
        if len(np.unique(y_val_true)) > 1:
            fpr, tpr, thresholds = roc_curve(y_val_true, y_val_probs)
            best_thresh = thresholds[np.argmax(tpr - fpr)]
            print(f"   🎯 Best Val Thresh: {best_thresh:.4f}")
        else:
            print("   ⚠️ Val set mono-classe, uso thresh=0.5")

    # --- SALVATAGGIO CSV con COLONNA EVENT ---
    probs_to_save = y_test_probs
    if args.num_classes > 2:
        probs_to_save = [list(p) for p in y_test_probs]

    df = pd.DataFrame({
        'fold': args.fold_index, 
        'subject': test_subjects, 
        'event': test_events,  # <--- NUOVA COLONNA
        'true': y_test_true,
        'prob': probs_to_save, 
        'thresh': best_thresh
    })
    
    os.makedirs(os.path.dirname(res_file), exist_ok=True)
    df.to_csv(res_file, index=False)
    
    print(f"   ✅ Saved with EVENTS: {res_file}")

if __name__ == "__main__":
    cli_main()