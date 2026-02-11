import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score, roc_curve
from models.lightning_model import LightningModel
from utils.data_module import fMRIDataModule
from argparse import Namespace

# ================= CONFIGURAZIONE =================
# ⚠️ INSERISCI QUI IL PATH DEL TUO CHECKPOINT MIGLIORE
CHECKPOINT_PATH = "output/neurostorm/ucla_ft_neurostorm_task3_dx_train1.0/last.ckpt" 
# (Oppure cerca quello con 'epoch=XX.ckpt')

# Parametri usati nel training (DEVONO ESSERE IDENTICI)
hparams = Namespace(
    dataset_name='UCLA',
    image_path='./data/UCLA_MNI_to_TRs_minmax',
    batch_size=8,
    num_workers=4,
    dataset_split_num=2,     # <--- FONDAMENTALE: Usa lo stesso numero dello script bash!
    seed=1,
    train_split=0.8,
    val_split=0.1,
    limit_training_samples=1.0,
    downstream_task_type='classification',
    num_classes=2,           # Binario
    task_name='diagnosis',
    sequence_length=20,
    img_size=[96, 96, 96, 20],
    use_contrastive=False,
    use_mae=False,
    stride_between_seq=1,
    stride_within_seq=1,
    with_voxel_norm=False,
    shuffle_time_sequence=False,
    label_scaling_method='standardization',
    downstream_task_id=3,
    contrastive_type=0,
    bad_subj_path=None,
    strategy='ddp',
    pretraining =False,
    eval_batch_size=4,
)

def get_predictions(model, loader):
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            subj, logits, target, _ = model._compute_logits(batch)
            
            # Gestione dimensioni output (1 o 2 neuroni)
            if logits.ndim == 1 or (logits.ndim == 2 and logits.shape[1] == 1):
                probs = torch.sigmoid(logits)
                if probs.ndim == 2: probs = probs.squeeze()
            else:
                probs = torch.softmax(logits, dim=1)[:, 1]
            
            all_targets.extend(target.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
    return np.array(all_targets), np.array(all_probs)

def evaluate_rigorous():
    print("🚀 Caricamento Dati...")
    dm = fMRIDataModule(**vars(hparams))
    dm.setup()
    
    # Carichiamo ENTRAMBI i loader
    val_loader = dm.val_dataloader()[0] # [0] perché val_dataloader restituisce [val, test]
    test_loader = dm.test_dataloader()

    print(f"📦 Validation Set: {len(dm.val_dataset)} soggetti")
    print(f"📦 Test Set:       {len(dm.test_dataset)} soggetti")

    print(f"🧠 Caricamento Modello...")
    model = LightningModel.load_from_checkpoint(CHECKPOINT_PATH, data_module=dm)
    model.cuda()

    # --- FASE 1: CALIBRAZIONE SU VALIDATION ---
    print("\n🔍 FASE 1: Calibrazione Soglia su VALIDATION SET")
    y_val, probs_val = get_predictions(model, val_loader)
    
    fpr, tpr, thresholds = roc_curve(y_val, probs_val)
    # Youden's J statistic
    J = tpr - fpr
    ix = np.argmax(J)
    best_thresh = thresholds[ix]
    
    print(f"   👉 Soglia Ottimale Trovata: {best_thresh:.4f}")
    print(f"   (Performance attese su Val: Sens={tpr[ix]:.2f}, Spec={1-fpr[ix]:.2f})")

    # --- FASE 2: APPLICAZIONE SU TEST ---
    print("\n⚖️  FASE 2: Valutazione Rigorosa su TEST SET")
    y_test, probs_test = get_predictions(model, test_loader)
    
    # Applichiamo la soglia del VALIDATION ai dati del TEST
    preds_test = (probs_test >= best_thresh).astype(int)
    
    # Metriche
    cm = confusion_matrix(y_test, preds_test)
    acc = accuracy_score(y_test, preds_test)
    try:
        auroc = roc_auc_score(y_test, probs_test)
    except: auroc = 0.5

    print("\n" + "="*40)
    print(f"📊 RISULTATI FINALI (Threshold trasferita: {best_thresh:.4f})")
    print("="*40)
    print(f"ACCURACY: {acc:.4f}")
    print(f"AUROC:    {auroc:.4f}")
    print("-" * 40)
    print("Classification Report:")
    print(classification_report(y_test, preds_test, target_names=['Control', 'Schizo']))

    # Plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
                xticklabels=['Pred: Control', 'Pred: Schizo'],
                yticklabels=['True: Control', 'True: Schizo'])
    plt.title(f'Test Matrix (Thresh from Val: {best_thresh:.2f})')
    plt.ylabel('Ground Truth')
    plt.xlabel('Prediction')
    plt.savefig('confusion_matrix_rigorous.png')
    print("\n✅ Matrice salvata: confusion_matrix_rigorous.png")

if __name__ == "__main__":
    evaluate_rigorous()