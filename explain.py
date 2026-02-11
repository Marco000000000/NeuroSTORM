import torch
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nb
import os
from models.lightning_model import LightningModel
from utils.data_module import fMRIDataModule
from argparse import Namespace
from captum.attr import IntegratedGradients, Saliency, InputXGradient

# ================= CONFIGURAZIONE =================
CHECKPOINT_PATH = "./output/neurostorm/ucla_ft_neurostorm_task3_dx_train1.0/last.ckpt"


# Vuoi nascondere il rumore fuori dal cervello? (True consigliato)
APPLY_BRAIN_MASK = True 

# Soglia per pulire la mappa (mostra solo il top X% delle attivazioni)
PERCENTILE_THRESHOLD = 95 # Mostra solo il 5% più forte (più alto = più pulito)

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
    eval_batch_size=1,
)

def explain(method='IG'):
    print("🧠 Caricamento Dati...")
    dm = fMRIDataModule(**vars(hparams))
    dm.setup()
    test_loader = dm.test_dataloader()
    
    # Trova uno Schizofrenico (Target=1)
    target_subject = None
    target_data = None
    
    print("🔍 Ricerca paziente Schizofrenico nel Test Set...")
    for batch in test_loader:
        fmri = batch['fmri_sequence']
        target = batch['target']
        subj_id = batch['subject_name']
        
        if target.item() == 1: 
            target_subject = subj_id[0]
            target_data = fmri
            print(f"🎯 Trovato paziente: {target_subject}")
            break
    
    if target_data is None:
        print("❌ Nessuno schizofrenico trovato.")
        return

    print(f"📦 Caricamento modello da: {CHECKPOINT_PATH}")
    model = LightningModel.load_from_checkpoint(CHECKPOINT_PATH, data_module=dm)
    model.eval()
    model.cuda()
    target_data = target_data.cuda().requires_grad_()

    # Wrapper function
    def forward_func(inputs):
        logits = model.model(inputs)
        output = model.output_head(logits)
        if output.shape[1] == 1: return torch.sigmoid(output)
        else: return torch.softmax(output, dim=1)[:, 1].unsqueeze(1)

    # --- SELEZIONE METODO XAI ---
    print(f"🔦 Calcolo mappa con metodo: {method}...")
    
    if method == 'IG':
        explainer = IntegratedGradients(forward_func)
        attributions = explainer.attribute(target_data, target=0, n_steps=20) # n_steps basso per velocità
    elif method == 'Saliency':
        explainer = Saliency(forward_func)
        attributions = explainer.attribute(target_data, target=0)
    elif method == 'InputXGradient':
        explainer = InputXGradient(forward_func)
        attributions = explainer.attribute(target_data, target=0)
    else:
        raise ValueError("Metodo non riconosciuto")
    
    # Post-processing
    attr_np = attributions.squeeze().detach().cpu().numpy()
    original_img = target_data.squeeze().detach().cpu().numpy()
    
    if len(attr_np.shape) == 5: attr_np = attr_np[0]
    if len(original_img.shape) == 5: original_img = original_img[0]

    # Collasso temporale (Media) -> [H, W, D]
    saliency_map = np.mean(np.abs(attr_np), axis=-1)
    
    # --- CLEANING (Brain Masking) ---
    if APPLY_BRAIN_MASK:
        # Creiamo una maschera basata sull'intensità dell'immagine originale
        # Prendiamo un frame centrale
        anatomy_ref = np.mean(original_img, axis=-1)
        # Soglia empirica: se l'anatomia è nera (< 0.05), spegni la saliency
        mask = anatomy_ref > 0.05 
        
        print("🧹 Applicazione Brain Mask per rimuovere rumore di fondo...")
        saliency_map = saliency_map * mask

    # --- THRESHOLDING ---
    # Mantiene solo i valori più alti per pulire la visualizzazione
    threshold_val = np.percentile(saliency_map[saliency_map > 0], PERCENTILE_THRESHOLD)
    saliency_map[saliency_map < threshold_val] = 0
    
    print("🎨 Generazione Mosaico...")
    
    # Selezioniamo 9 slice significative lungo l'asse Z
    n_slices = original_img.shape[2]
    slice_indices = np.linspace(20, n_slices-20, 9).astype(int) # Salta le estremità vuote
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()
    
    for i, idx in enumerate(slice_indices):
        # Anatomia (Sfondo)
        bg = original_img[:, :, idx, 10] # Frame temporale 10
        # Saliency (Overlay)
        sal = saliency_map[:, :, idx]
        
        axes[i].imshow(np.rot90(bg), cmap='gray')
        # Mostra saliency solo se > 0
        if np.max(sal) > 0:
            axes[i].imshow(np.rot90(sal), cmap='hot', alpha=0.6, vmin=0)
        
        axes[i].set_title(f"Slice Z={idx}")
        axes[i].axis('off')
    
    plt.suptitle(f"XAI ({method}): Subject {target_subject}", fontsize=16)
    plt.tight_layout()
    save_png = f'mosaic_{method}_{target_subject}.png'
    plt.savefig(save_png)
    print(f"✅ Mosaico salvato: {save_png}")
    
    # --- SALVATAGGIO NIFTI 3D ---
    # Salva la mappa 3D reale per aprirla con visualizzatori professionali
    save_nii = f'saliency_{method}_{target_subject}.nii.gz'
    # Usa matrice identità come affine base se non abbiamo l'originale
    # nifti_img = nb.Nifti1Image(saliency_map, affine=np.eye(4))
    # nb.save(nifti_img, save_nii)
    print(f"✅ Volume 3D salvato: {save_nii} (Aprilo con ITK-SNAP o FSLeyes!)")

if __name__ == "__main__":
    for method in ['IG', 'Saliency', 'InputXGradient']:
        explain(method=method)
