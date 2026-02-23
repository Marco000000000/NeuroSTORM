import os
import torch
import numpy as np
import nibabel as nib
from nilearn import datasets, image, masking
from nilearn.image import resample_to_img

# === CONFIGURAZIONE ===
# Sostituisci con il soggetto che stai usando
SUBJECT_ID = "sub-NDARAA948VFH_task-movieDM" 
DATA_ROOT = "/home/mfinocchiaro/miccai2026/NeuroSTORM/data/neurostorm_input_4d/img"
DEBUG_DIR = "debug_output"

def check_alignment():
    os.makedirs(DEBUG_DIR, exist_ok=True)
    subj_dir = os.path.join(DATA_ROOT, SUBJECT_ID)
    
    # 1. Carica il primo frame
    frames = sorted([f for f in os.listdir(subj_dir) if f.startswith('frame_') and f.endswith('.pt')])
    if not frames:
        print("❌ Nessun frame trovato.")
        return

    print(f"📂 Carico frame di riferimento: {frames[0]}")
    # Carichiamo il tensore
    t = torch.load(os.path.join(subj_dir, frames[0]), map_location='cpu').float().numpy()
    if len(t.shape) == 4: t = t.squeeze(-1) # (96, 96, 96)
    
    # 2. Creiamo l'affine "fittizia" (quella che abbiamo usato finora)
    # Se i dati sono già 2mm MNI, questo dovrebbe essere ok, ma verifichiamo.
    affine = np.diag([2, 2, 2, 1]) 
    # Centriamo l'immagine nello spazio
    affine[:3, 3] = -np.array(t.shape) / 2 * 2 
    
    subj_img = nib.Nifti1Image(t, affine)
    
    # Salva l'immagine del soggetto per controllo visivo
    subj_out_path = os.path.join(DEBUG_DIR, "subject_reference.nii.gz")
    nib.save(subj_img, subj_out_path)
    print(f"💾 Salvato riferimento soggetto: {subj_out_path}")

    # 3. Carica Atlas e crea Maschera
    print("🧠 Carico Atlas Harvard-Oxford...")
    atlas = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
    
    # Indici visivi
    visual_indices = [i for i, l in enumerate(atlas.labels) if any(k in str(l).lower() for k in ['visual', 'occipital', 'lingual', 'fusiform'])]
    print(f"   -> Trovate {len(visual_indices)} regioni visive.")
    
    atlas_img = image.load_img(atlas.maps)
    atlas_data = atlas_img.get_fdata()
    mask_data = np.zeros_like(atlas_data)
    for idx in visual_indices: mask_data[atlas_data == idx] = 1
    
    mask_img_orig = image.new_img_like(atlas_img, mask_data)
    
    # 4. Resampling Maschera sul Soggetto
    print("🔄 Resampling maschera...")
    mask_img_res = resample_to_img(mask_img_orig, subj_img, interpolation='nearest')
    
    # Salva la maschera per controllo visivo
    mask_out_path = os.path.join(DEBUG_DIR, "visual_mask_resampled.nii.gz")
    nib.save(mask_img_res, mask_out_path)
    print(f"💾 Salvata maschera: {mask_out_path}")
    
    # 5. DIAGNOSTICA NUMERICA
    # Calcoliamo l'overlap
    subj_data = subj_img.get_fdata()
    mask_data_res = mask_img_res.get_fdata()
    
    # Quanti voxel della maschera cadono su voxel del cervello che non sono zero?
    # Assumiamo che 0 sia sfondo.
    brain_voxels = (np.abs(subj_data) > 1e-6) # Voxel con segnale
    mask_voxels = (mask_data_res > 0)         # Voxel della maschera
    
    overlap = np.logical_and(brain_voxels, mask_voxels).sum()
    mask_total = mask_voxels.sum()
    
    print("\n📊 --- REPORT DIAGNOSTICO ---")
    print(f"Volume Maschera (voxel totali): {mask_total}")
    print(f"Overlap (Maschera su Cervello): {overlap}")
    print(f"Percentuale Copertura: {overlap / mask_total * 100:.2f}%")
    
    if overlap == 0:
        print("❌ DISASTRO: L'overlap è ZERO. La maschera è completamente fuori dal cervello!")
    elif overlap / mask_total < 0.1:
        print("⚠️  ALLARME: Overlap < 10%. Probabile disallineamento grave.")
    else:
        print("✅ Overlap geometrico sembra plausibile.")

    # 6. Analisi del Segnale (se c'è overlap)
    if overlap > 0:
        # Estraiamo i valori sotto la maschera
        signal_values = subj_data[mask_voxels]
        print(f"\nStats Segnale (sotto maschera):")
        print(f"  Mean: {np.mean(signal_values):.4f}")
        print(f"  Std : {np.std(signal_values):.4f}")
        print(f"  Min : {np.min(signal_values):.4f}")
        print(f"  Max : {np.max(signal_values):.4f}")
        
        if np.std(signal_values) < 1e-5:
            print("⚠️  Segnale PIATTO (Std ~ 0). I dati potrebbero essere corrotti.")

if __name__ == "__main__":
    check_alignment()