import os
import numpy as np
import pandas as pd
import torch
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import nibabel as nib
from tqdm import tqdm
from scipy.stats import pearsonr
from nilearn import datasets, image
from nilearn.input_data import NiftiMasker
from nilearn.image import resample_to_img
from sklearn.decomposition import PCA

# === CONFIGURAZIONE ===
DATA_ROOT = "/home/mfinocchiaro/miccai2026/NeuroSTORM/data/neurostorm_input_4d/img"
METADATA_PATH = "/home/mfinocchiaro/miccai2026/NeuroSTORM/data/neurostorm_input_4d/metadata/subject_labels.csv"
OUTPUT_FILE = "isc_results.csv"
TASK_FILTER = "movieDM"

# Setup Atlas una volta sola (Globale)
print("🧠 Setup Atlas Harvard-Oxford...")
atlas = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
keywords = ['occipital', 'calcarine', 'cuneal', 'lingual', 'fusiform']
visual_indices = [i for i, l in enumerate(atlas.labels) if any(k in str(l).lower() for k in keywords)]
atlas_img = image.load_img(atlas.maps)
mask_data_base = np.zeros_like(atlas_img.get_fdata())
for idx in visual_indices: mask_data_base[atlas_img.get_fdata() == idx] = 1
mask_img_base = image.new_img_like(atlas_img, mask_data_base)

def extract_subject_pc1(subject_id):
    """Estrae la PC1 della corteccia visiva per un soggetto."""
    subj_dir = os.path.join(DATA_ROOT, subject_id)
    if not os.path.exists(subj_dir): return None
    
    frames = sorted([f for f in os.listdir(subj_dir) if f.startswith('frame_') and f.endswith('.pt')])
    if not frames: return None
    
    # Costruiamo il volume
    # Per velocità, carichiamo e stackiamo
    try:
        # Carica primo frame per riferimento
        f0 = torch.load(os.path.join(subj_dir, frames[0]), map_location='cpu', weights_only=True).float().numpy()
        if len(f0.shape) == 4: f0 = f0.squeeze(-1)
        
        affine = np.diag([2, 2, 2, 1])
        affine[:3, 3] = -np.array(f0.shape) / 2 * 2
        
        vol_data = []
        for f in frames:
            t = torch.load(os.path.join(subj_dir, f), map_location='cpu', weights_only=True).float().numpy()
            if len(t.shape) == 4: t = t.squeeze(-1)
            vol_data.append(t)
        
        vol_4d = np.stack(vol_data, axis=-1)
        subj_img = nib.Nifti1Image(vol_4d, affine)
        
        # Resample mask
        mask_img = resample_to_img(mask_img_base, subj_img, interpolation='nearest')
        masker = NiftiMasker(mask_img=mask_img, standardize=True, detrend=True)
        
        # Extract -> (Time, Voxels)
        signals = masker.fit_transform(subj_img)
        
        # PCA -> (Time,) prendiamo solo la componente principale (il "Gist")
        pca = PCA(n_components=1)
        pc1 = pca.fit_transform(np.nan_to_num(signals)).flatten()
        
        return pc1
        
    except Exception as e:
        print(f"Error extracting {subject_id}: {e}")
        return None

def main():
    # 1. Carica Metadata
    meta = pd.read_csv(METADATA_PATH)
    # Filtra soggetti che hanno la cartella
    available_subjects = [d for d in os.listdir(DATA_ROOT) if TASK_FILTER in d]
    print(f"📂 Trovati {len(available_subjects)} cartelle soggetti.")

    # 2. Estrazione Segnali
    signals = {} # {id: time_series}
    labels = {}  # {id: 0/1}
    
    print("🚀 Estrazione segnali (PC1) in corso...")
    for folder in tqdm(available_subjects):
        # Estrai ID corto per matching col CSV (es. sub-NDAR... -> NDAR...)
        short_id = folder.split('-')[1].split('_')[0]
        
        # Cerca label
        row = meta[meta['subject_id'] == short_id]
        if row.empty: continue
        label = row.iloc[0]['label'] # 0=Control, 1=ADHD
        
        # Estrai
        sig = extract_subject_pc1(folder)
        if sig is not None:
            signals[folder] = sig
            labels[folder] = label

    # 3. Allineamento Lunghezze
    if not signals:
        print("❌ Nessun segnale estratto.")
        return

    min_len = min([len(s) for s in signals.values()])
    print(f"✂️  Taglio segnali alla lunghezza minima comune: {min_len} TR")
    
    # Matrice dati: (N_Subj, Time)
    subject_ids = list(signals.keys())
    data_matrix = np.array([signals[sid][:min_len] for sid in subject_ids])
    subj_labels = np.array([labels[sid] for sid in subject_ids])
    
    # Indici dei Controlli
    ctrl_indices = np.where(subj_labels == 0)[0]
    
    if len(ctrl_indices) < 2:
        print("❌ Troppi pochi controlli per calcolare ISC.")
        return

    # 4. Calcolo ISC
    isc_scores = []
    
    print("🧮 Calcolo ISC (vs Control Group Mean)...")
    for i in range(len(subject_ids)):
        subj_sig = data_matrix[i]
        
        # Calcolo segnale di riferimento (Mean of Controls)
        if i in ctrl_indices:
            # Leave-One-Out: Media degli altri controlli (escluso se stesso)
            others = ctrl_indices[ctrl_indices != i]
            ref_sig = np.mean(data_matrix[others], axis=0)
        else:
            # ADHD: Media di tutti i controlli
            ref_sig = np.mean(data_matrix[ctrl_indices], axis=0)
            
        # Correlazione
        r, _ = pearsonr(subj_sig, ref_sig)
        isc_scores.append(r)

    # 5. Salvataggio e Plot
    df_res = pd.DataFrame({
        'subject_id': subject_ids,
        'label': ['ADHD' if l==1 else 'Control' for l in subj_labels],
        'isc': isc_scores
    })
    
    # Ordina per ISC decrescente per trovare il "Golden Subject"
    df_sorted = df_res.sort_values('isc', ascending=False)
    print("\n🏆 TOP 5 SUBJECTS (Most Synchronized):")
    print(df_sorted.head(5))
    
    # Salva
    df_sorted.to_csv(OUTPUT_FILE, index=False)
    print(f"💾 Risultati salvati in {OUTPUT_FILE}")
    
    # Plot
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='label', y='isc', data=df_res, palette="Set2")
    sns.swarmplot(x='label', y='isc', data=df_res, color=".25", alpha=0.5)
    plt.title(f"ISC Distribution: Controls vs ADHD\n(Reference: Mean Control Signal)")
    plt.ylabel("Inter-Subject Correlation (Pearson r)")
    plt.grid(True, alpha=0.3)
    plt.savefig("isc_boxplot.png")
    print("📊 Grafico salvato in isc_boxplot.png")

if __name__ == "__main__":
    main()