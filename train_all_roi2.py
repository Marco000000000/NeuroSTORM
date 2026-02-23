import os
import numpy as np
import pandas as pd
import argparse
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import nibabel as nib
from tqdm import tqdm
from nilearn import datasets, image
from nilearn.input_data import NiftiMasker
from nilearn.image import resample_to_img
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV

# === CONFIGURAZIONE ===
DATA_ROOT = "/home/mfinocchiaro/miccai2026/NeuroSTORM/data/neurostorm_input_4d/img"
VIDEO_FEAT_PATH = "/home/mfinocchiaro/miccai2026/NeuroSTORM/data/neurostorm_input_4d/dataset_fmri_features/features_despicable.pt" 
METADATA_PATH = "/home/mfinocchiaro/miccai2026/NeuroSTORM/data/neurostorm_input_4d/metadata/subject_labels.csv"
OUTPUT_FILE = "final_decoding_results.csv"
TASK_FILTER = "movieDM"

# Parametri Vincenti
HRF_LAG_TR = 13
N_VOXELS_KEEP = 2000
N_VIDEO_PCS = 10 

# --- SETUP ATLAS (Una volta sola per velocità) ---
print("🧠 Inizializzazione Atlas...")
atlas = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
keywords = ['occipital', 'calcarine', 'cuneal', 'lingual', 'fusiform']
vis_indices = [i for i, l in enumerate(atlas.labels) if any(k in str(l).lower() for k in keywords)]
atlas_img = image.load_img(atlas.maps)
mask_data_base = np.zeros_like(atlas_img.get_fdata())
for idx in vis_indices: mask_data_base[atlas_img.get_fdata() == idx] = 1
mask_img_base = image.new_img_like(atlas_img, mask_data_base)

def process_subject(subject_id, vid_feat_pca):
    """Esegue la pipeline Low-Rank su un soggetto e ritorna l'accuratezza."""
    subj_dir = os.path.join(DATA_ROOT, subject_id)
    frames = sorted([f for f in os.listdir(subj_dir) if f.startswith('frame_') and f.endswith('.pt')])
    if len(frames) < 100: return None # Salta soggetti corrotti
    
    try:
        # 1. Load fMRI
        first = torch.load(os.path.join(subj_dir, frames[0]), map_location='cpu', weights_only=True).float().numpy()
        if len(first.shape) == 4: first = first.squeeze(-1)
        affine = np.diag([2, 2, 2, 1])
        affine[:3, 3] = -np.array(first.shape) / 2 * 2
        
        vol_data = [torch.load(os.path.join(subj_dir, f), map_location='cpu', weights_only=True).float().numpy().squeeze() for f in frames]
        vol_4d = np.stack(vol_data, axis=-1)
        subj_img = nib.Nifti1Image(vol_4d, affine)
        
        # 2. Masking
        mask_img = resample_to_img(mask_img_base, subj_img, interpolation='nearest')
        masker = NiftiMasker(mask_img=mask_img, standardize=True, detrend=True)
        fmri_data = np.nan_to_num(masker.fit_transform(subj_img))
        
        # 3. Sync & Split
        n = min(len(fmri_data), len(vid_feat_pca))
        X = fmri_data[HRF_LAG_TR:n]
        Y = vid_feat_pca[:n-HRF_LAG_TR]
        
        n_train = int(len(X) * 0.60)
        n_val = int(len(X) * 0.70)
        X_train, Y_train = X[:n_train], Y[:n_train]
        X_test, Y_test = X[n_val:], Y[n_val:]
        
        # 4. Feature Selection (Top Voxel su Train)
        selector = SelectKBest(f_regression, k=N_VOXELS_KEEP)
        # Usa PC1 video come target per selezione
        selector.fit(X_train, Y_train[:, 0]) 
        X_train = selector.transform(X_train)
        X_test = selector.transform(X_test)
        
        # 5. Ridge Regression
        model = RidgeCV(alphas=[100, 1000, 10000])
        model.fit(X_train, Y_train)
        preds = model.predict(X_test)
        
        # 6. Pairwise Accuracy
        # Normalize
        p_norm = preds / np.linalg.norm(preds, axis=1, keepdims=True)
        t_norm = Y_test / np.linalg.norm(Y_test, axis=1, keepdims=True)
        sim = np.dot(p_norm, t_norm.T)
        
        hits = 0
        n_test = len(sim)
        for i in range(n_test):
            s_true = sim[i, i]
            s_others = sim[i, np.arange(n_test) != i]
            hits += np.sum(s_true > s_others)
            
        acc = hits / (n_test * (n_test - 1))
        return acc

    except Exception as e:
        print(f"⚠️ Error {subject_id}: {e}")
        return None

def main():
    # Load Metadata
    meta = pd.read_csv(METADATA_PATH)
    subjects = [d for d in os.listdir(DATA_ROOT) if TASK_FILTER in d]
    print(f"📂 Trovati {len(subjects)} soggetti.")
    
    # Load Video PCA (Target)
    vid_data = torch.load(VIDEO_FEAT_PATH, map_location='cpu', weights_only=True)["clip_avg"].numpy()
    pca_vid = PCA(n_components=N_VIDEO_PCS)
    vid_feat_pca = pca_vid.fit_transform(vid_data)
    
    results = []
    
    print("🚀 Avvio Batch Decoding...")
    for sub in tqdm(subjects):
        short_id = sub.split('-')[1].split('_')[0]
        # Trova label
        row = meta[meta['subject_id'] == short_id]
        if row.empty: continue
        label = row.iloc[0]['label']
        
        acc = process_subject(sub, vid_feat_pca)
        
        if acc is not None:
            results.append({'subject': sub, 'label': label, 'accuracy': acc})
            # Salva parziale ogni 5 soggetti
            if len(results) % 5 == 0:
                pd.DataFrame(results).to_csv(OUTPUT_FILE, index=False)

    # Plot Finale
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_FILE, index=False)
    
    plt.figure(figsize=(10, 6))
    # Mapping label: 0->Control, 1->ADHD
    df['Group'] = df['label'].map({0: 'Control', 1: 'ADHD'})
    
    sns.violinplot(x='Group', y='accuracy', data=df, inner='quartile', palette='muted')
    sns.swarmplot(x='Group', y='accuracy', data=df, color='k', alpha=0.5)
    plt.axhline(0.5, color='r', linestyle='--', label='Chance Level')
    plt.title("Neural Decoding Accuracy: ADHD vs Controls")
    plt.ylabel("Pairwise Accuracy")
    plt.savefig("final_decoding_plot.png")
    print("\n✅ Finito! Risultati in final_decoding_results.csv e Plot PNG.")

if __name__ == "__main__":
    main()