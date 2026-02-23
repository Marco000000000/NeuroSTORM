import os
import glob
import numpy as np
import argparse
import torch
import nibabel as nib
from nilearn import datasets, image
from nilearn.input_data import NiftiMasker
from nilearn.image import resample_to_img
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from scipy.stats import pearsonr

# === CONFIGURAZIONE ===
DEFAULT_SUBJECT = "sub-NDARAA948VFH_task-movieDM" # Usa il soggetto che preferisci
DATA_ROOT = "/home/mfinocchiaro/miccai2026/NeuroSTORM/data/neurostorm_input_4d/img"
VIDEO_FEAT_PATH = "/home/mfinocchiaro/miccai2026/NeuroSTORM/data/neurostorm_input_4d/dataset_fmri_features/features_despicable.pt" 

HRF_LAG_TR = 13
N_VOXELS_KEEP = 2000
N_VIDEO_COMPONENTS = 10  # Prediciamo solo le prime 10 componenti principali del video!

def load_and_mask_subject(subject_id):
    subj_dir = os.path.join(DATA_ROOT, subject_id)
    if not os.path.exists(subj_dir): return None
    
    frames = sorted([f for f in os.listdir(subj_dir) if f.startswith('frame_') and f.endswith('.pt')])
    if not frames: return None
    
    first = torch.load(os.path.join(subj_dir, frames[0]), map_location='cpu').float().numpy()
    if len(first.shape) == 4: first = first.squeeze(-1)
    affine = np.diag([2, 2, 2, 1]) 
    affine[:3, 3] = -np.array(first.shape) / 2 * 2
    
    print(f"📂 Loading {len(frames)} frames...")
    vol_data = []
    for f in frames:
        t = torch.load(os.path.join(subj_dir, f), map_location='cpu').float().numpy()
        if len(t.shape) == 4: t = t.squeeze(-1)
        vol_data.append(t)
    vol_4d = np.stack(vol_data, axis=-1)
    subj_img = nib.Nifti1Image(vol_4d, affine)
    
    print("🧠 Masking (Visual Cortex)...")
    atlas = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
    keywords = ['occipital', 'calcarine', 'cuneal', 'lingual', 'fusiform']
    visual_indices = [i for i, l in enumerate(atlas.labels) if any(k in str(l).lower() for k in keywords)]
    
    atlas_img = image.load_img(atlas.maps)
    mask_data = np.zeros_like(atlas_img.get_fdata())
    for idx in visual_indices: mask_data[atlas_img.get_fdata() == idx] = 1
    
    mask_img = resample_to_img(image.new_img_like(atlas_img, mask_data), subj_img, interpolation='nearest')
    masker = NiftiMasker(mask_img=mask_img, standardize=True, detrend=True)
    features = masker.fit_transform(subj_img)
    return np.nan_to_num(features)

def prepare_data(fmri, clip):
    n = min(len(fmri), len(clip))
    X = fmri[HRF_LAG_TR:n]
    Y = clip[:n-HRF_LAG_TR]
    
    # Split 60-10-30
    n_train = int(len(X) * 0.60)
    n_val = int(len(X) * 0.70)
    
    return (X[:n_train], Y[:n_train]), (X[n_train:n_val], Y[n_train:n_val]), (X[n_val:], Y[n_val:])

def evaluate_pairwise_lowrank(pred_pcs, true_pcs):
    """Calcola Pairwise Accuracy nello spazio PCA ridotto"""
    # Normalizziamo le componenti
    pred_norm = pred_pcs / np.linalg.norm(pred_pcs, axis=1, keepdims=True)
    true_norm = true_pcs / np.linalg.norm(true_pcs, axis=1, keepdims=True)
    
    sim = np.dot(pred_norm, true_norm.T)
    n = len(sim)
    
    hits = 0
    total = 0
    for i in range(n):
        s_true = sim[i, i]
        s_others = sim[i, np.arange(n) != i]
        hits += np.sum(s_true > s_others)
        total += len(s_others)
        
    return hits / total if total > 0 else 0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=str, default=DEFAULT_SUBJECT)
    args = parser.parse_args()
    
    # 1. Caricamento
    vid_data = torch.load(VIDEO_FEAT_PATH, map_location='cpu', weights_only=True)
    vid_feat = vid_data["clip_avg"].numpy()
    fmri = load_and_mask_subject(args.subject)
    if fmri is None: return
    
    (X_train, Y_train_raw), (X_val, Y_val_raw), (X_test, Y_test_raw) = prepare_data(fmri, vid_feat)
    
    # 2. PCA sul VIDEO (Target Reduction)
    # Trasformiamo i 512 dim di CLIP in 10 componenti principali
    print(f"📉 Reducing Video Target (512 -> {N_VIDEO_COMPONENTS} PCs)...")
    pca_vid = PCA(n_components=N_VIDEO_COMPONENTS)
    Y_train = pca_vid.fit_transform(Y_train_raw)
    Y_val = pca_vid.transform(Y_val_raw)
    Y_test = pca_vid.transform(Y_test_raw)
    
    print(f"   Video Variance Explained: {np.sum(pca_vid.explained_variance_ratio_):.1%}")

    # 3. Feature Selection (Voxel)
    # Usiamo la PC1 del video per selezionare i voxel
    print(f"🕵️  Selecting top {N_VOXELS_KEEP} voxels correlated with Video PC1...")
    selector = SelectKBest(f_regression, k=N_VOXELS_KEEP)
    selector.fit(X_train, Y_train[:, 0]) # Usa solo PC1 per selezione
    
    X_train = selector.transform(X_train)
    X_val = selector.transform(X_val)
    X_test = selector.transform(X_test)
    
    # 4. Training (Ridge Regression Multi-Output)
    print("🚀 Training Ridge Regression...")
    model = RidgeCV(alphas=[100, 1000, 5000, 10000, 50000])
    model.fit(X_train, Y_train)
    print(f"   Best Alpha: {model.alpha_}")
    
    # 5. Valutazione
    preds_test = model.predict(X_test)
    
    # Metrica 1: Correlazione media per componente
    corrs = []
    for i in range(N_VIDEO_COMPONENTS):
        r, _ = pearsonr(preds_test[:, i], Y_test[:, i])
        corrs.append(r)
    
    # Metrica 2: Pairwise Accuracy nello spazio ridotto
    pair_acc = evaluate_pairwise_lowrank(preds_test, Y_test)
    
    print(f"\n🏆 RISULTATI LOW-RANK ({args.subject})")
    print(f"Pairwise Accuracy:  {pair_acc:.1%}")
    print(f"Avg Correlation:    {np.mean(corrs):.3f}")
    print("-" * 30)
    print("Correlations per PC:")
    for i, r in enumerate(corrs):
        print(f"  PC{i+1}: {r:.3f}")

if __name__ == "__main__":
    main()