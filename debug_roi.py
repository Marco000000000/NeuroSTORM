import os
import torch
import numpy as np
import argparse
import matplotlib.pyplot as plt
import nibabel as nib
from nilearn import datasets, image
from nilearn.input_data import NiftiMasker
from nilearn.image import resample_to_img
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from scipy.special import gamma # <--- CORREZIONE QUI (prima era scipy.stats)

# === CONFIGURAZIONE ===
DATA_ROOT = "/home/mfinocchiaro/miccai2026/NeuroSTORM/data/neurostorm_input_4d/img"
VIDEO_FEAT_PATH = "/home/mfinocchiaro/miccai2026/NeuroSTORM/data/neurostorm_input_4d/dataset_fmri_features/features_despicable.pt" 
TR = 0.8

def glover_hrf(tr, oversampling=16, time_length=32, onset=0):
    """
    Funzione Emodinamica Canonica (HRF) per convoluzione.
    """
    dt = tr / oversampling
    t = np.linspace(0, time_length, int(time_length / dt))
    
    # Parametri Glover (spm)
    peak1 = 6; undershoot = 16; disp1 = 1; disp2 = 1; ratio = 6
    
    # Ora 'gamma' è la funzione matematica corretta
    hrf = (t ** (peak1 - 1) * np.exp(-t / disp1) / (disp1 ** peak1 * gamma(peak1)) -
           1 / ratio * t ** (undershoot - 1) * np.exp(-t / disp2) / (disp2 ** undershoot * gamma(undershoot)))
    
    hrf = hrf / np.max(hrf)
    return hrf

def convolve_hrf(signal, tr):
    """Convolve il segnale video con l'HRF per simulare cosa vede la fMRI"""
    hrf = glover_hrf(tr, oversampling=1) # 1pt per TR per semplicità
    n_sig = len(signal)
    # Full convolution, poi tagliamo all'inizio (o fine? solitamente si taglia per mantenere lunghezza)
    convolved = np.convolve(signal, hrf)[:n_sig]
    return convolved

def load_and_mask_subject(subject_id):
    subj_dir = os.path.join(DATA_ROOT, subject_id)
    if not os.path.exists(subj_dir):
        print(f"❌ Directory {subj_dir} non trovata.")
        return None
        
    frames = sorted([f for f in os.listdir(subj_dir) if f.startswith('frame_') and f.endswith('.pt')])
    if not frames: return None

    print(f"📂 Loading {len(frames)} frames...")
    
    # Geometria
    first = torch.load(os.path.join(subj_dir, frames[0]), map_location='cpu').float().numpy()
    if len(first.shape) == 4: first = first.squeeze(-1)
    affine = np.diag([2, 2, 2, 1])
    affine[:3, 3] = -np.array(first.shape) / 2 * 2
    
    vol_data = []
    for f in frames:
        t = torch.load(os.path.join(subj_dir, f), map_location='cpu').float().numpy()
        if len(t.shape) == 4: t = t.squeeze(-1)
        vol_data.append(t)
    vol_4d = np.stack(vol_data, axis=-1)
    subj_img = nib.Nifti1Image(vol_4d, affine)
    
    print("🧠 Masking (Visual Cortex)...")
    atlas = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
    # Keywords anatomiche sicure
    keywords = ['occipital', 'calcarine', 'cuneal', 'lingual', 'fusiform']
    indices = [i for i, l in enumerate(atlas.labels) if any(k in str(l).lower() for k in keywords)]
    
    atlas_img = image.load_img(atlas.maps)
    mask_data = np.zeros_like(atlas_img.get_fdata())
    for idx in indices: mask_data[atlas_img.get_fdata() == idx] = 1
    
    mask_img = resample_to_img(image.new_img_like(atlas_img, mask_data), subj_img, interpolation='nearest')
    masker = NiftiMasker(mask_img=mask_img, standardize=True, detrend=True)
    features = masker.fit_transform(subj_img)
    
    return np.nan_to_num(features)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=str, required=True)
    args = parser.parse_args()
    
    # 1. Load Data
    fmri = load_and_mask_subject(args.subject) # (Time, Voxels)
    if fmri is None: return

    vid_data = torch.load(VIDEO_FEAT_PATH, map_location='cpu', weights_only=True)
    vid_feat = vid_data["clip_avg"].numpy()
    
    # 2. Extract Video PC1 (The "Gist")
    print("🎬 Extracting Video PC1 & Convolving with HRF...")
    pca_vid = PCA(n_components=1)
    vid_pc1 = pca_vid.fit_transform(vid_feat).flatten()
    
    # CONVOLUZIONE HRF
    vid_pc1_hrf = convolve_hrf(vid_pc1, TR)
    
    # Align lengths
    n = min(len(fmri), len(vid_pc1_hrf))
    X = fmri[:n]
    Y = vid_pc1_hrf[:n]
    
    # Split
    n_train = int(n * 0.60) 
    X_train, Y_train = X[:n_train], Y[:n_train]
    X_test, Y_test = X[n_train:], Y[n_train:]
    
    print(f"📊 Train: {len(X_train)} samples | Test: {len(X_test)} samples")
    
    # 3. Ridge Regression (Predict PC1 from Brain)
    print("🚀 Training Ridge Regression (Brain -> Video PC1)...")
    # Alphas standard
    model = RidgeCV(alphas=[10, 100, 1000, 10000, 50000])
    model.fit(X_train, Y_train)
    
    print(f"✅ Best Alpha: {model.alpha_}")
    
    # 4. Evaluate & Plot
    preds_train = model.predict(X_train)
    preds_test = model.predict(X_test)
    
    r2_train = r2_score(Y_train, preds_train)
    r2_test = r2_score(Y_test, preds_test)
    
    # Correlation
    corr_train = np.corrcoef(Y_train, preds_train)[0, 1]
    corr_test = np.corrcoef(Y_test, preds_test)[0, 1]
    
    print(f"\n⚖️  RISULTATI:")
    print(f"Train R2: {r2_train:.3f} | Corr: {corr_train:.3f}")
    print(f"Test  R2: {r2_test:.3f}  | Corr: {corr_test:.3f}")
    
    # Plotting
    plt.figure(figsize=(12, 6))
    
    zoom = min(100, len(Y_test))
    # Normalizziamo per visualizzare meglio la forma (z-score)
    y_plot = (Y_test[:zoom] - Y_test[:zoom].mean()) / Y_test[:zoom].std()
    p_plot = (preds_test[:zoom] - preds_test[:zoom].mean()) / preds_test[:zoom].std()
    
    plt.plot(y_plot, label='Actual Video PC1 (HRF)', color='black', linewidth=2, alpha=0.7)
    plt.plot(p_plot, label=f'Predicted from V1 (Corr={corr_test:.2f})', color='red', linestyle='--')
    
    plt.title(f"Decoding Movie Gist (PC1) from V1\nSubject: {args.subject} | Test Set")
    plt.legend()
    plt.xlabel("Time (TR)")
    plt.grid(True, alpha=0.3)
    plt.savefig("sanity_check_adhd_pc1.png")
    print("💾 Plot salvato: sanity_check_adhd_pc1.png")

if __name__ == "__main__":
    main()