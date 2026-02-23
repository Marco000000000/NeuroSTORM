import os
import glob
import numpy as np
import pandas as pd
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import nibabel as nib
from nilearn import datasets, image
from nilearn.input_data import NiftiMasker
from nilearn.image import resample_to_img
from sklearn.feature_selection import SelectKBest, f_regression
import copy

# === CONFIGURAZIONE ===
DATA_ROOT = "/home/mfinocchiaro/miccai2026/NeuroSTORM/data/neurostorm_input_4d/img"
VIDEO_FEAT_PATH = "/home/mfinocchiaro/miccai2026/NeuroSTORM/data/neurostorm_input_4d/dataset_fmri_features/features_despicable.pt" 
RESULTS_FILE = "metrics_contrastive_selection_v2.csv"

# Parametri
HRF_LAG_TR = 13  
BATCH_SIZE = 64
EPOCHS = 100
LR = 3e-4
N_VOXELS_KEEP = 2000 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class BrainEncoder(nn.Module):
    def __init__(self, input_dim, output_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Dropout(0.6),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.Linear(512, output_dim)
        )
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, x):
        return self.net(x)

def load_and_mask_subject(subject_id):
    subj_dir = os.path.join(DATA_ROOT, subject_id)
    frames = sorted([f for f in os.listdir(subj_dir) if f.startswith('frame_') and f.endswith('.pt')])
    if not frames: return None
    
    first_frame = torch.load(os.path.join(subj_dir, frames[0]), map_location='cpu').float().numpy()
    if len(first_frame.shape) == 4: first_frame = first_frame.squeeze(-1)
    affine = np.diag([2, 2, 2, 1]) 
    affine[:3, 3] = -np.array(first_frame.shape) / 2 * 2
    
    print(f"📂 Loading {len(frames)} frames...")
    vol_data = []
    for f in frames:
        t = torch.load(os.path.join(subj_dir, f), map_location='cpu').float().numpy()
        if len(t.shape) == 4: t = t.squeeze(-1)
        vol_data.append(t)
    vol_4d = np.stack(vol_data, axis=-1)
    subj_img = nib.Nifti1Image(vol_4d, affine)
    
    print("🧠 Masking (Full Visual Cortex)...")
    atlas = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
    keywords = ['occipital', 'calcarine', 'cuneal', 'lingual', 'fusiform']
    visual_indices = [i for i, l in enumerate(atlas.labels) if any(k in str(l).lower() for k in keywords)]
    
    atlas_img = image.load_img(atlas.maps)
    atlas_data = atlas_img.get_fdata()
    mask_data = np.zeros_like(atlas_data)
    for idx in visual_indices: mask_data[atlas_data == idx] = 1
    
    mask_img = resample_to_img(image.new_img_like(atlas_img, mask_data), subj_img, interpolation='nearest')
    masker = NiftiMasker(mask_img=mask_img, standardize=True, detrend=True)
    features = masker.fit_transform(subj_img)
    
    return np.nan_to_num(features)

def prepare_data(fmri, clip):
    n = min(len(fmri), len(clip))
    X = fmri[HRF_LAG_TR:n]
    Y = clip[:n-HRF_LAG_TR]
    
    n_train = int(len(X) * 0.50)
    n_val = int(len(X) * 0.60)
    
    return (X[:n_train], Y[:n_train]), (X[n_train:n_val], Y[n_train:n_val]), (X[n_val:], Y[n_val:])

def select_best_voxels(X_train, Y_train, k=2000):
    print(f"🕵️  Feature Selection: Selecting top {k} voxels...")
    from sklearn.decomposition import PCA
    pca = PCA(n_components=1)
    Y_scalar = pca.fit_transform(Y_train).flatten()
    selector = SelectKBest(f_regression, k=k)
    selector.fit(X_train, Y_scalar)
    print(f"   -> Done. Max F-score: {np.max(selector.scores_):.2f}")
    return selector.get_support()

# --- METRICHE ---
def compute_metrics(brain_emb, video_emb):
    b_norm = torch.nn.functional.normalize(brain_emb, dim=1)
    v_norm = torch.nn.functional.normalize(video_emb, dim=1)
    sim = b_norm @ v_norm.T
    n = len(sim)
    
    # Top-1
    labels = torch.arange(n).to(sim.device)
    _, preds = sim.topk(1, dim=1)
    top1 = (preds.T == labels).float().mean().item()
    
    # Pairwise
    diag = sim.diag()
    hits = 0
    total = 0
    for i in range(n):
        row = sim[i]
        negatives = row[torch.arange(n) != i]
        hits += (diag[i] > negatives).sum().item()
        total += len(negatives)
    pairwise = hits / total if total > 0 else 0
    return top1, pairwise

def compute_generalization(preds, true_targets, train_pool):
    """Accuratezza test vs distrattori del Training Set"""
    n_test = preds.shape[0]
    n_train = train_pool.shape[0]
    preds_norm = torch.nn.functional.normalize(preds, p=2, dim=1)
    true_targets_norm = torch.nn.functional.normalize(true_targets, p=2, dim=1)
    train_pool_norm = torch.nn.functional.normalize(train_pool, p=2, dim=1)
    
    sim_true = torch.sum(preds_norm * true_targets_norm, dim=1)
    
    total_hits = 0
    total_comps = 0
    N_BOOTSTRAP = 100 
    
    for i in range(n_test):
        score_correct = sim_true[i].item()
        # Confronta con N distrattori presi dal TRAIN
        rand_idx = torch.randint(0, n_train, (N_BOOTSTRAP,), device=DEVICE)
        distractors = train_pool_norm[rand_idx]
        scores_wrong = torch.mm(preds_norm[i].unsqueeze(0), distractors.T).squeeze()
        total_hits += (score_correct > scores_wrong).sum().item()
        total_comps += N_BOOTSTRAP
        
    return total_hits / total_comps

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=str, required=True)
    args = parser.parse_args()
    
    vid_data = torch.load(VIDEO_FEAT_PATH, map_location='cpu', weights_only=True)
    vid = vid_data["clip_avg"].float().numpy()
    fmri = load_and_mask_subject(args.subject)
    if fmri is None: return
    
    (X_train, Y_train), (X_val, Y_val), (X_test, Y_test) = prepare_data(fmri, vid)
    
    # Feature Selection
    voxel_mask = select_best_voxels(X_train, Y_train, k=N_VOXELS_KEEP)
    X_train = X_train[:, voxel_mask]
    X_val = X_val[:, voxel_mask]
    X_test = X_test[:, voxel_mask]
    
    print(f"📊 New Input Dim: {X_train.shape[1]} voxels (Cleaned)")
    
    # Training
    train_ds = TensorDataset(torch.tensor(X_train).float(), torch.tensor(Y_train).float())
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    
    model = BrainEncoder(X_train.shape[1], Y_train.shape[1]).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS)
    
    best_pairwise = 0
    best_model_state = None
    
    print("🚀 Training on Selected Voxels...")
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for bx, by in train_dl:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            optimizer.zero_grad()
            
            b_norm = torch.nn.functional.normalize(model(bx), dim=1)
            v_norm = torch.nn.functional.normalize(by, dim=1)
            logit_scale = model.logit_scale.exp()
            logits = logit_scale * (b_norm @ v_norm.T)
            labels = torch.arange(len(bx)).to(DEVICE)
            
            loss = (nn.CrossEntropyLoss()(logits, labels) + nn.CrossEntropyLoss()(logits.T, labels)) / 2
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        scheduler.step()
        
        # Validation Check
        if (epoch+1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_emb = model(torch.tensor(X_val).float().to(DEVICE))
                v_target = torch.tensor(Y_val).float().to(DEVICE)
                top1, pair = compute_metrics(val_emb, v_target)
            
            # Save Best
            if pair > best_pairwise:
                best_pairwise = pair
                best_model_state = copy.deepcopy(model.state_dict())
                
            print(f"   Ep {epoch+1} | Loss: {train_loss/len(train_dl):.4f} | Val Top1: {top1:.1%} | Pairwise: {pair:.1%} (Best: {best_pairwise:.1%})")

    # Restore Best Model for Test
    print("\n🔄 Caricamento miglior modello (Early Stopping)...")
    if best_model_state: model.load_state_dict(best_model_state)
    
    model.eval()
    with torch.no_grad():
        test_emb = model(torch.tensor(X_test).float().to(DEVICE))
        t_target = torch.tensor(Y_test).float().to(DEVICE)
        
        # Calcolo Metriche Finali
        top1, pair = compute_metrics(test_emb, t_target)
        
        # Calcolo Generalizzazione (vs Train Pool)
        train_pool = torch.tensor(Y_train).float().to(DEVICE)
        gen_acc = compute_generalization(test_emb, t_target, train_pool)
        
    print(f"\n⚖️  RISULTATI FINALI (Test Set)")
    print(f"Top-1 Acc:      {top1:.1%}")
    print(f"Pairwise Acc:   {pair:.1%}")
    print(f"Gen (vs Train): {gen_acc:.1%}") # New metric!

if __name__ == "__main__":
    main()