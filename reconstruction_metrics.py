import os
import torch
import torch.nn as nn
import numpy as np
import argparse
import pandas as pd
from tqdm import tqdm
import glob

# === CONFIGURAZIONE ===
FMRI_FEAT_DIR = "/home/mfinocchiaro/miccai2026/NeuroSTORM/hbn_data/neurostorm_features_split"
VIDEO_FEAT_PATH = "/home/mfinocchiaro/miccai2026/NeuroSTORM/data/neurostorm_input_4d/dataset_fmri_features/features_brain_target_250.pt" 
MODELS_BASE_DIR = "trained_aligners_contrastive"
RESULTS_FILE = "metrics_vs_val_distractors.csv"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
INPUT_DIM = 2304
SEQ_LEN = 20
HRF_SHIFT = 14 

class LinearAligner(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
    def forward(self, x): return self.linear(x)

def load_data_split(subject_id, split_name, video_targets):
    subj_path = os.path.join(FMRI_FEAT_DIR, split_name, subject_id)
    files = sorted(glob.glob(os.path.join(subj_path, "window_*.pt")))
    if not files: return None, None
    
    X_list, Y_list = [], []
    max_video_idx = video_targets.shape[0]

    for fpath in files:
        try:
            start_idx = int(os.path.basename(fpath).split('_')[1].split('.')[0])
            target_idx = (start_idx + 10) - HRF_SHIFT
            if 0 <= target_idx < max_video_idx:
                fmri = torch.load(fpath, map_location='cpu', weights_only=True).float().view(-1, SEQ_LEN)
                X_list.append(fmri[:, 10]) 
                Y_list.append(video_targets[target_idx])
        except: continue
        
    if not X_list: return None, None
    return torch.stack(X_list).to(DEVICE), torch.stack(Y_list).to(DEVICE)

def compute_gen_acc(preds, true_targets, distractor_pool):
    """Confronta Pred(Test) vs Target(Test) contro Target(Validation)"""
    n_test = preds.shape[0]
    n_dist = distractor_pool.shape[0]
    
    preds_norm = torch.nn.functional.normalize(preds, p=2, dim=1)
    true_targets_norm = torch.nn.functional.normalize(true_targets, p=2, dim=1)
    preds_norm =true_targets_norm
    dist_pool_norm = torch.nn.functional.normalize(distractor_pool, p=2, dim=1)
    
    sim_true = torch.sum(preds_norm * true_targets_norm, dim=1)
    
    total_hits = 0
    total_comps = 0
    N_BOOTSTRAP = 500 
    
    for i in range(n_test):
        score_correct = sim_true[i].item()
        # Prendi distrattori dal pool del Validation Set
        rand_idx = torch.randint(0, n_dist, (N_BOOTSTRAP,), device=DEVICE)
        distractors = dist_pool_norm[rand_idx]
        
        scores_wrong = torch.mm(preds_norm[i].unsqueeze(0).half(), distractors.T.half()).squeeze()
        total_hits += (score_correct > scores_wrong).sum().item()
        total_comps += N_BOOTSTRAP
        
    return total_hits / total_comps

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=str, required=True)
    args = parser.parse_args()
    
    video_data = torch.load(VIDEO_FEAT_PATH, weights_only=True)
    variants = ["clip_single", "clip_avg", "clip_dominant", "alex_single", "alex_avg", "alex_dominant"]
    results = []
    
    print(f"🔍 Valutazione: TEST vs VAL Distractors - {args.subject}")
    
    for var in variants:
        if var not in video_data: continue
        
        model_path = os.path.join(MODELS_BASE_DIR, args.subject, var, "best_model.pth")
        if not os.path.exists(model_path): continue
        
        targets = video_data[var].to(DEVICE)
        model = LinearAligner(INPUT_DIM, targets.shape[1]).to(DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
        model.eval()
        
        # Carica Test Set (per le predizioni)
        X_test, Y_test = load_data_split(args.subject, "test", targets)
        # Carica Validation Set (come pool di distrattori)
        _, Y_val = load_data_split(args.subject, "val", targets)
        
        if X_test is None or Y_val is None: continue
        
        with torch.no_grad():
            preds = model(X_test)
            
        gen_vs_val = compute_gen_acc(preds, Y_test, Y_val)
        
        print(f"   🔹 {var:15s} | Gen (Test vs Val Distractors): {gen_vs_val:.1%}")
        
        results.append({
            "subject": args.subject, 
            "variant": var, 
            "gen_vs_val_distractors": gen_vs_val
        })
            
    if results:
        df = pd.DataFrame(results)
        df.to_csv(RESULTS_FILE, mode='a', header=not os.path.exists(RESULTS_FILE), index=False)

if __name__ == "__main__":
    main()