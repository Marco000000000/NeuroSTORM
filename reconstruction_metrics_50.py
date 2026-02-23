import os
import torch
import torch.nn as nn
import numpy as np
import argparse
import pandas as pd
import glob

# === CONFIGURAZIONE ===
# FMRI_FEAT_DIR = "/home/mfinocchiaro/miccai2026/NeuroSTORM/hbn_data/neurostorm_features_split_50"
# VIDEO_FEAT_PATH = "/home/mfinocchiaro/miccai2026/NeuroSTORM/data/neurostorm_input_4d/dataset_fmri_features/features_brain_target_250.pt" 
MODELS_BASE_DIR = "trained_aligners_centered_d" 
FMRI_FEAT_DIR = "/home/mfinocchiaro/miccai2026/NeuroSTORM/hbn_data/neurostorm_features_despicable_50"
VIDEO_FEAT_PATH = "/home/mfinocchiaro/miccai2026/NeuroSTORM/data/neurostorm_input_4d/dataset_fmri_features/features_despicable.pt" 
# SAVE_BASE_DIR = "trained_aligners_centered_d"
RESULTS_FILE = "metrics_centered.csv"
HRF_SHIFT = 9
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
INPUT_DIM = 2304
SEQ_LEN = 20

# --- MODELLO (Lo stesso del training aggressive/centered) ---
class CenteredAligner(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        # Se hai usato 'train_centered.py' classico:
        self.norm = nn.LayerNorm(input_dim)
        self.net = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.GELU(),
            nn.Dropout(0.2), 
            nn.Linear(2048, 2048),
            nn.GELU(),
            nn.Linear(2048, output_dim)
        )
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, x):
        x = self.norm(x)
        return self.net(x)

def get_mean_brain(subject_id, split='train'):
    """Calcola la media del cervello dal training set per centrare i dati"""
    subj_path = os.path.join(FMRI_FEAT_DIR, split, subject_id)
    files = glob.glob(os.path.join(subj_path, "window_*.pt"))
    X = []
    # Ne carichiamo un po' per stimare la media
    for f in files[:100]: 
        try:
            t = torch.load(f, map_location='cpu', weights_only=True).float().view(-1, 20)[:, 10]
            X.append(t)
        except: pass
    if not X: return torch.zeros(2304).to(DEVICE)
    return torch.stack(X).mean(dim=0).to(DEVICE)

def load_data_split(subject_id, split_name, video_targets):
    subj_path = os.path.join(FMRI_FEAT_DIR, split_name, subject_id)
    files = sorted(glob.glob(os.path.join(subj_path, "window_*.pt")))
    if not files: return None, None
    X, Y = [], []
    max_video = video_targets.shape[0]
    for fpath in files:
        try:
            start_idx = int(os.path.basename(fpath).split('_')[1].split('.')[0])
            fmri = torch.load(fpath, map_location='cpu', weights_only=True).float().view(-1, SEQ_LEN)
            k = 10 
            target_idx = (start_idx + k) - HRF_SHIFT
            if 0 <= target_idx < max_video:
                X.append(fmri[:, k])
                Y.append(video_targets[target_idx])
        except: continue
    if not X: return None, None
    return torch.stack(X).to(DEVICE), torch.stack(Y).to(DEVICE)

def compute_metrics(preds, targets):
    n_samples = preds.shape[0]
    preds_norm = torch.nn.functional.normalize(preds, p=2, dim=1)
    targets_norm = torch.nn.functional.normalize(targets, p=2, dim=1)
    similarity_matrix = torch.mm(preds_norm.half(), targets_norm.T.half()).float()
    
    top1 = 0
    ranks = []
    hits = 0
    pairs = 0
    for i in range(n_samples):
        scores = similarity_matrix[i, :]
        true_score = scores[i].item()
        sorted_indices = torch.argsort(scores, descending=True)
        rank = (sorted_indices == i).nonzero(as_tuple=True)[0].item()
        ranks.append(rank)
        if rank == 0: top1 += 1
        other_scores = scores[torch.arange(n_samples) != i]
        if len(other_scores) > 0:
            hits += (true_score > other_scores).sum().item()
            pairs += len(other_scores)
    return {"pairwise_acc": hits/pairs if pairs>0 else 0, "top1_acc": top1/n_samples, "mean_rank": np.mean(ranks)}

def compute_generalization(preds, true_targets, train_pool):
    n_test = preds.shape[0]
    n_train = train_pool.shape[0]
    preds_norm = torch.nn.functional.normalize(preds, p=2, dim=1)
    true_targets_norm = torch.nn.functional.normalize(true_targets, p=2, dim=1)
    train_pool_norm = torch.nn.functional.normalize(train_pool, p=2, dim=1)
    
    sim_true = torch.sum(preds_norm * true_targets_norm, dim=1)
    total_hits = 0
    total_comps = 0
    N_BOOTSTRAP = 500
    for i in range(n_test):
        score_correct = sim_true[i].item()
        rand_idx = torch.randint(0, n_train, (N_BOOTSTRAP,), device=DEVICE)
        distractors = train_pool_norm[rand_idx]
        scores_wrong = torch.mm(preds_norm[i].unsqueeze(0).half(), distractors.T.half()).squeeze()
        total_hits += (score_correct > scores_wrong).sum().item()
        total_comps += N_BOOTSTRAP
    return total_hits / total_comps

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=str, required=True)
    args = parser.parse_args()
    
    if not os.path.exists(VIDEO_FEAT_PATH): return
    video_data = torch.load(VIDEO_FEAT_PATH, weights_only=True)
    variants = ["clip_avg", "alex_avg"]
    results = []
    
    print(f"🧠 Eval CENTERED (Lag={HRF_SHIFT}) for {args.subject}")
    print(f"{'Variant':<15} | {'Split':<5} | {'2-Way':<8} | {'Top-1':<8} | {'Gen(vsPool)':<12}")
    print("-" * 65)

    # 1. Calcola Mean Brain (fondamentale!)
    mean_brain = get_mean_brain(args.subject)

    for var in variants:
        if var not in video_data: continue
        model_path = os.path.join(MODELS_BASE_DIR, args.subject, var, "best_model.pth")
        if not os.path.exists(model_path): continue
        
        targets = video_data[var].to(DEVICE)
        model = CenteredAligner(INPUT_DIM, targets.shape[1]).to(DEVICE)
        try:
            model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
        except: continue
        model.eval()
        
        train_pool = targets[:int(targets.shape[0]*0.5)]

        for split in ["train", "val", "test"]:
            X, Y = load_data_split(args.subject, split, targets)
            if X is None or len(X) < 2: continue
            
            # APPLICA CENTERING ANCHE QUI
            X = X - mean_brain
            
            with torch.no_grad(): preds = model(X)
            std = compute_metrics(preds, Y)
            gen = compute_generalization(preds, Y, train_pool)
            print(f"{var:<15} | {split.upper():<5} | {std['pairwise_acc']:.1%}    | {std['top1_acc']:.1%}    | {gen:.1%}")
            
            res = {"subject": args.subject, "variant": var, "split": split}
            res.update(std)
            res["gen_acc"] = gen
            results.append(res)

    if results:
        df = pd.DataFrame(results)
        df.to_csv(RESULTS_FILE, mode='a', header=not os.path.exists(RESULTS_FILE), index=False)

if __name__ == "__main__":
    main()