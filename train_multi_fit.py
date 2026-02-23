import os
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import numpy as np
import glob

# === CONFIGURAZIONE ===
BLOCK_SIZE = 30 
CONFIGS = {
    "movieTP": {
        "suffix": "_task-movieTP",
        "fmri_dir_base": "/home/mfinocchiaro/miccai2026/NeuroSTORM/hbn_data/neurostorm_features_split_50",
        "video_path": "/home/mfinocchiaro/miccai2026/NeuroSTORM/data/neurostorm_input_4d/dataset_fmri_features/features_brain_target_250.pt"
    },
    "movieDM": {
        "suffix": "_task-movieDM",
        "fmri_dir_base": "/home/mfinocchiaro/miccai2026/NeuroSTORM/hbn_data/neurostorm_features_despicable_50",
        "video_path": "/home/mfinocchiaro/miccai2026/NeuroSTORM/data/neurostorm_input_4d/dataset_fmri_features/features_despicable.pt"
    }
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_BASE_DIR = "trained_aligners_BLOCK_SHUFFLE"

class BalancedAligner(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.net = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(input_dim, 512), nn.GELU(),
            nn.Dropout(0.5), nn.Linear(512, 512), nn.GELU(), nn.Linear(512, output_dim)
        )
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
    def forward(self, x): return self.net(self.norm(x))

def load_block_shuffled(base_subject):
    all_x, all_y, all_ids = [], [], []
    for t_name, conf in CONFIGS.items():
        if not os.path.exists(conf["video_path"]): continue
        v_data = torch.load(conf["video_path"], weights_only=True)["clip_avg"]
        search_path = os.path.join(conf["fmri_dir_base"], "*", base_subject + conf["suffix"], "*.pt")
        files = glob.glob(search_path)
        for f in files:
            try:
                idx = int(os.path.basename(f).split('_')[1].split('.')[0])
                t_idx = (idx + 10) - 9
                if 0 <= t_idx < len(v_data):
                    feat = torch.load(f, map_location='cpu', weights_only=True).float().view(-1, 20)
                    all_x.append(feat.mean(dim=1))
                    all_y.append(v_data[t_idx])
                    all_ids.append(idx)
            except: continue
    
    X, Y, T = torch.stack(all_x), torch.stack(all_y), np.array(all_ids)
    
    unique_blocks = np.unique(T // BLOCK_SIZE)
    np.random.seed(42)
    np.random.shuffle(unique_blocks)
    
    split = int(0.8 * len(unique_blocks))
    train_blocks = unique_blocks[:split]
    
    train_mask = np.isin(T // BLOCK_SIZE, train_blocks)
    return X[train_mask], Y[train_mask], X[~train_mask], Y[~train_mask]

def train_block_shuffle(base_subject):
    print(f"\n🧱 TRAINING BLOCK-SHUFFLE per {base_subject}")
    X_tr_raw, Y_tr, X_te_raw, Y_te = load_block_shuffled(base_subject)
    
    mean_brain = X_tr_raw.mean(dim=0, keepdim=True)
    X_tr, X_te = (X_tr_raw - mean_brain).to(DEVICE), (X_te_raw - mean_brain).to(DEVICE)
    Y_tr, Y_te = Y_tr.to(DEVICE), Y_te.to(DEVICE)

    model = BalancedAligner(2304, 1280).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    save_path = os.path.join(SAVE_BASE_DIR, base_subject)
    os.makedirs(save_path, exist_ok=True)
    torch.save(mean_brain.cpu(), os.path.join(save_path, "mean_brain.pt"))

    best_v = float('inf')
    for ep in range(100):
        model.train()
        idx = torch.randperm(len(X_tr))
        for s in range(0, len(idx), 32):
            b = idx[s:s+32]
            if len(b) < 4: continue
            preds = model(X_tr[b])
            # CORREZIONE: keepdim=True invece di k=True
            p = preds / preds.norm(dim=1, keepdim=True)
            t = Y_tr[b] / Y_tr[b].norm(dim=1, keepdim=True)
            
            logits = (p.half() @ t.half().t()).float() * model.logit_scale.exp()
            labels = torch.arange(len(b), device=DEVICE)
            loss = (nn.functional.cross_entropy(logits, labels) + nn.functional.cross_entropy(logits.t(), labels))/2
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        
        model.eval()
        with torch.no_grad():
            p_v = model(X_te)
            p_v = p_v / p_v.norm(dim=1, keepdim=True)
            t_v = Y_te / Y_te.norm(dim=1, keepdim=True)
            v_logits = (p_v.half() @ t_v.half().t()).float() * model.logit_scale.exp()
            v_loss = nn.functional.cross_entropy(v_logits, torch.arange(len(X_te), device=DEVICE)).item()
            if v_loss < best_v:
                best_v = v_loss
                torch.save(model.state_dict(), os.path.join(save_path, "block_best.pth"))
        if ep % 20 == 0: print(f" Ep {ep} | Val Loss: {v_loss:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=str, required=True)
    train_block_shuffle(parser.parse_args().subject)