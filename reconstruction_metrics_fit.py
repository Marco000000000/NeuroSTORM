import os
import torch
import torch.nn as nn
import numpy as np
import argparse
import glob

# === CONFIGURAZIONE ===
BLOCK_SIZE = 30 # Blocchi di 30 frame (~24 secondi) per evitare leaking
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
                    all_ids.append(idx) # Teniamo traccia del tempo originale
            except: continue
    
    X, Y, T = torch.stack(all_x), torch.stack(all_y), np.array(all_ids)
    
    # LOGICA BLOCK SHUFFLE
    unique_blocks = np.unique(T // BLOCK_SIZE)
    np.random.seed(42)
    np.random.shuffle(unique_blocks)
    
    split = int(0.8 * len(unique_blocks))
    train_blocks = unique_blocks[:split]
    
    train_mask = np.isin(T // BLOCK_SIZE, train_blocks)
    return X[train_mask], Y[train_mask], X[~train_mask], Y[~train_mask]
def evaluate_block_shuffle(base_subject):
    save_path = os.path.join(SAVE_BASE_DIR, base_subject)
    model = BalancedAligner(2304, 1280).to(DEVICE)
    model.load_state_dict(torch.load(os.path.join(save_path, "block_best.pth"), map_location=DEVICE, weights_only=True))
    model.eval()
    mean_brain = torch.load(os.path.join(save_path, "mean_brain.pt"), weights_only=True).to(DEVICE)

    X_tr_raw, Y_tr, X_te_raw, Y_te = load_block_shuffled(base_subject)
    X_tr, X_te = (X_tr_raw - mean_brain.cpu()).to(DEVICE), (X_te_raw - mean_brain.cpu()).to(DEVICE)
    Y_tr, Y_te = Y_tr.to(DEVICE), Y_te.to(DEVICE)

    def get_acc(p, t):
        # CORREZIONE: keepdim=True
        p = p / p.norm(dim=1, keepdim=True)
        t = t / t.norm(dim=1, keepdim=True)
        sim = (p.half() @ t.half().t()).float()
        acc = 0
        for i in range(len(sim)):
            acc += (sim[i,i] > sim[i, torch.arange(len(sim))!=i]).sum().item()
        return acc / (len(sim)*(len(sim)-1))

    print(f"\n🧱 RISULTATI BLOCK-SHUFFLE per {base_subject}")
    with torch.no_grad():
        print(f" TRAIN (Blocchi 80%) | 2-Way Accuracy: {get_acc(model(X_tr), Y_tr):.1%}")
        print(f" TEST  (Blocchi 20%) | 2-Way Accuracy: {get_acc(model(X_te), Y_te):.1%}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=str, required=True)
    evaluate_block_shuffle(parser.parse_args().subject)