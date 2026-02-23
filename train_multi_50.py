import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import glob

# === CONFIGURAZIONE ===
FMRI_DIR = "/home/mfinocchiaro/miccai2026/NeuroSTORM/hbn_data/neurostorm_features_despicable_50"
VIDEO_PATH = "/home/mfinocchiaro/miccai2026/NeuroSTORM/data/neurostorm_input_4d/dataset_fmri_features/features_despicable.pt"
SUBJECT = "sub-NDARAA948VFH_task-movieDM"
SAVE_DIR = "reconstruction_analysis_DM"
HRF_SHIFT = 9
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class BalancedAligner(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.net = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(input_dim, 1024), nn.GELU(),
            nn.Dropout(0.5), nn.Linear(1024, 512), nn.GELU(),
            nn.Linear(512, output_dim)
        )
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
    def forward(self, x): return self.net(self.norm(x))

def load_split_data(split_name, video_targets):
    subj_path = os.path.join(FMRI_DIR, split_name, SUBJECT)
    files = sorted(glob.glob(os.path.join(subj_path, "*.pt")))
    X, Y, IDs = [], [], []
    for f in files:
        try:
            idx = int(os.path.basename(f).split('_')[1].split('.')[0])
            t_idx = (idx + 10) - HRF_SHIFT
            if 0 <= t_idx < len(video_targets):
                feat = torch.load(f, map_location='cpu', weights_only=True).float().view(-1, 20)
                X.append(feat.mean(dim=1))
                Y.append(video_targets[t_idx])
                IDs.append(idx)
        except: continue
    return (torch.stack(X), torch.stack(Y), np.array(IDs)) if X else (None, None, None)

def run_analysis():
    os.makedirs(SAVE_DIR, exist_ok=True)
    video_data = torch.load(VIDEO_PATH, weights_only=True)["clip_avg"]
    
    # 1. Caricamento dati (50-10-40 temporale)
    X_tr_raw, Y_tr, _ = load_split_data('train', video_data)
    X_val_raw, Y_val, _ = load_split_data('val', video_data)
    X_te_raw, Y_te, T_te = load_split_data('test', video_data)
    
    # 2. Centering
    mean_brain = X_tr_raw.mean(dim=0, keepdim=True)
    X_tr, X_val, X_te = (X_tr_raw - mean_brain).to(DEVICE), (X_val_raw - mean_brain).to(DEVICE), (X_te_raw - mean_brain).to(DEVICE)
    Y_tr, Y_val, Y_te = Y_tr.to(DEVICE), Y_val.to(DEVICE), Y_te.to(DEVICE)

    # 3. Training (Veloce, Balanced)
    model = BalancedAligner(2304, 1280).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    
    print("🚀 Training in corso...")
    best_loss = float('inf')
    for ep in range(100):
        model.train()
        idx = torch.randperm(len(X_tr))
        for s in range(0, len(idx), 32):
            b = idx[s:s+32]
            if len(b) < 4: continue
            preds = model(X_tr[b])
            p, t = nn.functional.normalize(preds, dim=1), nn.functional.normalize(Y_tr[b], dim=1)
            logits = (p.half() @ t.half().t()).float() * model.logit_scale.exp()
            loss = nn.functional.cross_entropy(logits, torch.arange(len(b), device=DEVICE))
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        
        model.eval()
        with torch.no_grad():
            preds_v = model(X_val)
            p_v, t_v = nn.functional.normalize(preds_v, dim=1), nn.functional.normalize(Y_val, dim=1)
            v_loss = nn.functional.cross_entropy((p_v.half() @ t_v.half().t()).float() * model.logit_scale.exp(), torch.arange(len(X_val), device=DEVICE))
            if v_loss < best_loss:
                best_loss = v_loss
                torch.save(model.state_dict(), f"{SAVE_DIR}/best_model.pth")

    # 4. Inferenza Temporale sul Test Set
    print("📊 Generazione pattern temporale...")
    model.load_state_dict(torch.load(f"{SAVE_DIR}/best_model.pth"))
    model.eval()
    
    with torch.no_grad():
        preds_te = model(X_te)
        p_n = nn.functional.normalize(preds_te, dim=1)
        t_n = nn.functional.normalize(Y_te, dim=1)
        
        # Similarità cosina punto per punto
        similarities = torch.sum(p_n * t_n, dim=1).cpu().numpy()
        
        # Identificazione (Percentile Rank)
        # Per ogni volume, quanti frame del test set sono "più lontani" del vero target?
        sim_matrix = torch.matmul(p_n.half(), t_n.half().t()).float()
        ranks = []
        for i in range(len(sim_matrix)):
            true_sim = sim_matrix[i, i]
            rank = (sim_matrix[i] < true_sim).float().mean().item()
            ranks.append(rank)

    # 5. Plotting
    plt.figure(figsize=(15, 6))
    plt.subplot(2, 1, 1)
    plt.plot(T_te, similarities, color='blue', label='Cosine Similarity')
    plt.title(f"Reconstruction Pattern - {SUBJECT}")
    plt.ylabel("Similarity")
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 1, 2)
    plt.fill_between(T_te, ranks, 0.5, where=(np.array(ranks) >= 0.5), color='green', alpha=0.3, label='Above Chance')
    plt.fill_between(T_te, ranks, 0.5, where=(np.array(ranks) < 0.5), color='red', alpha=0.3, label='Below Chance')
    plt.axhline(y=0.5, color='black', linestyle='--')
    plt.ylabel("Identification Rank")
    plt.xlabel("Volume Index (TR)")
    plt.ylim(0, 1)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/reconstruction_pattern.png")
    print(f"✅ Analisi completata. Grafico salvato in {SAVE_DIR}")

if __name__ == "__main__":
    run_analysis()