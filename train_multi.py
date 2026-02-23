import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import glob
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# === CONFIGURAZIONE RIGIDA (Identica ai test precedenti) ===
BASE_FMRI_DIR = "/home/mfinocchiaro/miccai2026/NeuroSTORM/hbn_data/neurostorm_features_despicable_50"
VIDEO_PATH = "/home/mfinocchiaro/miccai2026/NeuroSTORM/data/neurostorm_input_4d/dataset_fmri_features/features_despicable.pt"
METADATA_PATH = "/home/mfinocchiaro/miccai2026/NeuroSTORM/data/neurostorm_input_4d/metadata/subject_labels.csv"
OUTPUT_DIR = "clinical_master_results"
HRF_SHIFT = 9
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class BalancedAligner(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.net = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(input_dim, 1024), nn.GELU(),
            nn.Dropout(0.5), nn.Linear(1024, 512), nn.GELU(), nn.Linear(512, output_dim)
        )
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
    def forward(self, x): return self.net(self.norm(x))

def load_split(full_id, split_name, video_targets):
    path = os.path.join(BASE_FMRI_DIR, split_name, full_id)
    files = sorted(glob.glob(os.path.join(path, "*.pt")))
    X, Y = [], []
    for f in files:
        try:
            idx = int(os.path.basename(f).split('_')[1].split('.')[0])
            t_idx = (idx + 10) - HRF_SHIFT
            if 0 <= t_idx < len(video_targets):
                X.append(torch.load(f, weights_only=True).float().view(-1, 20).mean(dim=1))
                Y.append(video_targets[t_idx])
        except: continue
    return torch.stack(X) if X else None, torch.stack(Y) if Y else None

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    video_data = torch.load(VIDEO_PATH, weights_only=True)["clip_avg"]
    meta_df = pd.read_csv(METADATA_PATH).set_index('subject_id')
    subjects = [d for d in os.listdir(os.path.join(BASE_FMRI_DIR, "train")) if "task-movieDM" in d]
    
    all_features = []
    all_test_preds = {} # Per ISC

    print(f"🚀 Avvio Pipeline su {len(subjects)} soggetti...")

    for full_id in tqdm(subjects):
        short_id = full_id.split('-')[1].split('_')[0]
        if short_id not in meta_df.index: continue
        
        # 1. Caricamento Split Temporale 50-10-40
        X_tr_raw, Y_tr = load_split(full_id, 'train', video_data)
        X_val_raw, Y_val = load_split(full_id, 'val', video_data)
        X_te_raw, Y_te = load_split(full_id, 'test', video_data)
        
        if X_tr_raw is None or X_te_raw is None: continue

        # 2. Centering e Training (Balanced Aligner)
        mean_brain = X_tr_raw.mean(dim=0, keepdim=True)
        X_tr, X_val, X_te = (X_tr_raw - mean_brain).to(DEVICE), (X_val_raw - mean_brain).to(DEVICE), (X_te_raw - mean_brain).to(DEVICE)
        Y_tr, Y_val, Y_te = Y_tr.to(DEVICE), Y_val.to(DEVICE), Y_te.to(DEVICE)

        model = BalancedAligner(2304, 1280).to(DEVICE)
        optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
        
        best_v_loss = float('inf')
        for ep in range(60): # Training standard
            model.train()
            idx = torch.randperm(len(X_tr))
            for s in range(0, len(idx), 32):
                b = idx[s:s+32]
                if len(b) < 4: continue
                p, t = nn.functional.normalize(model(X_tr[b]), dim=1), nn.functional.normalize(Y_tr[b], dim=1)
                logits = (p.half() @ t.half().t()).float() * model.logit_scale.exp()
                loss = nn.functional.cross_entropy(logits, torch.arange(len(b), device=DEVICE))
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            
            model.eval()
            with torch.no_grad():
                p_v, t_v = nn.functional.normalize(model(X_val), dim=1), nn.functional.normalize(Y_val, dim=1)
                v_loss = nn.functional.cross_entropy((p_v.half() @ t_v.half().t()).float() * model.logit_scale.exp(), torch.arange(len(X_val), device=DEVICE))
                if v_loss < best_v_loss:
                    best_v_loss = v_loss
                    best_state = model.state_dict()

        # 3. Estrazione Feature dal Test Set (40%)
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            preds = model(X_te)
            p_n, t_n = nn.functional.normalize(preds, dim=1), nn.functional.normalize(Y_te, dim=1)
            sim_matrix = (p_n.half() @ t_n.half().t()).float()
            
            # Rank temporale
            ranks = np.array([(sim_matrix[i,i] > sim_matrix[i, torch.arange(len(sim_matrix))!=i]).float().mean().item() for i in range(len(sim_matrix))])
            
            all_features.append({
                "subject_id": short_id,
                "label": meta_df.loc[short_id, 'label'],
                "mean_rank": np.mean(ranks),
                "stability": np.std(ranks),
                "engagement": np.mean(ranks > 0.5)
            })
            all_test_preds[short_id] = p_n # Per calcolo ISC

    # 4. Intersubject Correlation (ISC)
    # Calcoliamo quanto ogni predizione del soggetto è simile alla media dei CONTROLLI (label 0)
    df = pd.DataFrame(all_features)
    control_ids = df[df['label'] == 0]['subject_id'].values
    mean_control_pred = torch.stack([all_test_preds[sid] for sid in control_ids]).mean(dim=0)

    for i, row in df.iterrows():
        sid = row['subject_id']
        # Correlazione temporale tra la predizione del soggetto e la media dei controlli
        isc_val = torch.cosine_similarity(all_test_preds[sid], mean_control_pred, dim=1).mean().item()
        df.at[i, 'ISC'] = isc_val

    # 5. Classificazione Multi-Metodo
    X = df[['mean_rank', 'stability', 'engagement', 'ISC']].values
    y = df['label'].values
    loo = LeaveOneOut()
    
    y_preds_rf = []
    print("\n⚖️  Classificazione Cross-Validation...")
    for train_idx, test_idx in loo.split(X):
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X[train_idx], y[train_idx])
        y_preds_rf.append(clf.predict(X[test_idx])[0])

    # 6. Report e Matrice
    cm = confusion_matrix(y, y_preds_rf)
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=['Ctrl', 'ADHD'], yticklabels=['Ctrl', 'ADHD'])
    plt.title(f"Confusion Matrix (Acc: {accuracy_score(y, y_preds_rf):.2%})")

    plt.subplot(1, 2, 2)
    sns.barplot(x=['Rank', 'Stability', 'Engage', 'ISC'], y=clf.feature_importances_)
    plt.title("Importanza Marker Diagnostici")
    plt.savefig(f"{OUTPUT_DIR}/diagnostic_summary.png")
    
    df.to_csv(f"{OUTPUT_DIR}/complete_adhd_stats.csv", index=False)
    print(f"\n✅ Studio completato. Risultati in {OUTPUT_DIR}/complete_adhd_stats.csv")

if __name__ == "__main__":
    main()