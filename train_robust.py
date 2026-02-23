import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.manifold import TSNE
from tqdm import tqdm

# === CONFIGURAZIONE ===
BASE_FMRI_DIR = "/home/mfinocchiaro/miccai2026/NeuroSTORM/hbn_data/neurostorm_features_despicable_50"
VIDEO_PATH = "/home/mfinocchiaro/miccai2026/NeuroSTORM/data/neurostorm_input_4d/dataset_fmri_features/features_despicable.pt"
METADATA_PATH = "subject_labels.csv" # Usiamo il file locale
OUTPUT_DIR = "knn_subject_analysis"
HRF_SHIFT = 9
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class BalancedAligner(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.net = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(input_dim, 512), nn.GELU(),
            nn.Dropout(0.5), nn.Linear(512, 512), nn.GELU(), nn.Linear(512, output_dim)
        )
    def forward(self, x): return self.net(self.norm(x))

def get_subject_trajectory(full_id, split, video_targets):
    """Allena un modello rapido e restituisce la traiettoria semantica (predizioni) sul Test Set"""
    # 1. Carica Train e Test
    def load(s):
        path = os.path.join(BASE_FMRI_DIR, s, full_id)
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

    X_tr, Y_tr = load('train')
    X_te, _ = load('test') # Non ci servono i target del test, solo le predizioni
    
    if X_tr is None or X_te is None: return None

    # 2. Allenamento Rapido (Solo per allineare)
    mean_brain = X_tr.mean(dim=0, keepdim=True)
    X_tr, X_te = (X_tr - mean_brain).to(DEVICE), (X_te - mean_brain).to(DEVICE)
    Y_tr = Y_tr.to(DEVICE)

    model = BalancedAligner(2304, 1280).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=2e-4)
    
    for _ in range(30): # Pochi step bastano per proiettare nello spazio comune
        model.train()
        idx = torch.randperm(len(X_tr))
        for s in range(0, len(idx), 64):
            b = idx[s:s+64]
            if len(b) < 4: continue
            p, t = nn.functional.normalize(model(X_tr[b]), dim=1), nn.functional.normalize(Y_tr[b], dim=1)
            loss = nn.functional.cross_entropy((p.half() @ t.t().half()).float() * np.log(1/0.07), torch.arange(len(b), device=DEVICE))
            optimizer.zero_grad(); loss.backward(); optimizer.step()

    # 3. Estrai Traiettoria Test
    model.eval()
    with torch.no_grad():
        preds = nn.functional.normalize(model(X_te), dim=1).cpu()
    
    return preds # Matrice [Time x Features]

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    video_data = torch.load(VIDEO_PATH, weights_only=True)["clip_avg"]
    meta_df = pd.read_csv(METADATA_PATH).set_index('subject_id')
    subjects = [d for d in os.listdir(os.path.join(BASE_FMRI_DIR, "train")) if "task-movieDM" in d]
    
    # Raccogli tutte le traiettorie
    trajectories = {}
    labels = {}
    
    print(f"🔄 Estrazione traiettorie semantiche per {len(subjects)} soggetti...")
    for full_id in tqdm(subjects):
        sid = full_id.split('-')[1].split('_')[0]
        if sid not in meta_df.index: continue
        
        traj = get_subject_trajectory(full_id, 'train', video_data)
        if traj is not None:
            trajectories[sid] = traj.flatten().numpy() # Appiattiamo il tempo per fare correlazione
            labels[sid] = meta_df.loc[sid, 'label']

    # Costruiamo la matrice Soggetti x Features (dove Feature = Tutta la storia temporale)
    sids = list(trajectories.keys())
    X_matrix = np.stack([trajectories[s] for s in sids])
    y_labels = np.array([labels[s] for s in sids])
    
    print(f"\n📊 Calcolo Matrice di Similarità ({len(sids)}x{len(sids)})...")
    # Correlazione di Pearson tra soggetti
    # (Normalizziamo le righe e facciamo prodotto scalare = Cosine Similarity)
    X_norm = X_matrix / np.linalg.norm(X_matrix, axis=1, keepdims=True)
    similarity_matrix = np.dot(X_norm, X_norm.T)

    # --- 1. k-NN CLASSIFICATION ---
    knn = KNeighborsClassifier(n_neighbors=5, metric='precomputed')
    
    # Convertiamo similarità in distanza per scikit-learn (1 - sim)
    dist_matrix = 1 - similarity_matrix
    np.fill_diagonal(dist_matrix, 0) # Distanza da se stesso è 0
    
    # Leave-One-Out manuale per k-NN
    preds, true_vals = [], []
    for i in range(len(sids)):
        # Maschera per training (tutti tranne i)
        train_idx = np.arange(len(sids)) != i
        
        # Fit sui vicini (usando la matrice precalcolata)
        knn.fit(dist_matrix[train_idx][:, train_idx], y_labels[train_idx])
        
        # Predici l'escluso
        pred = knn.predict(dist_matrix[i, train_idx].reshape(1, -1))
        preds.append(pred[0])
        true_vals.append(y_labels[i])

    acc = accuracy_score(true_vals, preds)
    cm = confusion_matrix(true_vals, preds)
    print(f"\n🧠 k-NN Accuracy (k=5): {acc:.2%}")
    
    # --- 2. VISUALIZZAZIONE MATRICE (Ordinata per Gruppo) ---
    # Ordiniamo gli indici: Prima tutti i Controlli (0), poi tutti gli ADHD (1)
    sort_idx = np.argsort(y_labels)
    sorted_sim = similarity_matrix[sort_idx][:, sort_idx]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(sorted_sim, cmap="viridis", vmin=0, vmax=1)
    
    # Linee divisorie
    n_controls = (y_labels == 0).sum()
    plt.axvline(n_controls, color='white', linestyle='--', linewidth=1)
    plt.axhline(n_controls, color='white', linestyle='--', linewidth=1)
    
    plt.title(f"Subject Similarity Matrix (Sorted)\nTop-Left: Controls | Bottom-Right: ADHD")
    plt.savefig(f"{OUTPUT_DIR}/subject_similarity_matrix.png")
    
    # --- 3. t-SNE CLUSTERING ---
    # Vediamo se si separano nello spazio 2D
    tsne = TSNE(n_components=2, metric='precomputed', random_state=42, init='random')
    X_emb = tsne.fit_transform(dist_matrix)
    
    plt.figure(figsize=(8, 6))
    colors = ['blue' if l == 0 else 'red' for l in y_labels]
    plt.scatter(X_emb[:, 0], X_emb[:, 1], c=colors, alpha=0.7)
    # Legend trick
    plt.scatter([], [], c='blue', label='Control')
    plt.scatter([], [], c='red', label='ADHD')
    plt.legend()
    plt.title(f"t-SNE of Subject Brains (k-NN Acc: {acc:.1%})")
    plt.savefig(f"{OUTPUT_DIR}/tsne_subjects.png")

    print(f"✅ Analisi completata. Grafici in {OUTPUT_DIR}")

if __name__ == "__main__":
    main()