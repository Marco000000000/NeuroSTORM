import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from tqdm import tqdm
import glob


FMRI_FEAT_DIR = "/home/mfinocchiaro/miccai2026/NeuroSTORM/hbn_data/neurostorm_features_split"
# File creato dal tuo script di estrazione feature video
VIDEO_FEAT_PATH = "/home/mfinocchiaro/miccai2026/NeuroSTORM/data/neurostorm_input_4d/dataset_fmri_features/features_brain_target_250.pt" 
SAVE_BASE_DIR = "trained_aligners_unified"

# Parametri
INPUT_DIM = 2304 
CLIP_DIM = 1280  
SEQ_LEN = 20     

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- MODELLO UNIFICATO ---
class UnifiedLinearAligner(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        # Un solo layer per dominare tutti i lag
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)

def load_flattened_data(subject_id, split, video_feats):
    """
    Carica TUTTI i dati e li appiattisce in un unico grande tensore (N_samples * 20, Dim).
    """
    subj_path = os.path.join(FMRI_FEAT_DIR, split, subject_id)
    if not os.path.exists(subj_path):
        print(f"❌ Errore: Cartella {subj_path} non trovata.")
        sys.exit(1)
        
    print(f"📦 Caricamento e Appiattimento dati {split} per {subject_id}...")
    files = sorted(glob.glob(os.path.join(subj_path, "window_*.pt")))
    
    X_list = []
    Y_list = []
    video_len = video_feats.shape[0]
    
    for fpath in tqdm(files, desc=f"Loading {split}"):
        start_idx = int(os.path.basename(fpath).split('_')[1].split('.')[0])
        # Carica fMRI: [288, 2, 2, 2, 20]
        fmri_tensor = torch.load(fpath, map_location='cpu').float()
        # Appiattisci spazialmente: [2304, 20]
        fmri_flat = fmri_tensor.view(-1, SEQ_LEN) 
        print(f"Caricato {fpath} | fmri_flat shape: {fmri_flat.shape}")
        # Per ogni istante nella finestra temporale (0..19)
        print(SEQ_LEN)
        for k in range(SEQ_LEN):
            video_idx = start_idx + k
            print(video_idx, video_len)
            # Bounds check
            if video_idx >= video_len: continue
            
            # X: Feature fMRI al lag k
            # Y: Feature Video corrispondente
            X_list.append(fmri_flat[:, k])
            Y_list.append(video_feats[video_idx])

    if not X_list:
        return None, None

    print(f"🔄 Stack dei tensori {split}...")
    X_all = torch.stack(X_list).to(device)
    Y_all = torch.stack(Y_list).to(device)
    
    print(f"   Dataset {split}: {X_all.shape[0]} campioni totali.")
    return X_all, Y_all

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=str, required=True, help="ID Soggetto")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--l1", type=float, default=1e-4)
    parser.add_argument("--l2", type=float, default=1e-2)
    args = parser.parse_args()

    save_dir = os.path.join(SAVE_BASE_DIR, args.subject)
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Carica Target
    video_data = torch.load(VIDEO_FEAT_PATH)
    print(f"✅ Video targets caricati: {video_data['clip_single'].shape}, video key: {list(video_data.keys())}")
    video_targets = video_data['clip_single'].to(device)

    # 2. Carica Dati (Flattened)
    X_tr, Y_tr = load_flattened_data(args.subject, 'train', video_targets)
    X_val, Y_val = load_flattened_data(args.subject, 'val', video_targets)
    print(f"✅ Dati caricati. Train: {X_tr.shape if X_tr is not None else 'None'} | Val: {X_val.shape if X_val is not None else 'None'}")
    if X_tr is None:
        print("❌ Nessun dato di training trovato.")
        return

    # 3. Training Loop (Singolo Modello)
    print(f"\n🚀 Avvio Training Unificato (L1: {args.l1} | L2: {args.l2})")
    
    model = UnifiedLinearAligner(INPUT_DIM, CLIP_DIM).to(device)
    model.train()
    
    # Optimizer L-BFGS
    # Nota: Con tanti dati L-BFGS potrebbe essere lento. Se è troppo lento, switcha a AdamW.
    optimizer = optim.LBFGS(model.parameters(), lr=1.0, history_size=50, max_iter=20, line_search_fn="strong_wolfe")
    criterion = nn.CosineEmbeddingLoss()
    
    y_target_tr = torch.ones(X_tr.shape[0]).to(device)
    if X_val is not None:
        y_target_val = torch.ones(X_val.shape[0]).to(device)

    best_val_loss = float('inf')

    for ep in range(args.epochs):
        def closure():
            optimizer.zero_grad()
            preds = model(X_tr)
            loss = criterion(preds, Y_tr.squeeze(1), y_target_tr)
            
            # Elastic Net
            reg_loss = 0
            epsilon = 1e-6
            for param in model.parameters():
                l2 = torch.sum(param ** 2)
                l1 = torch.sum(torch.sqrt(param ** 2 + epsilon))
                reg_loss += (0.5 * args.l2 * l2) + (args.l1 * l1)
            
            total_loss = loss + reg_loss
            total_loss.backward()
            return total_loss

        loss_tr = optimizer.step(closure)
        
        # Validation
        val_msg = ""
        if X_val is not None:
            model.eval()
            with torch.no_grad():
                preds_val = model(X_val)
                loss_val = criterion(preds_val, Y_val.squeeze(1), y_target_val)
                val_msg = f"| Val: {loss_val.item():.5f}"
                
                if loss_val.item() < best_val_loss:
                    best_val_loss = loss_val.item()
                    torch.save(model.state_dict(), os.path.join(save_dir, "unified_aligner.pth"))
                    val_msg += " [*]"
            model.train()
        else:
            torch.save(model.state_dict(), os.path.join(save_dir, "unified_aligner.pth"))

        print(f"Ep {ep+1:02d} | Train: {loss_tr.item():.5f} {val_msg}")

    print(f"✅ Training finito. Modello salvato in {save_dir}")

if __name__ == "__main__":
    main()