import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from tqdm import tqdm
import glob

# --- CONFIGURAZIONE PERCORSI ---
# Cartella creata da extract_features_split.py
FMRI_FEAT_DIR = "/home/mfinocchiaro/miccai2026/NeuroSTORM/hbn_data/neurostorm_features_split"
# File creato dal tuo script di estrazione feature video
VIDEO_FEAT_PATH = "/home/mfinocchiaro/miccai2026/NeuroSTORM/data/neurostorm_input_4d/dataset_fmri_features/features_brain_target_250.pt" 
SAVE_MODELS_DIR = "trained_aligners_hbn"

# Parametri
INPUT_DIM = 2304 # 288 * 2 * 2 * 2
CLIP_DIM = 1280  # CLIP embed size
SEQ_LEN = 20     # Lunghezza sequenza temporale

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- MODELLO LINEARE ---
class SimpleLinearAligner(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        # Regressione pura y = Wx + b (nessuna attivazione)
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)

def load_subject_data(subject_id, split, video_feats):
    """
    Carica SOLO i dati di un soggetto specifico.
    """
    subj_path = os.path.join(FMRI_FEAT_DIR, split, subject_id)
    if not os.path.exists(subj_path):
        print(f"❌ Errore: Cartella soggetto non trovata: {subj_path}")
        sys.exit(1)
        
    print(f"📦 Caricamento dati {split} per {subject_id}...")
    files = sorted(glob.glob(os.path.join(subj_path, "window_*.pt")))
    
    # Struttura: data_per_lag[k] = {'X': [], 'Y': []}
    data_per_lag = [{'X': [], 'Y': []} for _ in range(SEQ_LEN)]
    video_len = video_feats.shape[0]
    
    for fpath in files:
        try:
            # Estrae start_idx dal nome file (window_0123.pt)
            start_idx = int(os.path.basename(fpath).split('_')[1].split('.')[0])
            # Carica fMRI e appiattisci: [2304, 20]
            fmri_tensor = torch.load(fpath, map_location='cpu').float()
            fmri_flat = fmri_tensor.view(-1, SEQ_LEN) 
        except:
            continue
            
        # Distribuisci sui 20 lag
        for k in range(SEQ_LEN):
            video_idx = start_idx + k
            if video_idx >= video_len: continue
            
            # X: Feature fMRI al tempo k della finestra
            # Y: Feature Video al tempo assoluto (start + k)
            data_per_lag[k]['X'].append(fmri_flat[:, k])
            data_per_lag[k]['Y'].append(video_feats[video_idx])

    # Converti in tensori
    final_data = []
    for k in range(SEQ_LEN):
        if len(data_per_lag[k]['X']) > 0:
            X = torch.stack(data_per_lag[k]['X']).to(device)
            Y = torch.stack(data_per_lag[k]['Y']).to(device)
            final_data.append((X, Y))
        else:
            final_data.append((None, None))
            
    return final_data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=str, required=True, help="ID Soggetto (es. sub-NDARAA948VFH)")
    parser.add_argument("--epochs", type=int, default=20, help="Epoche L-BFGS")
    # Parametri Elastic Net (dal tuo script di riferimento)
    parser.add_argument("--l1", type=float, default=1e-3, help="Lasso Regularization")
    parser.add_argument("--l2", type=float, default=1e-2, help="Ridge Regularization")
    args = parser.parse_args()

    # Creazione cartella salvataggio specifica per il soggetto
    save_dir = os.path.join(SAVE_MODELS_DIR, args.subject)
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Carica Target Video
    print(f"📥 Loading Video Targets...")
    if not os.path.exists(VIDEO_FEAT_PATH):
        print("❌ File feature video non trovato!")
        return
    video_data = torch.load(VIDEO_FEAT_PATH)
    video_targets = video_data['clip_single'].to(device) # [N, 1280]

    # 2. Carica Dati Soggetto
    train_datasets = load_subject_data(args.subject, 'train', video_targets)
    val_datasets = load_subject_data(args.subject, 'val', video_targets)

    # 3. Training Loop (20 Modelli)
    print(f"\n🚀 Avvio Training per {args.subject} (L1: {args.l1} | L2: {args.l2})")
    criterion = nn.CosineEmbeddingLoss() # Ottimo per feature CLIP
    
    # Dummy target per CosineLoss (1 = similar)
    # Calcolato dinamicamente nel loop per adattarsi alle dim del batch

    for k in range(SEQ_LEN):
        print(f"\n🔹 Lag {k:02d}/{SEQ_LEN-1}")
        
        X_tr, Y_tr = train_datasets[k]
        X_val, Y_val = val_datasets[k]
        
        if X_tr is None:
            print("   ⚠️ No data. Skip.")
            continue

        # Inizializza modello
        model = SimpleLinearAligner(INPUT_DIM, CLIP_DIM).to(device)
        model.train()
        
        # Optimizer L-BFGS
        optimizer = optim.LBFGS(model.parameters(), lr=1.0, history_size=100, max_iter=20, line_search_fn="strong_wolfe")
        
        best_val_loss = float('inf')
        y_target_tr = torch.ones(X_tr.shape[0]).to(device)
        
        if X_val is not None:
            y_target_val = torch.ones(X_val.shape[0]).to(device)

        for ep in range(args.epochs):
            def closure():
                optimizer.zero_grad()
                preds = model(X_tr)

                # Main Loss (Cosine)
                loss = criterion(preds.half(), Y_tr.squeeze(1).half(), y_target_tr.half())
                # Elastic Net Regularization
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
                    loss_val = criterion(preds_val.half(), Y_val.squeeze(1).half(), y_target_val.half())
                    val_msg = f"| Val: {loss_val.item():.5f}"
                    
                    if loss_val.item() < best_val_loss:
                        best_val_loss = loss_val.item()
                        torch.save(model.state_dict(), os.path.join(save_dir, f"aligner_lag_{k:02d}.pth"))
                        val_msg += " [*]"
                model.train()
            else:
                # Se non c'è val set (es. pochi dati), salviamo sempre l'ultimo
                torch.save(model.state_dict(), os.path.join(save_dir, f"aligner_lag_{k:02d}.pth"))

            # Sparsity Check
            weights = model.linear.weight
            sparsity = ((weights.abs() < 1e-4).sum().item() / weights.numel()) * 100
            
            if ep % 5 == 0 or ep == args.epochs - 1:
                print(f"   Ep {ep+1:02d} | Train: {loss_tr.item():.5f} {val_msg} | Sparsity: {sparsity:.1f}%")

    print(f"\n✅ Training finito per {args.subject}.")

if __name__ == "__main__":
    main()