import os
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import shutil

# --- CONFIGURAZIONE ---
# Percorso dei frame .pt creati da preprocessing_volume
DATA_ROOT = "/home/mfinocchiaro/miccai2026/NeuroSTORM/data/neurostorm_input_4d/img"

# Percorso del checkpoint MAE addestrato
# CKPT_PATH = "/home/mfinocchiaro/miccai2026/NeuroSTORM/output/neurostorm/hbn_mae_domain_adaptation/checkpt-epoch=09-valid_loss=0.09.ckpt" 
CKPT_PATH = "/home/mfinocchiaro/miccai2026/NeuroSTORM/output/neurostorm/hbn_mae_split_50_10_40/checkpt-epoch=09-valid_loss=0.09.ckpt"
# CKPT_PATH = "/home/mfinocchiaro/miccai2026/NeuroSTORM/pretrained_models/fmrifound/pt_fmrifound_mae_ratio0.5.ckpt"
# Dove salvare le feature divise
OUTPUT_DIR = "/home/mfinocchiaro/miccai2026/NeuroSTORM/hbn_data/neurostorm_features_despicable_50"

# Filtro task (es. 'movieTP')
TASK_FILTER = "movieDM"

# Parametri Modello (devono corrispondere al training)
IMG_SIZE = (96, 96, 96, 20)
EMBED_DIM = 36
WINDOW_SIZE = [4, 4, 4, 4]
FIRST_WINDOW_SIZE = [4, 4, 4, 4]
PATCH_SIZE = [6, 6, 6, 1]
DEPTHS = [2, 2, 6, 2]
NUM_HEADS = [3, 6, 12, 24]

# --- SETUP ---
sys.path.append(os.getcwd())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Creiamo le cartelle di destinazione
for split in ['train', 'val', 'test']:
    os.makedirs(os.path.join(OUTPUT_DIR, split), exist_ok=True)

try:
    from models.neurostorm import NeuroSTORMMAE
except ImportError:
    print("❌ Errore importazione. Esegui dalla root di NeuroSTORM.")
    sys.exit(1)

# --- DATASET INTELLIGENTE ---
class HBN_Split_Inference(Dataset):
    def __init__(self, root, task_filter, seq_len=20):
        self.root = root
        self.seq_len = seq_len
        self.data = []
        
        subjects = sorted([d for d in os.listdir(root) if task_filter in d])
        print(f"🔍 Analisi di {len(subjects)} soggetti per lo split temporale...")

        for subj in subjects:
            subj_path = os.path.join(root, subj)
            
            # Conta frame totali
            frames = sorted([f for f in os.listdir(subj_path) if f.startswith('frame_')])
            num_frames = len(frames)
            
            if num_frames < seq_len:
                continue
            
            # --- LOGICA DI SPLIT ---
            # Calcoliamo gli indici di confine basati sul numero totale di frame
            idx_train_end = int(num_frames * 0.50)
            idx_val_end = int(num_frames * 0.60)
            
            # Calcoliamo l'ultimo indice di partenza possibile
            max_start = num_frames - seq_len
            
            for start_idx in range(0, max_start + 1, 1): # Stride = 1
                # Determina a quale split appartiene questa finestra
                # La finestra inizia a start_idx. 
                # Se inizia prima del 50%, è train.
                # Se inizia tra 50% e 60%, è val.
                # Altrimenti è test.
                if start_idx < idx_train_end:
                    split = 'train'
                elif start_idx < idx_val_end:
                    split = 'val'
                else:
                    split = 'test'
                
                self.data.append({
                    "subject": subj,
                    "path": subj_path,
                    "start_frame": start_idx,
                    "split": split
                })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        frames = []
        
        # Carica i 20 frame della sequenza
        for i in range(self.seq_len):
            fname = os.path.join(item["path"], f"frame_{item['start_frame'] + i}.pt")
            try:
                t = torch.load(fname).float()
                if len(t.shape) == 3: t = t.unsqueeze(-1)
                frames.append(t)
            except:
                frames.append(torch.zeros(96,96,96,1)) # Fallback silenzioso

        # Crea volume 4D (H,W,D,T) -> Aggiungi batch (C,H,W,D,T)
        volume = torch.cat(frames, dim=3).unsqueeze(0)
        
        return volume, item["subject"], item["start_frame"], item["split"]

def main():
    print("🚀 Avvio Estrazione Feature con Split (Train/Val/Test)")
    
    # 1. Carica Dataset
    dataset = HBN_Split_Inference(DATA_ROOT, TASK_FILTER, seq_len=IMG_SIZE[3])
    # Batch size > 1 per velocizzare, ma occhio alla VRAM con i volumi 4D
    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=8)
    
    print(f"📦 Totale finestre da elaborare: {len(dataset)}")

    # 2. Carica Modello
    print("🏗️  Inizializzazione Modello...")
    model = NeuroSTORMMAE(
        img_size=IMG_SIZE, in_chans=1, embed_dim=EMBED_DIM,
        window_size=WINDOW_SIZE, first_window_size=FIRST_WINDOW_SIZE,
        patch_size=PATCH_SIZE, depths=DEPTHS, num_heads=NUM_HEADS,
        c_multiplier=2, last_layer_full_MSA=False,
        mask_ratio=0.0, # ⚠️ Fondamentale: Nessun mascheramento per l'estrazione!
        spatial_mask="window", time_mask="random"
    ).to(device)

    # 3. Carica Pesi
    if os.path.exists(CKPT_PATH):
        print(f"📥 Loading checkpoint: {CKPT_PATH}")
        ckpt = torch.load(CKPT_PATH, map_location='cpu')
        # Pulisce le chiavi
        sd = {k.replace('model.', '').replace('net.', ''): v for k, v in ckpt['state_dict'].items()}
        model.load_state_dict(sd, strict=False)
    else:
        print("⚠️  Checkpoint non trovato! Procedo con pesi random (SCONSIGLIATO).")

    model.eval()
    
    # 4. Loop di Estrazione
    print("running...")
    with torch.no_grad():
        for batch_vol, subjects, start_frames, splits in tqdm(loader):
            batch_vol = batch_vol.to(device)
            
            # --- ESTRAZIONE FEATURE ---
            # Usiamo solo l'encoder per ottenere la rappresentazione latente
            features, _ = model.forward_encoder(batch_vol)            
            features = features.cpu() # Sposta in RAM
            
            # --- SALVATAGGIO ---
            for i in range(len(subjects)):
                subj_name = subjects[i]
                start_f = start_frames[i].item()
                split_name = splits[i]
                feat_tensor = features[i] # Feature singola
                
                # Percorso: output/train/sub-XXX/window_YYY.pt
                save_dir = os.path.join(OUTPUT_DIR, split_name, subj_name)
                os.makedirs(save_dir, exist_ok=True)
                
                save_name = f"window_{start_f:04d}.pt"
                torch.save(feat_tensor, os.path.join(save_dir, save_name))

    print(f"\n✅ Fatto! Feature salvate in {OUTPUT_DIR}")
    print("Struttura creata: train/, val/, test/")

if __name__ == "__main__":
    main()