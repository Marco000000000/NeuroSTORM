import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import glob

# Aggiungi path NeuroSTORM
sys.path.append(os.path.abspath("NeuroSTORM"))

try:
    from models.neurostorm import NeuroSTORMMAE
except ImportError:
    print("❌ Errore importazione NeuroSTORMMAE.")
    sys.exit(1)

# --- CONFIGURAZIONE ---
# Cartella creata da preprocessing_volume.py
DATA_ROOT = "NeuroSTORM/data/UCLA_MNI_to_TRs_minmax/img" 
CHECKPOINT_PATH = "NeuroSTORM/pretrained_models/fmrifound/last13.ckpt" # Assicurati sia qui
SAVE_IMG_PATH = "check_official.png"

# Parametri modello
EMBED_DIM = 36
IMG_SIZE = (96, 96, 96, 20)
SEQ_LEN = 20

def load_sequence_from_folder(sub_folder, seq_len=20):
    """
    Carica i primi 'seq_len' frame da una cartella soggetto
    formato: frame_0.pt, frame_1.pt, ...
    """
    frames = []
    print(f"   Caricamento frame da: {sub_folder}")
    
    for i in range(seq_len):
        fname = os.path.join(sub_folder, f"frame_{i}.pt")
        if not os.path.exists(fname):
            print(f"⚠️  Manca {fname}! Il soggetto ha meno di {seq_len} frame?")
            return None
        
        # Carica tensore (H, W, D, 1)
        # Nota: preprocessing_volume.py salva (96, 96, 96, 1)
        f_tensor = torch.load(fname)
        frames.append(f_tensor)
        
    # Concatena nel tempo -> (H, W, D, T)
    volume = torch.cat(frames, dim=3)
    
    # Aggiungi Batch e Canale -> (B, C, H, W, D, T)
    # (H, W, D, T) -> (1, 1, H, W, D, T)
    return volume.unsqueeze(0).unsqueeze(0)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔥 Verifica Dati Ufficiali su {device}")

    # 1. Trova un soggetto a caso
    subjects = sorted(os.listdir(DATA_ROOT))
    if not subjects:
        print(f"❌ Nessuna cartella soggetto trovata in {DATA_ROOT}")
        return

    test_sub = subjects[0] # Prendiamo il primo
    sub_path = os.path.join(DATA_ROOT, test_sub)
    print(f"📂 Soggetto test: {test_sub}")

    # 2. Carica i dati
    input_tensor = load_sequence_from_folder(sub_path, SEQ_LEN)
    if input_tensor is None: return
    
    input_tensor = input_tensor.to(device)
    print(f"   Input shape: {input_tensor.shape}") # Dovrebbe essere (1, 1, 96, 96, 96, 20)

    # 3. Inizializza Modello
    print("🏗️  Inizializzazione NeuroSTORM MAE...")
    model = NeuroSTORMMAE(
        img_size=IMG_SIZE,
        in_chans=1,
        embed_dim=EMBED_DIM,
        window_size=[4, 4, 4, 4],
        first_window_size=[2, 2, 2, 2],
        patch_size=[6, 6, 6, 1],
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        c_multiplier=2,
        last_layer_full_MSA=False,
        mask_ratio=0.5,
        spatial_mask="window", 
        time_mask="random"
    ).to(device)

    # 4. Carica Pesi
    if os.path.exists(CHECKPOINT_PATH):
        ckpt = torch.load(CHECKPOINT_PATH, map_location='cpu')
        sd = ckpt.get('state_dict', ckpt)
        clean_sd = {k.replace('model.', '').replace('net.', ''): v for k, v in sd.items()}
        try:
            model.load_state_dict(clean_sd, strict=False)
            print("✅ Pesi caricati correttamente.")
        except Exception as e:
            print(f"⚠️ Errore caricamento pesi: {e}")
    else:
        print(f"⚠️  ATTENZIONE: File pesi non trovato in {CHECKPOINT_PATH}")
        print("   Il test girerà con pesi random (Loss sarà alta).")

    model.eval()

    # 5. Inferenza
    with torch.no_grad():
        (pred, mask), loss = model(input_tensor)
        print(f"✨ Loss di Ricostruzione: {loss.item():.4f}")

    # 6. Plotting
    # Estrai slice centrale (z=48) al tempo t=10
    slice_idx = 48
    time_idx = 10
    
    img_in = input_tensor[0, 0, :, :, slice_idx, time_idx].cpu().numpy()
    img_out = pred[0, 0, :, :, slice_idx, time_idx].cpu().numpy()
    
    # Ruotiamo per visualizzazione (spesso i NIfTI sono ruotati di 90 gradi)
    img_in = np.rot90(img_in)
    img_out = np.rot90(img_out)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Originale
    axes[0].imshow(img_in, cmap='gray', vmin=-2, vmax=2)
    axes[0].set_title(f"Input (Official Preproc)\n{test_sub}")
    axes[0].axis('off')
    
    # Ricostruito
    axes[1].imshow(img_out, cmap='gray', vmin=-2, vmax=2)
    axes[1].set_title(f"Ricostruzione MAE\nLoss: {loss.item():.4f}")
    axes[1].axis('off')
    
    # Differenza
    diff = np.abs(img_in - img_out)
    im = axes[2].imshow(diff, cmap='inferno')
    axes[2].set_title("Errore")
    axes[2].axis('off')
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    
    plt.suptitle("Verifica Pipeline Ufficiale", fontsize=16)
    plt.tight_layout()
    plt.savefig(SAVE_IMG_PATH)
    print(f"\n📸 Immagine salvata: {SAVE_IMG_PATH}")
    print("   Controllala! Se l'input sembra un cervello MNI (non distorto) e la loss è < 0.6, è perfetto.")

if __name__ == "__main__":
    main()