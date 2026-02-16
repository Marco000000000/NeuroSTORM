import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

# Assicuriamoci che python trovi i moduli di NeuroSTORM
sys.path.append(os.getcwd()) 

try:
    from models.neurostorm import NeuroSTORMMAE
except ImportError:
    print("❌ Errore importazione NeuroSTORMMAE. Assicurati di essere nella cartella root di NeuroSTORM.")
    sys.exit(1)

# --- CONFIGURAZIONE HBN ---
# Path dove preprocessing_volume.py ha salvato i dati
DATA_ROOT = "data/neurostorm_input_4d/img"
CHECKPOINT_PATH = "/home/mfinocchiaro/miccai2026/NeuroSTORM/output/neurostorm/hbn_mae_domain_adaptation/checkpt-epoch=09-valid_loss=0.09.ckpt" 

# Path del checkpoint. 
# ⚠️ MODIFICA QUI: Metti il percorso del tuo checkpoint appena allenato 
# (es. "lightning_logs/version_0/checkpoints/epoch=99-step=....ckpt")
# Se usi quello sotto, testerai il modello BASE prima dell'adattamento ai tuoi dati.
# CHECKPOINT_PATH = "/home/mfinocchiaro/miccai2026/NeuroSTORM/pretrained_models/fmrifound/pt_fmrifound_mae_ratio0.5.ckpt"
SAVE_IMG_PATH = "verify_hbn_result.png"

# Parametri modello (DEVONO coincidere con il tuo script run_hbn_mae_adapt.slurm)
EMBED_DIM = 36
IMG_SIZE = (96, 96, 96, 20) # (H, W, D, T)
SEQ_LEN = 20
WINDOW_SIZE = [4, 4, 4, 4]       # Come nel tuo script bash
FIRST_WINDOW_SIZE = [4, 4, 4, 4] # Come nel tuo script bash

def load_sequence_from_folder(sub_folder, seq_len=20):
    """
    Carica i primi 'seq_len' frame .pt da una cartella soggetto
    """
    frames = []
    sub_name = os.path.basename(sub_folder)
    print(f" 📂 Caricamento dati da: {sub_name}")
    
    # Cerchiamo i file frame_X.pt in ordine
    found_count = 0
    # Proviamo a cercare in un range ampio nel caso i frame non partano da 0
    for i in range(500): 
        fname = os.path.join(sub_folder, f"frame_{i}.pt")
        if os.path.exists(fname):
            try:
                # Carica tensore e converti in float32
                f_tensor = torch.load(fname).float()
                # Il preprocessing salva (H, W, D, 1) o (H, W, D)
                # Assicuriamoci che abbia l'ultima dimensione temporale
                if len(f_tensor.shape) == 3:
                    f_tensor = f_tensor.unsqueeze(-1)
                
                frames.append(f_tensor)
                found_count += 1
                if found_count == seq_len:
                    break
            except Exception as e:
                print(f"   Errore lettura {fname}: {e}")
        
    if len(frames) < seq_len:
        print(f"⚠️  Saltato: Trovati solo {len(frames)} frame (richiesti {seq_len}) in {sub_name}.")
        return None
        
    # Concatena nel tempo -> (H, W, D, T)
    volume = torch.cat(frames, dim=3)
    
    # Aggiungi Batch e Canale per il modello -> (B, C, H, W, D, T)
    # Da (96, 96, 96, 20) a (1, 1, 96, 96, 96, 20)
    return volume.unsqueeze(0).unsqueeze(0)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔥 Verifica NeuroSTORM su HBN (Device: {device})")

    # 1. Trova un soggetto a caso
    if not os.path.exists(DATA_ROOT):
        print(f"❌ Errore: Cartella dati non trovata: {DATA_ROOT}")
        return

    subjects = sorted([d for d in os.listdir(DATA_ROOT) if os.path.isdir(os.path.join(DATA_ROOT, d))])
    if not subjects:
        print(f"❌ Nessuna cartella soggetto trovata in {DATA_ROOT}")
        return

    # Prendiamo un soggetto random per vedere se generalizza
    test_sub = np.random.choice(subjects)
    sub_path = os.path.join(DATA_ROOT, test_sub)
    
    # 2. Carica i dati
    input_tensor = load_sequence_from_folder(sub_path, SEQ_LEN)
    if input_tensor is None: return
    
    input_tensor = input_tensor.to(device)
    print(f" ✅ Input shape: {input_tensor.shape}") 

    # 3. Inizializza Modello
    print("🏗️  Inizializzazione Modello...")
    model = NeuroSTORMMAE(
        img_size=IMG_SIZE,
        in_chans=1,
        embed_dim=EMBED_DIM,
        window_size=WINDOW_SIZE,
        first_window_size=FIRST_WINDOW_SIZE,
        patch_size=[6, 6, 6, 1], 
        depths=[2, 2, 6, 2],     
        num_heads=[3, 6, 12, 24], 
        c_multiplier=2,
        last_layer_full_MSA=False,
        mask_ratio=0.9,          
        spatial_mask="window", 
        time_mask="random"
    ).to(device)

    # 4. Carica Pesi
    if os.path.exists(CHECKPOINT_PATH):
        print(f"📥 Loading weights: {CHECKPOINT_PATH}")
        try:
            ckpt = torch.load(CHECKPOINT_PATH, map_location='cpu')
            sd = ckpt.get('state_dict', ckpt)
            # Pulisce le chiavi dai prefissi di Lightning/DDP
            clean_sd = {k.replace('model.', '').replace('net.', ''): v for k, v in sd.items()}
            
            msg = model.load_state_dict(clean_sd, strict=False)
            print(f"✅ Pesi caricati con successo.")
        except Exception as e:
            print(f"⚠️  Errore critico caricamento pesi: {e}")
            return
    else:
        print(f"⚠️  FILE CHECKPOINT NON TROVATO: {CHECKPOINT_PATH}")
        print("   Il test non può procedere senza pesi.")
        return

    model.eval()

    # 5. Inferenza
    print("🔄 Esecuzione inferenza...")
    with torch.no_grad():
        # Il modello MAE restituisce: (pred, mask), loss
        (pred, mask), loss = model(input_tensor)
        print(f"✨ Loss di Ricostruzione: {loss.item():.4f}")

    # 6. Plotting (Slice Assiale Centrale)
    slice_idx = IMG_SIZE[0] // 2 
    time_idx = SEQ_LEN // 2
    
    # Converti in numpy
    img_in = input_tensor[0, 0, :, :, slice_idx, time_idx].cpu().numpy()
    img_out = pred[0, 0, :, :, slice_idx, time_idx].cpu().numpy()
    
    # Ruota per visualizzazione corretta (anteriore in alto)
    img_in = np.rot90(img_in)
    img_out = np.rot90(img_out)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Originale
    # vmin/vmax a -2/2 perché i dati sono Z-normalized
    axes[0].imshow(img_in, cmap='gray', vmin=-2.5, vmax=2.5) 
    axes[0].set_title(f"Originale (HBN)\n{test_sub}")
    axes[0].axis('off')
    
    # Ricostruito
    axes[1].imshow(img_out, cmap='gray', vmin=-2.5, vmax=2.5)
    axes[1].set_title(f"Ricostruzione NeuroSTORM\nLoss: {loss.item():.4f}")
    axes[1].axis('off')
    
    # Errore
    diff = np.abs(img_in - img_out)
    im = axes[2].imshow(diff, cmap='inferno', vmin=0, vmax=2)
    axes[2].set_title("Differenza (Errore)")
    axes[2].axis('off')
    
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    plt.suptitle(f"Verifica Training MAE su HBN", fontsize=16)
    plt.tight_layout()
    
    plt.savefig(SAVE_IMG_PATH)
    print(f"\n📸 Risultato salvato in: {os.path.abspath(SAVE_IMG_PATH)}")
    print("   Se l'immagine centrale assomiglia a un cervello, il training ha funzionato!")

if __name__ == "__main__":
    main()