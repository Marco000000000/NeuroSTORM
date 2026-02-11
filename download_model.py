from huggingface_hub import hf_hub_download
import os

# Crea la cartella se non esiste
save_dir = "NeuroSTORM/pretrained_models"
os.makedirs(save_dir, exist_ok=True)

print(f"⬇️ Avvio download in {save_dir}...")

# Scarica il checkpoint MAE (Ratio 0.5 è quello standard bilanciato)
file_path = hf_hub_download(
    repo_id="zxcvb20001/NeuroSTORM",
    filename="fmrifound/pt_fmrifound_mae_ratio0.8.ckpt",
    local_dir=save_dir
)

print(f"✅ Fatto! File salvato in: {file_path}")