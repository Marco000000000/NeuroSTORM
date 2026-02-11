import os
import glob
import shutil

# CONFIGURAZIONE
RAW_BIDS_DIR = "ucla_raw"  # La tua cartella originale con i dati BIDS
DEST_DIR = "data/UCLA_MNI_to_TRs_minmax/metadata/events" # Dove li mettiamo

os.makedirs(DEST_DIR, exist_ok=True)

print("🔍 Cerco file events.tsv...")
# Cerca in modo ricorsivo: ucla_raw/sub-XXX/func/*events.tsv
files = glob.glob(os.path.join(RAW_BIDS_DIR, "**", "*events.tsv"), recursive=True)

count = 0
for fpath in files:
    # Nome file: sub-10159_task-bart_events.tsv
    fname = os.path.basename(fpath)
    # Estrai subject id: sub-10159
    sub_id = fname.split('_')[0]
    
    # Copia rinominando in modo standard: sub-10159.tsv
    dest_path = os.path.join(DEST_DIR, f"{sub_id}.tsv")
    shutil.copy(fpath, dest_path)
    count += 1

print(f"✅ Copiati {count} file eventi in {DEST_DIR}")