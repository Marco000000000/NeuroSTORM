import os
import pandas as pd
import numpy as np

# Configurazione
IMG_DIR = "data/UCLA_MNI_to_TRs_minmax/img"
META_DIR = "data/UCLA_MNI_to_TRs_minmax/metadata"
LABEL_MAP_PATH = "ucla_labels/diagnosis_map.npy" 

os.makedirs(META_DIR, exist_ok=True)

if not os.path.exists(IMG_DIR):
    print(f"❌ Errore: La cartella {IMG_DIR} non esiste. Lancia prima preprocessing_volume.py!")
    exit()

processed_subjects = sorted(os.listdir(IMG_DIR))
label_map = np.load(LABEL_MAP_PATH, allow_pickle=True).item()
int_to_str = {0: 'CONTROL', 1: 'SCHZ', 2: 'BIPOLAR', 3: 'ADHD'}

csv_rows = []
print(f"Generazione metadati per {len(processed_subjects)} soggetti...")

for sub_id in processed_subjects:
    # Il nome cartella è tipo 'sub-10159'
    if sub_id in label_map:
        label_int = label_map[sub_id]
        if label_int in int_to_str:
            csv_rows.append({
                'subject_id': sub_id,
                'diagnosis': int_to_str[label_int],
                'gender': 'M', 
                'age': 0
            })

df = pd.DataFrame(csv_rows)
out_path = os.path.join(META_DIR, "ucla-rest.csv")
df.to_csv(out_path, index=False)
print(f"✅ Creato {out_path} con {len(df)} righe.")