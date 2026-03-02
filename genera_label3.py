import pandas as pd
import numpy as np
import os

# === CONFIGURAZIONE ===
# Percorso del file CBCL originale
CBCL_FILE_PATH = "/home/mfinocchiaro/miccai2026/NeuroSTORM/data/neurostorm_input_4d/metadata/data-2026-02-15T15_51_48.791Z.csv"

# File di output desiderato
OUTPUT_FILE = "labels_3class.csv"

# Soglia clinica per i T-score (65 è il cut-off clinico standard per il CBCL)
CLINICAL_THRESHOLD = 65

def main():
    if not os.path.exists(CBCL_FILE_PATH):
        print(f"❌ Errore: Il file {CBCL_FILE_PATH} non esiste.")
        return

    print(f"📂 Caricamento file CBCL: {CBCL_FILE_PATH}")
    
    # Leggiamo il file saltando la riga 1 (quella che contiene ",assessment","All", ecc.)
    try:
        df_cbcl = pd.read_csv(CBCL_FILE_PATH, skiprows=[1])
    except Exception as e:
        print(f"❌ Errore nella lettura del CSV: {e}")
        return

    print(f"✅ Lette {len(df_cbcl)} righe.")

    # 1. Identificazione Colonne (Usiamo i nomi esatti del tuo file)
    id_col = 'Identifiers'
    ap_col = 'CBCL,CBCL_AP_T'
    ext_col = 'CBCL,CBCL_Ext_T'

    # Controllo di sicurezza
    missing_cols = [col for col in [id_col, ap_col, ext_col] if col not in df_cbcl.columns]
    if missing_cols:
        print(f"❌ Colonne mancanti nel file: {missing_cols}")
        print(f"Colonne disponibili: {df_cbcl.columns.tolist()[:10]}...")
        return

    print(f"🔍 Colonne identificate con successo!")

    results = []

    # 2. Assegnazione delle Classi Cliniche
    for _, row in df_cbcl.iterrows():
        subj_raw = str(row[id_col])
        
        # Pulizia dell'ID (es. "NDARAA075AMK,assessment" diventa "NDARAA075AMK")
        subj_id = subj_raw.split(',')[0].strip()
            
        if not subj_id.startswith('NDAR'):
            continue

        try:
            ap_score = float(row[ap_col])
            ext_score = float(row[ext_col])
        except (ValueError, TypeError):
            # Salta soggetti con dati mancanti (NaN) in queste colonne
            continue

        # LOGICA CLINICA 3 CLASSI:
        if ap_score >= CLINICAL_THRESHOLD:
            if ext_score >= CLINICAL_THRESHOLD:
                label = 2  # ADHD-Combined
                ref_score = max(ap_score, ext_score)
            else:
                label = 1  # ADHD-Inattentive
                ref_score = ap_score
        else:
            label = 0  # Control (o ODD/CD puro non inattentivo)
            ref_score = ap_score

        results.append({
            'subject_id': subj_id,
            'label': label,
            't_score': ref_score
        })

    # 3. Pulizia e Salvataggio
    df_final = pd.DataFrame(results)
    
    if df_final.empty:
        print("❌ Nessun dato valido estratto. Verifica il contenuto del CSV.")
        return
        
    # Gestione duplicati: teniamo l'ultima valutazione se il soggetto compare più volte
    df_final = df_final.drop_duplicates(subset=['subject_id'], keep='last')

    print("\n📊 Distribuzione Classi finale:")
    print("0 (Control):", len(df_final[df_final['label']==0]))
    print("1 (ADHD-I):", len(df_final[df_final['label']==1]))
    print("2 (ADHD-C):", len(df_final[df_final['label']==2]))

    df_final.to_csv(OUTPUT_FILE, index=False)
    print(f"\n✅ File '{OUTPUT_FILE}' generato correttamente.")
    print("Esempio dati:")
    print(df_final.head(3))

if __name__ == "__main__":
    main()