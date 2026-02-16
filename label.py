import os
import glob
import pandas as pd
import numpy as np

# --- CONFIGURAZIONE ---
csv_path = "/home/mfinocchiaro/miccai2026/NeuroSTORM/data/neurostorm_input_4d/metadata/data-2026-02-15T15_51_48.791Z.csv" 
data_dir = "/home/mfinocchiaro/miccai2026/NeuroSTORM/data/neurostorm_input_4d/img"
output_filename = "subject_labels.csv"

def main():
    print(">>> 1. Scansione cartelle e pulizia ID Cartelle...")
    subject_dirs = glob.glob(os.path.join(data_dir, "sub-*"))
    
    subjects_in_folder = set()
    
    for p in subject_dirs:
        folder_name = os.path.basename(p)
        clean_name = folder_name.replace("sub-", "")
        if "_task" in clean_name:
            clean_id = clean_name.split("_task")[0]
        else:
            clean_id = clean_name
        subjects_in_folder.add(clean_id)
            
    print(f"Trovate {len(subjects_in_folder)} soggetti unici nelle cartelle.")

    print("\n>>> 2. Caricamento e Pulizia CSV...")
    try:
        df = pd.read_csv(csv_path, low_memory=False)
    except Exception as e:
        print(f"Errore caricamento CSV: {e}")
        return

    df.columns = df.columns.str.strip()
    id_col = 'Identifiers'      
    target_col = 'CBCL,CBCL_AP_T' 

    if id_col not in df.columns or target_col not in df.columns:
        print(f"ERRORE: Colonne non trovate.")
        return

    # Pulizia ID nel CSV
    df[id_col] = df[id_col].astype(str).str.replace(',assessment', '', regex=False)
    
    # Pulizia Score
    df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
    
    # Creiamo un set dei soggetti che hanno dati VALIDI nel CSV
    df_clean = df.dropna(subset=[target_col]).copy()
    subjects_in_csv = set(df_clean[id_col].unique())

    # --- 3. ANALISI DEI SOGGETTI MANCANTI ---
    print("\n>>> 3. Confronto e Soggetti Mancanti...")
    
    # Calcolo della differenza tra insiemi: (Cartella) - (CSV)
    missing_subjects = subjects_in_folder - subjects_in_csv
    
    print(f"Soggetti nella cartella: {len(subjects_in_folder)}")
    print(f"Soggetti nel CSV (con score validi): {len(subjects_in_csv)}")
    print(f"Soggetti presenti fisicamente ma ASSENTI nel CSV (o senza score): {len(missing_subjects)}")
    
    if len(missing_subjects) > 0:
        print("\n--- LISTA SOGGETTI NON TROVATI NEL FILE METADATA ---")
        # Ordiniamo la lista per leggibilità
        for subj in sorted(list(missing_subjects)):
            print(subj)
        print("----------------------------------------------------\n")
    else:
        print("\nOttimo! Tutti i soggetti nelle cartelle hanno una corrispondenza nel CSV.")

    # --- 4. PROCEDIAMO CON IL SALVATAGGIO DEI SOGGETTI VALIDI ---
    final_df = df_clean[df_clean[id_col].isin(subjects_in_folder)].copy()
    final_df = final_df.drop_duplicates(subset=[id_col])
    
    final_df['label'] = np.where(final_df[target_col] >= 65, 1, 0)
    
    n_0 = len(final_df[final_df['label'] == 0])
    n_1 = len(final_df[final_df['label'] == 1])
    
    print("\n--- RISULTATO FINALE (Soggetti salvati) ---")
    print(f"Classe 0 (Controlli): {n_0}")
    print(f"Classe 1 (ADHD Border/Clinico): {n_1}")
    
    out_df = final_df[[id_col, 'label', target_col]].rename(columns={id_col: 'subject_id', target_col: 't_score'})
    out_df.to_csv(output_filename, index=False)
    print(f"File salvato: {output_filename}")

if __name__ == "__main__":
    main()