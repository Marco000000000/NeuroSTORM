import pandas as pd
import numpy as np
import os
import glob
import torch

# --- CONFIGURAZIONE ---
RAW_DIR = "ucla_raw"                # Dove sta participants.tsv
DATA_DIR = "ucla_neurostorm_final"  # Dove sono i file .pt preprocessati
OUTPUT_DIR = "ucla_labels"          # Dove salviamo le etichette

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 1. Carica il file delle diagnosi (participants.tsv)
    part_path = os.path.join(RAW_DIR, "participants.tsv")
    if not os.path.exists(part_path):
        print(f"❌ Errore: Non trovo {part_path}")
        print("Assicurati che ucla_raw contenga il file participants.tsv originale.")
        return

    print(f"📖 Lettura diagnosi da {part_path}...")
    df = pd.read_csv(part_path, sep='\t')
    
    # Standardizziamo i nomi delle colonne (a volte è 'diagnosis', a volte 'group')
    if 'diagnosis' not in df.columns:
        # Fallback: cerca colonne simili
        possible_cols = [c for c in df.columns if 'diag' in c.lower() or 'group' in c.lower()]
        if possible_cols:
            print(f"⚠️ Colonna 'diagnosis' non trovata. Uso '{possible_cols[0]}' come diagnosi.")
            df.rename(columns={possible_cols[0]: 'diagnosis'}, inplace=True)
        else:
            print("❌ Impossibile trovare la colonna della diagnosi nel TSV.")
            return

    # Mappa delle Classi (Stringa -> Numero)
    # Ordine suggerito: CONTROL è sempre 0
    label_map = {
        'CONTROL': 0,
        'SCHZ': 1,
        'BIPOLAR': 2,
        'ADHD': 3
    }
    
    # 2. Trova i soggetti che abbiamo effettivamente preprocessato
    processed_files = glob.glob(os.path.join(DATA_DIR, "*.pt"))
    processed_subs = [os.path.basename(f).split('.')[0] for f in processed_files]
    
    print(f"🔍 Trovati {len(processed_subs)} soggetti preprocessati (.pt).")
    
    # 3. Costruiamo il dataset finale
    final_data = []
    
    for sub_id in processed_subs:
        # Cerca la riga corrispondente nel TSV
        # Nota: nel TSV i soggetti sono spesso 'sub-10159', ma a volte solo '10159'
        # Puliamo entrambe le stringhe per sicurezza
        clean_sub = sub_id.replace('sub-', '')
        
        row = df[df['participant_id'].astype(str).str.contains(clean_sub)]
        
        if not row.empty:
            diag_str = row.iloc[0]['diagnosis']
            
            # Gestione casi strani o null
            if pd.isna(diag_str):
                continue
                
            if diag_str in label_map:
                label = label_map[diag_str]
                final_data.append({'sub_id': sub_id, 'label': label, 'diagnosis': diag_str})
        else:
            print(f"⚠️ Nessuna diagnosi trovata per {sub_id}")

    # Converti in DataFrame
    df_labels = pd.DataFrame(final_data)
    
    # 4. Statistiche e Bilanciamento
    print("\n📊 DISTRIBUZIONE CLASSI:")
    counts = df_labels['diagnosis'].value_counts()
    print(counts)
    
    # Calcolo Class Weights (Inverse Frequency)
    # Formula: N_samples / (N_classes * Count_class)
    total_samples = len(df_labels)
    n_classes = len(label_map)
    
    # Ordiniamo i pesi in base all'indice 0, 1, 2, 3
    weights = []
    for diag_name, idx in label_map.items():
        count = len(df_labels[df_labels['diagnosis'] == diag_name])
        if count > 0:
            w = total_samples / (n_classes * count)
        else:
            w = 0.0
        weights.append(w)
        print(f"   ⚖️ Peso per {diag_name} (Classe {idx}): {w:.4f}")
    
    weights_tensor = torch.tensor(weights, dtype=torch.float32)

    # 5. Salvataggio
    # Salviamo un CSV leggibile
    csv_path = os.path.join(OUTPUT_DIR, "subject_diagnosis.csv")
    df_labels.to_csv(csv_path, index=False)
    
    # Salviamo i pesi per PyTorch
    weights_path = os.path.join(OUTPUT_DIR, "diagnosis_weights.pt")
    torch.save(weights_tensor, weights_path)
    
    # Salviamo un dizionario Python rapido per il Dataloader
    # { 'sub-10159': 0, 'sub-10171': 1, ... }
    label_dict = pd.Series(df_labels.label.values, index=df_labels.sub_id).to_dict()
    dict_path = os.path.join(OUTPUT_DIR, "diagnosis_map.npy")
    np.save(dict_path, label_dict)

    print(f"\n✅ Finito! File salvati in {OUTPUT_DIR}:")
    print(f"   1. {os.path.basename(csv_path)} (Tabella completa)")
    print(f"   2. {os.path.basename(weights_path)} (Pesi per Loss Function)")
    print(f"   3. {os.path.basename(dict_path)} (Mappa veloce per Dataloader)")

if __name__ == "__main__":
    main()