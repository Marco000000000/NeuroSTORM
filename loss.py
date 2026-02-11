import pandas as pd
import os
import glob
import numpy as np

# --- CONFIGURAZIONE ---
META_FILE = "data/UCLA_MNI_to_TRs_minmax/metadata/ucla-rest.csv"
EVENT_DIR = "data/UCLA_MNI_to_TRs_minmax/metadata/events"
TARGET_ACTION = "EXPLODE"

def main():
    print(f"📊 Analisi Bilanciamento Eventi: '{TARGET_ACTION}'")
    
    # 1. Carica le diagnosi
    if not os.path.exists(META_FILE):
        print(f"❌ Errore: Manca il file {META_FILE}")
        return
        
    meta_df = pd.read_csv(META_FILE)
    # Assicuriamoci che subject_id sia stringa
    meta_df['subject_id'] = meta_df['subject_id'].astype(str)
    
    # Dizionario per accumulare i conteggi: {Classe: [lista_conteggi_per_soggetto]}
    stats = {label: [] for label in meta_df['diagnosis'].unique()}
    
    print(f"   Soggetti nel file metadata: {len(meta_df)}")
    
    missing_events = 0
    zero_explode = 0
    
    # 2. Itera sui soggetti
    for idx, row in meta_df.iterrows():
        sub_id = row['subject_id']
        diagnosis = row['diagnosis']
        
        # Cerca file eventi (sub-XXXXX.tsv)
        tsv_path = os.path.join(EVENT_DIR, f"{sub_id}.tsv")
        
        count = 0
        if os.path.exists(tsv_path):
            try:
                evt_df = pd.read_csv(tsv_path, sep='\t')
                evt_df.columns = [c.lower() for c in evt_df.columns]
                
                if 'action' in evt_df.columns:
                    # Conta quante volte action == EXPLODE
                    expl_events = evt_df[evt_df['action'].astype(str).str.upper() == TARGET_ACTION]
                    count = len(expl_events)
                else:
                    # Se non c'è la colonna action, 0 eventi
                    pass
            except Exception:
                pass
        else:
            missing_events += 1
            
        if count == 0:
            zero_explode += 1
            
        stats[diagnosis].append(count)

    # 3. Stampa Report
    print("\n" + "="*50)
    print(f"RISULTATI PER CLASSE (Eventi '{TARGET_ACTION}')")
    print("="*50)
    print(f"{'CLASSE':<15} | {'SOGGETTI':<10} | {'TOT EVENTI':<12} | {'MEDIA/SOGG':<12}")
    print("-" * 55)
    
    total_samples = 0
    
    for label, counts in stats.items():
        n_subj = len(counts)
        tot_ev = sum(counts)
        avg_ev = np.mean(counts) if n_subj > 0 else 0
        total_samples += tot_ev
        
        print(f"{label:<15} | {n_subj:<10} | {tot_ev:<12} | {avg_ev:<12.1f}")

    print("-" * 55)
    print(f"TOTALE CAMPIONI TRAINING: {total_samples}")
    print(f"Soggetti senza file eventi: {missing_events}")
    print(f"Soggetti con 0 esplosioni: {zero_explode}")
    print("="*50)

    # Analisi Pesi Consigliati
    print("\n⚖️  PESI CONSIGLIATI (Inverse Frequency):")
    if total_samples > 0:
        for label, counts in stats.items():
            tot_ev = sum(counts)
            if tot_ev > 0:
                weight = total_samples / (len(stats) * tot_ev)
                print(f"   {label}: {weight:.4f}")
            else:
                print(f"   {label}: 0.0000 (Nessun dato!)")

if __name__ == "__main__":
    main()