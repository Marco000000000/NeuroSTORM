import os
import glob
import sys

# --- CONFIGURAZIONE ---
# Cartella output di fMRIPrep (dove ci sono le cartelle sub-XXXX)
FMRIPREP_ROOT = "ucla_preproc" 

# Cartella destinazione per il preprocessing di NeuroSTORM
DEST_DIR = "data_raw_flat"

def main():
    if not os.path.exists(DEST_DIR):
        os.makedirs(DEST_DIR)
        print(f"📂 Creata cartella destinazione: {DEST_DIR}")

    # Cerca tutti i soggetti
    subjects = glob.glob(os.path.join(FMRIPREP_ROOT, "sub-*"))
    print(f"🔍 Scansione di {len(subjects)} soggetti in {FMRIPREP_ROOT}...")

    count = 0
    skipped = 0
    
    for subj_dir in subjects:
        subj_id = os.path.basename(subj_dir) # es: sub-10171
        
        # 1. Cerca il file BOLD in spazio MNI con risoluzione 2mm (res-2)
        # Pattern esatto visto nell'immagine:
        # sub-XXXXX_task-bart_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz
        
        # Usiamo un glob flessibile ma specifico per MNI e preproc
        search_pattern = os.path.join(subj_dir, "func", "*space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz")
        bold_candidates = glob.glob(search_pattern)
        
        # Se non lo trova in func/, prova ricorsivamente (a volte fmriprep cambia struttura)
        if not bold_candidates:
            search_pattern_rec = os.path.join(subj_dir, "**", "*space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz")
            bold_candidates = glob.glob(search_pattern_rec, recursive=True)

        if not bold_candidates:
            print(f"⚠️  BOLD MNI (res-2) non trovato per {subj_id}")
            skipped += 1
            continue
            
        bold_src = os.path.abspath(bold_candidates[0])
        
        # 2. Cerca la MASCHERA corrispondente
        # La maschera ha lo stesso nome ma finisce con 'desc-brain_mask.nii.gz' invece di 'desc-preproc_bold.nii.gz'
        mask_src = bold_src.replace("desc-preproc_bold.nii.gz", "desc-brain_mask.nii.gz")
        
        if not os.path.exists(mask_src):
            print(f"⚠️  Maschera non trovata per {subj_id} (Cercavo: {os.path.basename(mask_src)})")
            skipped += 1
            continue

        # --- CREAZIONE LINK (Rinomina per NeuroSTORM) ---
        # NeuroSTORM vuole: 
        # 1. ID soggetto = primi 9 caratteri del nome file (sub-XXXXX) -> OK
        # 2. Maschera = nomefile[:-14] + 'brainmask.nii.gz'
        
        # Creiamo un nome "finto" per il link che soddisfi la regola dei 14 caratteri finali.
        # "taskx.nii.gz" è lungo 12 caratteri... troppo corto.
        # "_taskx.nii.gz" è lungo 13 caratteri... troppo corto.
        # "__taskx.nii.gz" è lungo 14 caratteri. PERFETTO.
        
        # Nome link BOLD: sub-10171__taskx.nii.gz
        suffix = "__taskx.nii.gz"
        dest_bold_name = f"{subj_id}{suffix}"
        
        # Nome link MASK: sub-10171brainmask.nii.gz
        dest_mask_name = f"{subj_id}brainmask.nii.gz"
        
        dest_bold = os.path.join(DEST_DIR, dest_bold_name)
        dest_mask = os.path.join(DEST_DIR, dest_mask_name)

        # Rimuovi vecchi link se esistono
        if os.path.lexists(dest_bold): os.remove(dest_bold)
        if os.path.lexists(dest_mask): os.remove(dest_mask)
        
        os.symlink(bold_src, dest_bold)
        os.symlink(mask_src, dest_mask)
        count += 1

    print("-" * 40)
    print(f"✅ Link creati con successo per {count} soggetti.")
    if skipped > 0:
        print(f"⚠️  Saltati {skipped} soggetti (file mancanti).")
    
    print("\n🚀 PROSSIMO STEP:")
    print(f"   Esegui: python datasets/preprocessing_volume.py --dataset_name ucla --load_root {DEST_DIR} --save_root ./data/UCLA_MNI_to_TRs_minmax --num_processes 4")

if __name__ == "__main__":
    main()