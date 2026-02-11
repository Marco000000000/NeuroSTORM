import os
import subprocess
import time
import functools
print = functools.partial(print, flush=True)

def run_command(cmd):
    print(f"\n🚀 RUNNING: {cmd}")
    ret = subprocess.call(cmd, shell=True)
    if ret != 0:
        print("❌ ERROR detected. Stopping pipeline.")
        exit(1)

# ==============================================================================
# ⚙️ CONFIGURAZIONE GLOBALE
# ==============================================================================
PROJECT_PREFIX = "miccai_final" 
DATA_PATH = "./data/UCLA_MNI_to_TRs_minmax"
PRETRAINED_PATH = "./pretrained_models/fmrifound/last.ckpt"

# Parametri Architetturali (DEVONO MATCHARE IL CHECKPOINT)
# Fondamentale: embed_dim 36 per evitare mismatch pesi
ARCH_PARAMS = (
    "--model neurostorm "
    "--embed_dim 36 "
    "--depth 2 2 6 2 "
    "--c_multiplier 2 "
    "--last_layer_full_MSA True "
    "--clf_head_version v1 "
    "--img_size 96 96 96 20 "
    "--sequence_length 20 "
    "--patch_size 6 6 6 1 "
    "--first_window_size 4 4 4 4 "
    "--window_size 4 4 4 4 "
)

# Iperparametri di Training (Conservativi per stabilità SupCon)
BATCH_SIZE = 8
EPOCHS = 30
FOLDS = 5
LR = "5e-5" 

# Eventi da testare per OGNI configurazione
EVENTS = [
    ("boom", "EXPLODE"), 
    ("balloon", "ACCEPT"), 
    ("both", "EXPLODE,ACCEPT")
]

# ==============================================================================
# 🔄 FUNZIONE GENERICA DI ESECUZIONE
# ==============================================================================
def run_experiment(project_name, diag_mode, num_classes, evt_tag):
    print(f"\n{'='*60}")
    print(f"🧪 EXPERIMENT: {diag_mode} ({num_classes}-Class) | Input: {evt_tag}")
    print(f"📁 Folder: output/neurostorm/{project_name}")
    print(f"{'='*60}")

    # --- 1. TRAINING ---
    # Controlla se il training è già stato fatto per permettere il resume
    if not os.path.exists(f"output/neurostorm/{project_name}/cv_results_summary.txt"):
        train_cmd = (
            f"python train_miccai.py "
            f"--accelerator gpu --devices 1 --strategy ddp "
            f"--max_epochs {EPOCHS} "
            f"--dataset_name UCLA "
            f"--image_path {DATA_PATH} "
            f"--batch_size {BATCH_SIZE} --num_workers 4 "
            f"--project_name {project_name} "
            f"--load_model_path {PRETRAINED_PATH} "
            f"--num_folds {FOLDS} "
            f"--use_supcon --lambda_contrast 0.5 "
            f"--learning_rate {LR} --seed 1234 "
            # Parametri Variabili
            f"--num_classes {num_classes} "
            f"--target_diagnosis {diag_mode} "
            f"--target_events \"{evt_tag}\" "
            # Architettura
            f"{ARCH_PARAMS}" 
        )
        run_command(train_cmd)
    else:
        print("✅ Training already completed. Skipping.")

    # --- 2. INFERENCE (Fold by Fold) ---
    for fold in range(FOLDS):
        res_file = f"output/neurostorm/{project_name}/results_fold_{fold}.csv"
        if not os.path.exists(res_file):
            inf_cmd = (
                f"python inference_miccai.py "
                f"--project_name {project_name} "
                f"--image_path {DATA_PATH} "
                f"--dataset_name UCLA "
                f"--fold_index {fold} "
                f"--batch_size 4 --num_workers 0 "
                # Parametri Variabili che devono matchare il training
                f"--num_classes {num_classes} "
                f"--target_diagnosis {diag_mode} "
                f"--target_events \"{evt_tag}\" "
                f"{ARCH_PARAMS}"
            )
            run_command(inf_cmd)

    # --- 3. MERGE & PLOT ---
    print("📊 Generating Reports...")
    merge_py = f"""
import pandas as pd, glob, os
base='output/neurostorm/{project_name}'
files=glob.glob(os.path.join(base, 'results_fold_*.csv'))
if files:
    df = pd.concat([pd.read_csv(f) for f in files])
    df.to_csv(os.path.join(base, 'final_results_merged.csv'), index=False)
"""
    os.system(f"python -c \"{merge_py}\"")
    
    if os.path.exists(f"output/neurostorm/{project_name}/final_results_merged.csv"):
        run_command(f"python final_plots_and_metrics.py --project_name {project_name} --csv_file final_results_merged.csv")


# ==============================================================================
# 🚀 MAIN EXECUTION LOOP
# ==============================================================================

print("🔥 STARTING SUPER MASTER PIPELINE (MICCAI EDITION)")

# --- FASE 1: BINARY TASKS (One vs Control) ---
# Schizofrenia, Bipolare, ADHD
BINARY_DIAGNOSES = ["SCHZ", "BIPOLAR", "ADHD"]

for diag in BINARY_DIAGNOSES:
    for evt_name, evt_tag in EVENTS:
        # Nome univoco: miccai_final_SCHZ_boom
        p_name = f"{PROJECT_PREFIX}_{diag}_{evt_name}"
        
        run_experiment(
            project_name=p_name,
            diag_mode=diag,
            num_classes=2,
            evt_tag=evt_tag
        )

# --- FASE 2: MULTICLASS TASK (4-Way Classification) ---
# Control vs Schizo vs Bipolar vs ADHD
for evt_name, evt_tag in EVENTS:
    # Nome univoco: miccai_final_MULTICLASS_boom
    p_name = f"{PROJECT_PREFIX}_MULTICLASS_{evt_name}"
    
    run_experiment(
        project_name=p_name,
        diag_mode="MULTICLASS",
        num_classes=4,
        evt_tag=evt_tag
    )

print("\n🎉🎉🎉 TUTTI GLI ESPERIMENTI SONO COMPLETATI! 🎉🎉🎉")
print("Controlla la cartella 'output/neurostorm/' per i risultati.")