import os
import torch
import torch.nn as nn
import numpy as np
import argparse
import pandas as pd
import glob

# === CONFIGURAZIONE ===
FMRI_FEAT_DIR = "/home/mfinocchiaro/miccai2026/NeuroSTORM/hbn_data/neurostorm_features_split_50"
VIDEO_FEAT_PATH = "/home/mfinocchiaro/miccai2026/NeuroSTORM/data/neurostorm_input_4d/dataset_fmri_features/features_brain_target_250.pt" 
MODELS_BASE_DIR = "trained_aligners_split_50_10"  # Coerente con il training
RESULTS_FILE = "metrics_split_50.csv"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
INPUT_DIM = 2304
SEQ_LEN = 20
HRF_SHIFT = 0  # IMPORTANTE: Deve essere uguale al training!

# --- MODELLO SINCRONIZZATO ---
# Rimosso logit_scale perché il tuo training non lo usava
class LinearAligner(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        
    def forward(self, x): 
        return self.linear(x)

def load_data_split(subject_id, split_name, video_targets):
    subj_path = os.path.join(FMRI_FEAT_DIR, split_name, subject_id)
    files = sorted(glob.glob(os.path.join(subj_path, "window_*.pt")))
    
    if not files: return None, None
    
    X_list = []
    Y_list = []
    max_video_idx = video_targets.shape[0]

    for fpath in files:
        try:
            start_idx = int(os.path.basename(fpath).split('_')[1].split('.')[0])
            target_idx = (start_idx + 10) - HRF_SHIFT
            
            if target_idx < 0 or target_idx >= max_video_idx: continue
            
            # Caricamento sicuro
            fmri = torch.load(fpath, map_location='cpu', weights_only=True).float().view(-1, SEQ_LEN)
            X_list.append(fmri[:, 10]) 
            Y_list.append(video_targets[target_idx])
        except: continue
        
    if not X_list: return None, None
    return torch.stack(X_list).to(DEVICE), torch.stack(Y_list).to(DEVICE)

def compute_metrics(preds, targets):
    n_samples = preds.shape[0]
    preds_norm = torch.nn.functional.normalize(preds, p=2, dim=1)
    targets_norm = torch.nn.functional.normalize(targets, p=2, dim=1)
    
    # Calcolo su float32 per evitare errori di tipo
    similarity_matrix = torch.mm(preds_norm, targets_norm.T)
    
    top1 = 0
    ranks = []
    hits = 0
    pairs = 0
    
    for i in range(n_samples):
        scores = similarity_matrix[i, :]
        true_score = scores[i].item()
        # argsort decrescente
        sorted_indices = torch.argsort(scores, descending=True)
        # Rango (0-based)
        rank = (sorted_indices == i).nonzero(as_tuple=True)[0].item()
        ranks.append(rank)
        if rank == 0: top1 += 1
        
        # Pairwise Accuracy
        other_scores = scores[torch.arange(n_samples) != i]
        if len(other_scores) > 0:
            hits += (true_score > other_scores).sum().item()
            pairs += len(other_scores)

    return {
        "pairwise_acc": hits / pairs if pairs > 0 else 0,
        "top1_acc": top1 / n_samples,
        "mean_rank": np.mean(ranks),
        "n_samples": n_samples
    }

def compute_generalization(preds, true_targets, train_pool):
    n_test = preds.shape[0]
    n_train = train_pool.shape[0]
    
    preds_norm = torch.nn.functional.normalize(preds, p=2, dim=1)
    true_targets_norm = torch.nn.functional.normalize(true_targets, p=2, dim=1)
    train_pool_norm = torch.nn.functional.normalize(train_pool, p=2, dim=1)
    
    sim_true = torch.sum(preds_norm * true_targets_norm, dim=1)
    
    total_hits = 0
    total_comps = 0
    N_BOOTSTRAP = 500
    
    for i in range(n_test):
        score_correct = sim_true[i].item()
        rand_idx = torch.randint(0, n_train, (N_BOOTSTRAP,), device=DEVICE)
        distractors = train_pool_norm[rand_idx]
        
        scores_wrong = torch.mm(preds_norm[i].unsqueeze(0), distractors.T).squeeze()
        total_hits += (score_correct > scores_wrong).sum().item()
        total_comps += N_BOOTSTRAP
        
    return total_hits / total_comps

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=str, required=True)
    args = parser.parse_args()
    
    if not os.path.exists(VIDEO_FEAT_PATH): 
        print("❌ File Video Features non trovato")
        return
    video_data = torch.load(VIDEO_FEAT_PATH, weights_only=True)
    
    variants = ["clip_single", "clip_avg", "clip_dominant", "alex_single", "alex_avg", "alex_dominant"]
    results = []
    
    print(f"📊 Valutazione SPLIT 50/10/40 per {args.subject}")
    print(f"{'Variant':<15} | {'Split':<5} | {'2-Way':<8} | {'Top-1':<8} | {'Gen(vsPool)':<12}")
    print("-" * 65)
    
    for var in variants:
        if var not in video_data: continue
        
        # Verifica esistenza modello
        model_path = os.path.join(MODELS_BASE_DIR, args.subject, var, "best_model.pth")
        if not os.path.exists(model_path): 
            # print(f"⚠️ Manca modello per {var}")
            continue
        
        targets = video_data[var].to(DEVICE)
        model = LinearAligner(INPUT_DIM, targets.shape[1]).to(DEVICE)
        
        try:
            # Caricamento strict=True per essere sicuri che l'architettura sia identica
            model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
        except Exception as e:
            print(f"❌ Errore caricamento {var}: {e}")
            continue
            
        model.eval()
        
        # Pool di training (primi 50% dei target video, dato lo split 0.5)
        # Nota: Nel training abbiamo usato split 0.5, quindi il train pool sono i primi 50% dei frame
        train_pool = targets[:int(targets.shape[0]*0.5)]

        # --- CICLO SU TUTTI GLI SPLIT ---
        for split in ["train", "val", "test"]:
            X, Y = load_data_split(args.subject, split, targets)
            
            if X is None or len(X) < 2:
                continue
                
            with torch.no_grad(): preds = model(X)
            
            std = compute_metrics(preds, Y)
            gen = compute_generalization(preds, Y, train_pool)
            
            print(f"{var:<15} | {split.upper():<5} | {std['pairwise_acc']:.1%}    | {std['top1_acc']:.1%}    | {gen:.1%}")
            
            res = {"subject": args.subject, "variant": var, "split": split}
            res.update(std)
            res["gen_acc"] = gen
            results.append(res)
            
    if results:
        df = pd.DataFrame(results)
        header = not os.path.exists(RESULTS_FILE)
        df.to_csv(RESULTS_FILE, mode='a', header=header, index=False)
        print(f"\n✅ Salvato in {RESULTS_FILE}")

if __name__ == "__main__":
    main()