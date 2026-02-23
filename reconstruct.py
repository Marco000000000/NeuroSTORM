import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image, ImageDraw
import cv2
import glob
from tqdm import tqdm
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
import argparse

# === CONFIGURAZIONE ===
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FMRI_FEAT_DIR = "/home/mfinocchiaro/miccai2026/NeuroSTORM/hbn_data/neurostorm_features_split/test"
MODELS_BASE_DIR = "trained_aligners_hbn"
OUTPUT_DIR = "reconstruction_results_hbn"

# Video originale SOLO per confronto visivo (GT)
VIDEO_PATH = "/home/mfinocchiaro/miccai2026/The Present.mp4"
OFFSET_CALIBRAZIONE = 3.2 
WINDOW_SIZE = 0.8         

# Modelli Generativi
SDXL_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
CNET_ID = "diffusers/controlnet-depth-sdxl-1.0"

# --- MODELLO ALLINEATORE ---
class SimpleLinearAligner(nn.Module):
    def __init__(self, input_dim=2304, output_dim=1280):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    def forward(self, x): return self.linear(x)

def load_subject_aligners(subject_id):
    models = []
    subj_model_dir = os.path.join(MODELS_BASE_DIR, subject_id)
    print(f"📦 Caricamento modelli per {subject_id}...")
    
    if not os.path.exists(subj_model_dir):
        print(f"❌ Errore: Cartella modelli non trovata: {subj_model_dir}")
        return None

    for k in range(20):
        path = os.path.join(subj_model_dir, f"aligner_lag_{k:02d}.pth")
        if not os.path.exists(path):
            print(f"❌ Manca modello lag {k} in {path}")
            return None
        
        m = SimpleLinearAligner().to(DEVICE)
        m.load_state_dict(torch.load(path, map_location=DEVICE))
        m.eval()
        models.append(m)
    return models

def get_original_frame(cap, target_sec):
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(target_sec * fps))
    ret, frame = cap.read()
    if ret:
        return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).resize((1024, 1024))
    return Image.new('RGB', (1024, 1024))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=str, required=True, help="ID Soggetto (es. sub-NDARAA948VFH)")
    args = parser.parse_args()

    subject_out_dir = os.path.join(OUTPUT_DIR, args.subject)
    os.makedirs(subject_out_dir, exist_ok=True)

    # 1. Carica Pipeline Generativa
    print("🎨 Caricamento SDXL + IP-Adapter...")
    try:
        # Carichiamo ControlNet ma non lo useremo (Scale 0.0)
        controlnet = ControlNetModel.from_pretrained(CNET_ID, torch_dtype=torch.float16).to(DEVICE)
        pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            SDXL_MODEL, controlnet=controlnet, torch_dtype=torch.float16
        ).to(DEVICE)
        
        pipe.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter_sdxl.bin")
        pipe.set_ip_adapter_scale(0.7) # Forza del segnale fMRI
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.enable_model_cpu_offload()
    except Exception as e:
        print(f"❌ Errore caricamento HuggingFace: {e}")
        return

    # 2. Carica Allineatori Soggetto
    aligners = load_subject_aligners(args.subject)
    if not aligners: return

    # 3. Setup Video (Solo per Ground Truth)
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"❌ Impossibile aprire video: {VIDEO_PATH}")
        return
    
    # 4. Loop Ricostruzione
    subj_test_path = os.path.join(FMRI_FEAT_DIR, args.subject)
    test_files = sorted(glob.glob(os.path.join(subj_test_path, "window_*.pt")))
    
    if not test_files:
        print(f"❌ Nessun file trovato in {subj_test_path}")
        return

    print(f"🧪 Trovati {len(test_files)} file di test. Elaborazione primi 20...")
    
    # Immagine Nera Dummy per ControlNet (poiché scale=0.0)
    dummy_image = Image.new("RGB", (1024, 1024), (0, 0, 0))

    for fpath in tqdm(test_files[:20], desc="Generazione"):
        # start_idx della finestra corrente
        try:
            w_start = int(os.path.basename(fpath).split('_')[1].split('.')[0])
        except: continue
        
        # Frame centrale
        target_t = w_start + 10 
        
        preds_for_frame = []
        
        # ENSEMBLE: Sliding Window
        for lag in range(20):
            needed_start = target_t - lag
            w_file = os.path.join(subj_test_path, f"window_{needed_start:04d}.pt")
            
            if os.path.exists(w_file):
                try:
                    # Carica e estrai feature
                    feat = torch.load(w_file, map_location=DEVICE).float()
                    feat_flat = feat.view(-1, 20)
                    x_in = feat_flat[:, lag]
                    
                    # Predici
                    with torch.no_grad():
                        emb = aligners[lag](x_in.unsqueeze(0)) # Output: [1, 1280]
                        # Aggiungiamo dimensione Token: [1, 1, 1280]
                        emb = emb.unsqueeze(1) 
                        preds_for_frame.append(emb)
                except: pass
        
        if not preds_for_frame: continue
        
        # --- PREPARAZIONE EMBEDDING IP-ADAPTER ---
        # 1. Media degli embedding (Ensemble)
        # Stack su dim 0 -> [N, 1, 1, 1280] -> Mean -> [1, 1, 1280]
        avg_emb = torch.mean(torch.stack(preds_for_frame), dim=0)
        
        # 2. Creazione Embedding Negativo (Unconditional)
        # Deve essere zeri della stessa shape
        neg_emb = torch.zeros_like(avg_emb)
        
        # 3. Concatenazione [Negative, Positive] -> [2, 1, 1280]
        # Questo risolve l'errore "chunk(2)"
        final_emb = torch.cat([neg_emb, avg_emb], dim=0)
        
        # Recupera Ground Truth per confronto
        sec = OFFSET_CALIBRAZIONE + (target_t * WINDOW_SIZE)
        gt_image = get_original_frame(cap, sec)
        
        # Generazione Pura da fMRI
        with torch.no_grad():
            gen_image = pipe(
                prompt="", 
                image=dummy_image,  # Immagine nera
                ip_adapter_image_embeds=[final_emb], # Lista con tensore [2, 1, 1280]
                controlnet_conditioning_scale=0.0,   # DISATTIVATO COMPLETAMENTE
                num_inference_steps=30, 
                guidance_scale=5.0
            ).images[0]
            
        # Salvataggio Side-by-Side
        res = Image.new('RGB', (2048, 1024))
        res.paste(gt_image, (0, 0))
        res.paste(gen_image, (1024, 0))
        draw = ImageDraw.Draw(res)
        draw.text((20, 20), f"GT Frame {target_t} ({sec:.2f}s)", fill="white")
        draw.text((1044, 20), f"fMRI Reconstruct (No Depth)", fill="white")
        
        save_name = os.path.join(subject_out_dir, f"rec_frame_{target_t:04d}.png")
        res.save(save_name)

    cap.release()
    print(f"✅ Completato. Risultati in {subject_out_dir}")

if __name__ == "__main__":
    main()