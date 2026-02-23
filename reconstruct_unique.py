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
MODELS_BASE_DIR = "trained_aligners_unified"
OUTPUT_DIR = "reconstruction_unified_hbn"

VIDEO_PATH = "/home/mfinocchiaro/miccai2026/The Present.mp4"
OFFSET_CALIBRAZIONE = 3.2 
WINDOW_SIZE = 0.8         

# Generazione
SDXL_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
CNET_ID = "diffusers/controlnet-depth-sdxl-1.0"

# --- MODELLO ---
class UnifiedLinearAligner(nn.Module):
    def __init__(self, input_dim=2304, output_dim=1280):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    def forward(self, x): return self.linear(x)

def load_unified_model(subject_id):
    path = os.path.join(MODELS_BASE_DIR, subject_id, "unified_aligner.pth")
    if not os.path.exists(path):
        print(f"❌ Modello non trovato: {path}")
        return None
    
    print(f"📦 Caricamento modello unificato per {subject_id}...")
    model = UnifiedLinearAligner().to(DEVICE)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.eval()
    return model

def get_original_frame(cap, target_sec):
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(target_sec * fps))
    ret, frame = cap.read()
    if ret:
        return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).resize((1024, 1024))
    return Image.new('RGB', (1024, 1024))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=str, required=True, help="ID Soggetto")
    args = parser.parse_args()

    subject_out_dir = os.path.join(OUTPUT_DIR, args.subject)
    os.makedirs(subject_out_dir, exist_ok=True)

    # 1. Pipeline
    print("🎨 Caricamento Pipeline...")
    controlnet = ControlNetModel.from_pretrained(CNET_ID, torch_dtype=torch.float16).to(DEVICE)
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        SDXL_MODEL, controlnet=controlnet, torch_dtype=torch.float16
    ).to(DEVICE)
    pipe.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter_sdxl.bin")
    pipe.set_ip_adapter_scale(0.7)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()

    # 2. Modello Unificato
    model = load_unified_model(args.subject)
    if not model: return

    cap = cv2.VideoCapture(VIDEO_PATH)
    subj_test_path = os.path.join(FMRI_FEAT_DIR, args.subject)
    test_files = sorted(glob.glob(os.path.join(subj_test_path, "window_*.pt")))
    
    if not test_files: return
    print(f"🧪 Trovati {len(test_files)} finestre. Elaborazione primi 20...")
    
    dummy_image = Image.new("RGB", (1024, 1024), (0, 0, 0))

    for fpath in tqdm(test_files[:20], desc="Generazione"):
        try:
            w_start = int(os.path.basename(fpath).split('_')[1].split('.')[0])
        except: continue
        
        target_t = w_start + 10 
        preds_for_frame = []
        
        # ENSEMBLE: Cerchiamo il frame target in tutte le finestre
        for lag in range(20):
            needed_start = target_t - lag
            w_file = os.path.join(subj_test_path, f"window_{needed_start:04d}.pt")
            
            if os.path.exists(w_file):
                try:
                    feat = torch.load(w_file, map_location=DEVICE).float()
                    feat_flat = feat.view(-1, 20)
                    
                    # Estrarre la feature corrispondente al frame target
                    x_in = feat_flat[:, lag]
                    
                    with torch.no_grad():
                        # Passiamo TUTTO allo STESSO modello unificato
                        emb = model(x_in.unsqueeze(0)) 
                        emb = emb.unsqueeze(1) # [1, 1, 1280]
                        preds_for_frame.append(emb)
                except: pass
        
        if not preds_for_frame: continue
        
        # Average Ensemble
        avg_emb = torch.mean(torch.stack(preds_for_frame), dim=0)
        
        # Prepare for IP-Adapter (Neg + Pos)
        neg_emb = torch.zeros_like(avg_emb)
        final_emb = torch.cat([neg_emb, avg_emb], dim=0)
        
        # GT
        sec = OFFSET_CALIBRAZIONE + (target_t * WINDOW_SIZE)
        gt_image = get_original_frame(cap, sec)
        
        # Generation
        with torch.no_grad():
            gen_image = pipe(
                prompt="", 
                image=dummy_image, 
                ip_adapter_image_embeds=[final_emb],
                controlnet_conditioning_scale=0.0,
                num_inference_steps=30, 
                guidance_scale=5.0
            ).images[0]
            
        # Save
        res = Image.new('RGB', (2048, 1024))
        res.paste(gt_image, (0, 0))
        res.paste(gen_image, (1024, 0))
        draw = ImageDraw.Draw(res)
        draw.text((20, 20), f"GT Frame {target_t} ({sec:.2f}s)", fill="white")
        draw.text((1044, 20), f"Unified Rec (Avg {len(preds_for_frame)} views)", fill="white")
        
        save_name = os.path.join(subject_out_dir, f"unified_rec_{target_t:04d}.png")
        res.save(save_name)

    cap.release()
    print("✅ Completato.")

if __name__ == "__main__":
    main()