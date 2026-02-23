import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image, ImageDraw
import cv2
import glob
from tqdm import tqdm
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel
import argparse

# === CONFIGURAZIONE ===
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FMRI_FEAT_DIR = "/home/mfinocchiaro/miccai2026/NeuroSTORM/hbn_data/neurostorm_features_split"
MODELS_BASE_DIR = "trained_aligners_contrastive"
OUTPUT_DIR = "reconstruction_sequence_val_test"

VIDEO_PATH = "/home/mfinocchiaro/miccai2026/The Present.mp4"
OFFSET_CALIBRAZIONE = 3.2 
TR_DURATION = 0.8
HRF_SHIFT = 14 

# SDXL Config
SDXL_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
CNET_ID = "diffusers/controlnet-depth-sdxl-1.0"

class LinearAligner(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
    def forward(self, x): return self.linear(x)

def load_clip_models(subject_id):
    models = {}
    variants = ["clip_single", "clip_avg", "clip_dominant"]
    print(f"📦 Caricamento modelli CLIP per {subject_id}...")
    for var in variants:
        path = os.path.join(MODELS_BASE_DIR, subject_id, var, "best_model.pth")
        if os.path.exists(path):
            m = LinearAligner(2304, 1280).to(DEVICE)
            m.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
            m.eval()
            models[var] = m
    return models

def get_original_frame(cap, target_sec):
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(target_sec * fps))
    ret, frame = cap.read()
    if ret:
        return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).resize((512, 512))
    return Image.new('RGB', (512, 512))

def generate_image(pipe, embedding, dummy_img):
    # Normalizzazione necessaria per modelli contrastive
    emb = torch.nn.functional.normalize(embedding, p=2, dim=1)
    emb = emb.unsqueeze(1) 
    neg = torch.zeros_like(emb)
    final = torch.cat([neg, emb], dim=0) 
    
    with torch.no_grad():
        img = pipe(
            prompt="", 
            image=dummy_img, 
            ip_adapter_image_embeds=[final],
            controlnet_conditioning_scale=0.0, # Pure fMRI-to-Visual
            num_inference_steps=25, 
            guidance_scale=7.0
        ).images[0]
    return img.resize((512, 512))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=str, required=True)
    args = parser.parse_args()

    out_path = os.path.join(OUTPUT_DIR, args.subject)
    os.makedirs(out_path, exist_ok=True)

    # 1. Pipeline SDXL
    print("🎨 Inizializzazione SDXL + IP-Adapter...")
    controlnet = ControlNetModel.from_pretrained(CNET_ID, torch_dtype=torch.float16).to(DEVICE)
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        SDXL_MODEL, controlnet=controlnet, torch_dtype=torch.float16
    ).to(DEVICE)
    pipe.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter_sdxl.bin")
    pipe.set_ip_adapter_scale(0.8)
    pipe.enable_model_cpu_offload()

    # 2. Modelli Allineatori
    models = load_clip_models(args.subject)
    
    # 3. Raccolta File Val + Test (Continuità temporale)
    files = []
    for split in ["val", "test"]:
        path = os.path.join(FMRI_FEAT_DIR, split, args.subject)
        files.extend(glob.glob(os.path.join(path, "window_*.pt")))
    
    # Ordinamento per indice temporale
    files = sorted(files, key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
    print(f"🎬 Sequenza pronta: {len(files)} frame da ricostruire.")

    cap = cv2.VideoCapture(VIDEO_PATH)
    dummy_image = Image.new("RGB", (1024, 1024), (0, 0, 0))

    for fpath in tqdm(files, desc="Generazione"):
        w_start = int(os.path.basename(fpath).split('_')[1].split('.')[0])
        target_idx = (w_start + 10) - HRF_SHIFT
        if target_idx < 0: continue
        
        # Carica fMRI e predici
        feat = torch.load(fpath, map_location=DEVICE, weights_only=True).float().view(-1, 20)
        x_in = feat[:, 10].unsqueeze(0)
        
        preds_imgs = {}
        for name, model in models.items():
            with torch.no_grad():
                emb = model(x_in)
            preds_imgs[name] = generate_image(pipe, emb, dummy_image)

        # Ground Truth
        sec = OFFSET_CALIBRAZIONE + (target_idx * TR_DURATION)
        gt_img = get_original_frame(cap, sec)
        
        # Grid layout
        W, H = 512, 512
        grid = Image.new('RGB', (W * 4, H + 60), (15, 15, 15))
        grid.paste(gt_img, (0, 60))
        if "clip_single" in preds_imgs: grid.paste(preds_imgs["clip_single"], (W, 60))
        if "clip_avg" in preds_imgs: grid.paste(preds_imgs["clip_avg"], (W*2, 60))
        if "clip_dominant" in preds_imgs: grid.paste(preds_imgs["clip_dominant"], (W*3, 60))
        
        draw = ImageDraw.Draw(grid)
        draw.text((20, 20), f"GT Time: {sec:.1f}s", fill="white")
        draw.text((W+20, 20), "CLIP Single", fill="cyan")
        draw.text((W*2+20, 20), "CLIP Avg", fill="lime")
        draw.text((W*3+20, 20), "CLIP Dominant", fill="orange")
        
        grid.save(os.path.join(out_path, f"seq_{w_start:04d}.png"))

    cap.release()
    print(f"✨ Ricostruzione completata in {out_path}")

if __name__ == "__main__":
    main()