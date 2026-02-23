import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
import glob
from tqdm import tqdm
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from transformers import AutoImageProcessor, DepthAnythingForDepthEstimation
import argparse

# === CONFIGURAZIONE ===
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FMRI_BASE_DIR = "/home/mfinocchiaro/miccai2026/NeuroSTORM/hbn_data/neurostorm_features_split"
MODELS_BASE_DIR = "trained_aligners_hbn"
OUTPUT_DIR = "reconstruction_mirrored_hbn"

VIDEO_PATH = "/home/mfinocchiaro/miccai2026/The Present.mp4"
OFFSET_CALIBRAZIONE = 3.2 
WINDOW_SIZE = 0.8         

# Modelli Generativi
SDXL_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
CNET_ID = "diffusers/controlnet-depth-sdxl-1.0"
DEPTH_ID = "LiheYoung/depth-anything-small-hf"

# --- MODELLO ---
class SimpleLinearAligner(nn.Module):
    def __init__(self, input_dim=2304, output_dim=1280):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    def forward(self, x): return self.linear(x)

def load_subject_aligners(subject_id):
    models = []
    subj_model_dir = os.path.join(MODELS_BASE_DIR, subject_id)
    print(f"📦 Caricamento 20 modelli per {subject_id}...")
    
    for k in range(20):
        path = os.path.join(subj_model_dir, f"aligner_lag_{k:02d}.pth")
        if not os.path.exists(path):
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

def get_depth_map(depth_pipe, image):
    inputs = depth_pipe['proc'](images=image, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = depth_pipe['model'](**inputs)
    prediction = torch.nn.functional.interpolate(
        outputs.predicted_depth.unsqueeze(1), size=(1024, 1024), mode="bicubic", align_corners=False
    )
    depth_map = (prediction - prediction.min()) / (prediction.max() - prediction.min())
    return Image.fromarray((depth_map.squeeze().cpu().numpy() * 255).astype(np.uint8)).convert("RGB")

# --- LOGICA DI SPECCHIO (MIRRORING) ---
def get_safe_window_file(needed_idx, window_map, min_idx, max_idx):
    """
    Se l'indice richiesto esiste, lo ritorna.
    Se non esiste (bordo), applica il mirroring (riflessione).
    """
    is_mirrored = False
    idx_to_load = needed_idx

    # Caso 1: Siamo prima dell'inizio (es. serve -2, min è 0)
    # Mirroring: -2 diventa 2
    if needed_idx < min_idx:
        diff = min_idx - needed_idx
        idx_to_load = min_idx + diff
        is_mirrored = True
    
    # Caso 2: Siamo dopo la fine (es. serve 102, max è 100)
    # Mirroring: 102 diventa 98
    elif needed_idx > max_idx:
        diff = needed_idx - max_idx
        idx_to_load = max_idx - diff
        is_mirrored = True
    
    # Check finale (se il mirroring ci porta comunque fuori range, clampiamo)
    if idx_to_load < min_idx: idx_to_load = min_idx
    if idx_to_load > max_idx: idx_to_load = max_idx
    
    # Recupera il file path
    if idx_to_load in window_map:
        return window_map[idx_to_load], is_mirrored
    else:
        return None, False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=str, required=True, help="ID Soggetto")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test", "train"])
    args = parser.parse_args()

    save_dir = os.path.join(OUTPUT_DIR, f"{args.subject}_{args.split}")
    os.makedirs(save_dir, exist_ok=True)

    # 1. Pipeline
    print("🎨 Caricamento Pipeline...")
    try:
        depth_model = DepthAnythingForDepthEstimation.from_pretrained(DEPTH_ID).to(DEVICE)
        depth_proc = AutoImageProcessor.from_pretrained(DEPTH_ID)
        depth_pipe = {'model': depth_model, 'proc': depth_proc}

        controlnet = ControlNetModel.from_pretrained(CNET_ID, torch_dtype=torch.float16).to(DEVICE)
        pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            SDXL_MODEL, controlnet=controlnet, torch_dtype=torch.float16
        ).to(DEVICE)
        pipe.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter_sdxl.bin")
        pipe.set_ip_adapter_scale(0.7)
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.enable_model_cpu_offload()
    except Exception as e:
        print(f"❌ Errore HuggingFace: {e}")
        return

    # 2. Modelli
    aligners = load_subject_aligners(args.subject)
    if not aligners: return

    # 3. Indicizzazione Finestre
    subj_data_path = os.path.join(FMRI_BASE_DIR, args.split, args.subject)
    all_files = sorted(glob.glob(os.path.join(subj_data_path, "window_*.pt")))
    
    if not all_files:
        print(f"❌ Nessun file trovato in {subj_data_path}")
        return

    window_map = {}
    min_start = float('inf')
    max_start = float('-inf')

    print(f"📂 Indicizzazione {len(all_files)} finestre...")
    for f in all_files:
        try:
            start_idx = int(os.path.basename(f).split('_')[1].split('.')[0])
            window_map[start_idx] = f
            if start_idx < min_start: min_start = start_idx
            if start_idx > max_start: max_start = start_idx
        except: continue

    # Ricostruiamo TUTTO il range coperto dalle finestre
    reconstruct_start = min_start
    reconstruct_end = max_start + 20 
    
    cap = cv2.VideoCapture(VIDEO_PATH)
    dummy_image = Image.new("RGB", (1024, 1024), (0, 0, 0))

    print(f"⏱️  Ricostruzione Mirrored: {reconstruct_start} -> {reconstruct_end}")

    for target_t in tqdm(range(reconstruct_start, reconstruct_end), desc="Generating"):
        
        preds_for_frame = []
        mirrored_count = 0
        
        # Loop Ensemble
        for lag in range(20):
            needed_start = target_t - lag
            
            # --- LOGICA SPECCHIO ---
            w_file, is_mirrored = get_safe_window_file(needed_start, window_map, min_start, max_start)
            
            if w_file is not None:
                try:
                    # Carica (Solo la feature necessaria per risparmiare tempo)
                    feat = torch.load(w_file, map_location=DEVICE).float()
                    feat_flat = feat.view(-1, 20)
                    x_in = feat_flat[:, lag] # Feature specifica
                    
                    with torch.no_grad():
                        emb = aligners[lag](x_in.unsqueeze(0)).unsqueeze(1)
                        preds_for_frame.append(emb)
                        if is_mirrored: mirrored_count += 1
                except: pass

        if not preds_for_frame: continue
            
        # Media (Ora abbiamo quasi sempre 20 elementi grazie allo specchio)
        avg_emb = torch.mean(torch.stack(preds_for_frame), dim=0)
        
        # Condizionamento
        final_emb = torch.cat([torch.zeros_like(avg_emb), avg_emb], dim=0)
        
        # GT
        sec = OFFSET_CALIBRAZIONE + (target_t * WINDOW_SIZE)
        gt_image = get_original_frame(cap, sec)
        depth_image = get_depth_map(depth_pipe, gt_image)
        
        # Generazione
        with torch.no_grad():
            gen_image = pipe(
                prompt="", 
                image=dummy_image, # Solo per inizializzare ControlNet
                ip_adapter_image_embeds=[final_emb],
                controlnet_conditioning_scale=0.0, # NO DEPTH GUIDANCE
                num_inference_steps=30, 
                guidance_scale=5.0
            ).images[0]
            
        # Plot
        res = Image.new('RGB', (2048, 1024))
        res.paste(gt_image, (0, 0))
        res.paste(gen_image, (1024, 0))
        draw = ImageDraw.Draw(res)
        
        # Testi
        draw.text((20, 20), f"GT Frame {target_t} ({sec:.2f}s)", fill="white")
        draw.text((1044, 20), f"Full Ensemble (20 Lags)", fill="white")
        
        # Barra Verde (Reale) vs Barra Rossa (Mirrored)
        total_preds = len(preds_for_frame) # Dovrebbe essere 20
        real_preds = total_preds - mirrored_count
        
        # Disegna background barra
        bar_x, bar_y = 1044, 60
        draw.rectangle([(bar_x, bar_y), (bar_x + 200, bar_y + 10)], outline="white")
        
        # Parte Verde (Dati Reali)
        width_real = int((real_preds / 20) * 200)
        draw.rectangle([(bar_x, bar_y), (bar_x + width_real, bar_y + 10)], fill="lime")
        
        # Parte Rossa (Dati Specchiati)
        width_mirror = int((mirrored_count / 20) * 200)
        draw.rectangle([(bar_x + width_real, bar_y), (bar_x + width_real + width_mirror, bar_y + 10)], fill="red")
        
        draw.text((bar_x + 210, bar_y), f"Real: {real_preds} | Mirrored: {mirrored_count}", fill="white")

        save_name = os.path.join(save_dir, f"rec_{target_t:05d}.png")
        res.save(save_name)

    cap.release()
    print(f"✅ Finito! Guarda la barra: Verde=Dati Reali, Rosso=Padding.")

if __name__ == "__main__":
    main()