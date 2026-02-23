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
OUTPUT_DIR = "reconstruction_continuous_hbn"

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
            print(f"❌ Manca modello lag {k}")
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=str, required=True, help="ID Soggetto")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test", "train"], help="Quale split usare")
    args = parser.parse_args()

    # Cartella specifica per questo esperimento
    save_dir = os.path.join(OUTPUT_DIR, f"{args.subject}_{args.split}")
    os.makedirs(save_dir, exist_ok=True)

    # 1. Pipeline Generativa
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

    # 2. Modelli Allineamento
    aligners = load_subject_aligners(args.subject)
    if not aligners: return

    # 3. Video Setup
    cap = cv2.VideoCapture(VIDEO_PATH)

    # 4. Caricamento e Indicizzazione Finestre
    subj_data_path = os.path.join(FMRI_BASE_DIR, args.split, args.subject)
    all_files = sorted(glob.glob(os.path.join(subj_data_path, "window_*.pt")))
    
    if not all_files:
        print(f"❌ Nessun file trovato in {subj_data_path}")
        return

    # Mappa: Start_Frame -> File Path
    # Serve per accesso rapido randomico
    window_map = {}
    min_start = float('inf')
    max_start = float('-inf')

    print(f"📂 Indicizzazione {len(all_files)} finestre fMRI ({args.split})...")
    for f in all_files:
        try:
            start_idx = int(os.path.basename(f).split('_')[1].split('.')[0])
            window_map[start_idx] = f
            if start_idx < min_start: min_start = start_idx
            if start_idx > max_start: max_start = start_idx
        except: continue

    # Definiamo il range temporale COMPLETO da ricostruire
    # Dal primo frame della prima finestra, all'ultimo frame dell'ultima finestra
    reconstruct_start = min_start
    reconstruct_end = max_start + 20 
    
    print(f"⏱️  Ricostruzione Continua: Frame {reconstruct_start} -> {reconstruct_end}")
    print(f"    (Totale {reconstruct_end - reconstruct_start} frame video)")

    dummy_image = Image.new("RGB", (1024, 1024), (0, 0, 0))

    # LOOP CONTINUO SUI FRAME
    for target_t in tqdm(range(reconstruct_start, reconstruct_end), desc="Reconstruction"):
        
        preds_for_frame = []
        
        # LOGICA ENSEMBLE DINAMICO
        # Per ogni frame target T, controlliamo tutte le 20 possibili finestre che potrebbero contenerlo.
        # Window_Start = T - Lag
        
        for lag in range(20):
            needed_start = target_t - lag
            
            # Se esiste una finestra che inizia a 'needed_start', allora quella finestra
            # contiene il nostro frame 'target_t' esattamente alla posizione 'lag'.
            if needed_start in window_map:
                w_file = window_map[needed_start]
                try:
                    # Carica feature [288, 2, 2, 2, 20]
                    # Ottimizzazione: Non caricare tutto il tensore se non serve, ma qui serve slice.
                    # Caricamento su CPU poi move
                    feat = torch.load(w_file, map_location=DEVICE).float()
                    feat_flat = feat.view(-1, 20)
                    
                    x_in = feat_flat[:, lag] # Feature specifica
                    
                    with torch.no_grad():
                        # Predizione [1, 1280]
                        emb = aligners[lag](x_in.unsqueeze(0))
                        # Aggiungi dim token: [1, 1, 1280]
                        emb = emb.unsqueeze(1)
                        preds_for_frame.append(emb)
                except: pass

        # Se non abbiamo nessuna vista (es. buco nei dati), saltiamo
        if not preds_for_frame:
            continue
            
        # MEDIA EMBEDDING (Ensemble variabile: da 1 a 20 viste)
        avg_emb = torch.mean(torch.stack(preds_for_frame), dim=0)
        
        # Preparazione Condizionamento
        neg_emb = torch.zeros_like(avg_emb)
        final_emb = torch.cat([neg_emb, avg_emb], dim=0) # [2, 1, 1280]
        
        # GT e Depth
        sec = OFFSET_CALIBRAZIONE + (target_t * WINDOW_SIZE)
        gt_image = get_original_frame(cap, sec)
        depth_image = get_depth_map(depth_pipe, gt_image)
        
        # Generazione
        with torch.no_grad():
            gen_image = pipe(
                prompt="", 
                image=dummy_image, 
                ip_adapter_image_embeds=[final_emb],
                controlnet_conditioning_scale=0.0, # NO DEPTH GUIDANCE
                num_inference_steps=30, 
                guidance_scale=5.0
            ).images[0]
            
        # Salvataggio
        res = Image.new('RGB', (2048, 1024))
        res.paste(gt_image, (0, 0))
        res.paste(gen_image, (1024, 0))
        draw = ImageDraw.Draw(res)
        
        # Info a schermo
        draw.text((20, 20), f"GT Frame {target_t} ({sec:.2f}s)", fill="white")
        draw.text((1044, 20), f"Rec (Ensemble of {len(preds_for_frame)} lags)", fill="white")
        
        # Barra di progresso dell'ensemble (visiva)
        bar_width = int((len(preds_for_frame) / 20) * 200)
        draw.rectangle([(1044, 60), (1044 + 200, 70)], outline="white", width=1)
        draw.rectangle([(1044, 60), (1044 + bar_width, 70)], fill="lime")

        save_name = os.path.join(save_dir, f"rec_{target_t:05d}.png")
        res.save(save_name)

    cap.release()
    print(f"✅ Finito! Guarda i risultati in: {save_dir}")

if __name__ == "__main__":
    main()