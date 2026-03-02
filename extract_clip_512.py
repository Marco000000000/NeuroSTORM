import torch
import cv2
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPVisionModelWithProjection

# === CONFIGURAZIONE ===
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VIDEO_PATH = "/home/mfinocchiaro/miccai2026/NeuroSTORM/descme_10min_frame_samecodec.mp4"
DATA_DIR = "dataset_fmri_features"

# Sincronizzazione Temporale
START_SEC = 0        # Offset iniziale (ritardo fMRI)
TOTAL_TR = 750       # ESATTAMENTE 750 volumi
TR_DURATION = 0.8    # Durata di un volume fMRI (HBN standard)
FPS_SAMPLING = 8     # Frame campionati per ogni TR (finestra 0.8s)
SCENE_CUT_THRESHOLD = 0.90 # Soglia similarità coseno

# ID del modello CLIP base (produce embedding da 512 dimensioni)
CLIP_ID = "openai/clip-vit-base-patch32"

os.makedirs(DATA_DIR, exist_ok=True)

# === 1. CARICAMENTO MODELLI ===
print("🚀 Caricamento Modello CLIP Base (512 features)...")

# CLIP (Standard via Hugging Face)
clip_processor = CLIPProcessor.from_pretrained(CLIP_ID)
clip_model = CLIPVisionModelWithProjection.from_pretrained(CLIP_ID, use_safetensors=True).to(DEVICE).eval()

# === 2. FUNZIONI DI ESTRAZIONE ===

def get_clip_emb(pil_image):
    """Estrae embedding CLIP [1, 512]"""
    inputs = clip_processor(images=pil_image, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = clip_model(**inputs)
        # image_embeds contiene la proiezione finale dello spazio visivo (512 per il base)
        emb = outputs.image_embeds 
    return emb.cpu() # [1, 512]

def get_dominant_index(embeddings_list):
    """
    Trova l'INDICE del frame dominante nella lista di 8 frame.
    Usa CLIP per rilevare i cambi scena (più robusto semanticamente).
    """
    if len(embeddings_list) == 0: return 0
    if len(embeddings_list) == 1: return 0

    # Stack: [N, 512]
    stack = torch.cat(embeddings_list, dim=0).to(DEVICE) 
    if len(stack.shape) == 3: stack = stack.squeeze(1)

    # Similarità tra frame consecutivi
    sims = F.cosine_similarity(stack[:-1], stack[1:])
    
    # Trova i tagli
    cut_indices = torch.where(sims < SCENE_CUT_THRESHOLD)[0] + 1
    cut_indices = cut_indices.tolist()
    boundaries = [0] + cut_indices + [len(embeddings_list)]
    
    # Trova segmento più lungo
    max_len = -1
    best_segment_mid_idx = 0

    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i+1]
        length = end - start
        
        if length > max_len:
            max_len = length
            # Indice centrale relativo a questo segmento
            best_segment_mid_idx = start + (length // 2)
            
    return best_segment_mid_idx

# === 3. LOOP DI ESTRAZIONE ===
cap = cv2.VideoCapture(VIDEO_PATH)
fps_vid = cap.get(cv2.CAP_PROP_FPS)

dataset = {
    "clip_single": [],   # Frame centrale assoluto
    "clip_avg": [],      # Media temporale
    "clip_dominant": []  # Scena dominante
}

print(f"🎥 Video FPS: {fps_vid}")
print(f"⏱️  Estrazione: {START_SEC}s -> {START_SEC + TOTAL_TR*TR_DURATION}s")
print(f"📊 Target: {TOTAL_TR} volumi (TR={TR_DURATION}s)")

for tr in tqdm(range(TOTAL_TR), desc="Processing TRs"):
    # Calcolo tempo esatto: start + (indice_volume * durata_volume)
    t_start = START_SEC + (tr * TR_DURATION)
    
    tr_clip_embeds = []
    
    # Campionamento finestra (es. 0.8s)
    for f in range(FPS_SAMPLING):
        # Distribuiamo gli 8 frame uniformemente nei 0.8s
        offset = f * (TR_DURATION / FPS_SAMPLING)
        t_frame = t_start + offset
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t_frame * fps_vid))
        ret, frame = cap.read()
        
        if not ret:
            # Padding con l'ultimo frame valido se il video finisce
            if len(tr_clip_embeds) > 0:
                tr_clip_embeds.append(tr_clip_embeds[-1])
            continue
        
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        c_emb = get_clip_emb(img_pil)
        tr_clip_embeds.append(c_emb)

    if not tr_clip_embeds:
        print(f"⚠️ Errore critico al TR {tr}: nessun frame.")
        break

    # 1. SINGLE (Frame centrale della finestra, indice 4 su 8)
    mid_idx = len(tr_clip_embeds) // 2
    dataset["clip_single"].append(tr_clip_embeds[mid_idx])
    
    # 2. AVG (Media su tutti gli 8 frame)
    dataset["clip_avg"].append(torch.mean(torch.stack(tr_clip_embeds), dim=0))
    
    # 3. DOMINANT (Identifica l'indice migliore e usalo per entrambi)
    dom_idx = get_dominant_index(tr_clip_embeds)
    dataset["clip_dominant"].append(tr_clip_embeds[dom_idx])

cap.release()

# === 4. SALVATAGGIO ===
print("\n💾 Conversione e Salvataggio...")
final_path = os.path.join(DATA_DIR, "features_despicable_clip_512.pt")

for key in dataset:
    if len(dataset[key]) > 0:
        # Stack e pulizia dimensioni
        t = torch.cat(dataset[key], dim=0) # [750, 512]
        if len(t.shape) == 3 and t.shape[1] == 1:
            t = t.squeeze(1) # Rimuove la dimensione extra
        dataset[key] = t
        print(f"  • {key}: {t.shape}")

# Controllo finale dimensione
assert dataset["clip_dominant"].shape[0] == 750, f"Errore: Generati {dataset['clip_dominant'].shape[0]} frame invece di 750!"
assert dataset["clip_dominant"].shape[1] == 512, f"Errore: Dimensione feature {dataset['clip_dominant'].shape[1]} invece di 512!"

torch.save(dataset, final_path)
print(f"\n✅ Dataset salvato: {final_path}")
print("   Contiene chiavi: clip_single, clip_avg, clip_dominant")