import torch
import cv2
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import models, transforms
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel
import torch.nn.functional as F

# === CONFIGURAZIONE ===
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VIDEO_PATH = "/home/mfinocchiaro/miccai2026/NeuroSTORM/descme_10min_frame_samecodec.mp4"
DATA_DIR = "dataset_fmri_features"

# Sincronizzazione Temporale
START_SEC = 0        # Offset iniziale (ritardo fMRI)
TOTAL_TR = 750         # ESATTAMENTE 750 volumi
TR_DURATION = 0.8      # Durata di un volume fMRI (HBN standard)
FPS_SAMPLING = 8       # Frame campionati per ogni TR (finestra 0.8s)
SCENE_CUT_THRESHOLD = 0.90 # Soglia similarità coseno

CNET_ID = "diffusers/controlnet-depth-sdxl-1.0"

os.makedirs(DATA_DIR, exist_ok=True)

# === 1. CARICAMENTO MODELLI ===
print("🚀 Caricamento Modelli...")

# A. CLIP (via SDXL Pipeline)
controlnet = ControlNetModel.from_pretrained(CNET_ID, torch_dtype=torch.float16).to(DEVICE)
pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", controlnet=controlnet, torch_dtype=torch.float16
).to(DEVICE)
pipe.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter_sdxl.bin")

# B. AlexNet
alexnet = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1).to(DEVICE).eval()
alexnet_features = alexnet.classifier[:6] # FC7 layer (4096)

alex_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# === 2. FUNZIONI DI ESTRAZIONE ===

def get_clip_emb(pil_image):
    """Estrae embedding CLIP [1, 1280]"""
    with torch.no_grad():
        emb = pipe.prepare_ip_adapter_image_embeds(
            ip_adapter_image=pil_image, 
            ip_adapter_image_embeds=None, 
            device=DEVICE, 
            num_images_per_prompt=1, 
            do_classifier_free_guidance=False
        )
    return emb[0] # [1, 1, 1280]

def get_alex_emb(pil_image):
    """Estrae feature AlexNet FC7 [1, 4096]"""
    img_t = alex_transform(pil_image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        feats = alexnet.features(img_t)
        feats = alexnet.avgpool(feats)
        feats = torch.flatten(feats, 1)
        out = alexnet_features(feats)
    return out.cpu() 

def get_dominant_index(embeddings_list):
    """
    Trova l'INDICE del frame dominante nella lista di 8 frame.
    Usa CLIP per rilevare i cambi scena (più robusto semanticamente).
    """
    if len(embeddings_list) == 0: return 0
    if len(embeddings_list) == 1: return 0

    # Stack: [N, 1280]
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
    "clip_single": [],   "alex_single": [],   # Frame centrale assoluto
    "clip_avg": [],      "alex_avg": [],      # Media temporale
    "clip_dominant": [], "alex_dominant": []  # Scena dominante
}

print(f"🎥 Video FPS: {fps_vid}")
print(f"⏱️  Estrazione: {START_SEC}s -> {START_SEC + TOTAL_TR*TR_DURATION}s")
print(f"📊 Target: {TOTAL_TR} volumi (TR={TR_DURATION}s)")

for tr in tqdm(range(TOTAL_TR), desc="Processing TRs"):
    # Calcolo tempo esatto: start + (indice_volume * durata_volume)
    t_start = START_SEC + (tr * TR_DURATION)
    
    tr_clip_embeds = []
    tr_alex_embeds = []
    
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
                tr_alex_embeds.append(tr_alex_embeds[-1])
            continue
        
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        c_emb = get_clip_emb(img_pil)
        a_emb = get_alex_emb(img_pil)
        
        tr_clip_embeds.append(c_emb)
        tr_alex_embeds.append(a_emb)

    if not tr_clip_embeds:
        print(f"⚠️ Errore critico al TR {tr}: nessun frame.")
        break

    # 1. SINGLE (Frame centrale della finestra, indice 4 su 8)
    mid_idx = len(tr_clip_embeds) // 2
    dataset["clip_single"].append(tr_clip_embeds[mid_idx])
    dataset["alex_single"].append(tr_alex_embeds[mid_idx])
    
    # 2. AVG (Media su tutti gli 8 frame)
    dataset["clip_avg"].append(torch.mean(torch.stack(tr_clip_embeds), dim=0))
    dataset["alex_avg"].append(torch.mean(torch.stack(tr_alex_embeds), dim=0))
    
    # 3. DOMINANT (Identifica l'indice migliore e usalo per entrambi)
    dom_idx = get_dominant_index(tr_clip_embeds)
    dataset["clip_dominant"].append(tr_clip_embeds[dom_idx])
    dataset["alex_dominant"].append(tr_alex_embeds[dom_idx])

cap.release()

# === 4. SALVATAGGIO ===
print("\n💾 Conversione e Salvataggio...")
final_path = os.path.join(DATA_DIR, "features_despicable.pt")

for key in dataset:
    if len(dataset[key]) > 0:
        # Stack e pulizia dimensioni
        t = torch.cat(dataset[key], dim=0) # [750, 1, 1280]
        if len(t.shape) == 3 and t.shape[1] == 1:
            t = t.squeeze(1) # [750, 1280]
        dataset[key] = t
        print(f"  • {key}: {t.shape}")

# Controllo finale dimensione
assert dataset["clip_dominant"].shape[0] == 750, f"Errore: Generati {dataset['clip_dominant'].shape[0]} frame invece di 750!"

torch.save(dataset, final_path)
print(f"\n✅ Dataset salvato: {final_path}")
print("   Contiene chiavi: clip_single, clip_avg, clip_dominant, alex_single, alex_avg, alex_dominant")