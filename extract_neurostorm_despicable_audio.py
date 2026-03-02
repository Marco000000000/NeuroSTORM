import torch
import os
import numpy as np
from tqdm import tqdm
from transformers import ClapModel, ClapProcessor
import torch.nn.functional as F
import librosa

# === CONFIGURAZIONE ===
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VIDEO_PATH = "/home/mfinocchiaro/miccai2026/NeuroSTORM/descme_10min_frame_samecodec.mp4"
DATA_DIR = "dataset_fmri_features"

# Sincronizzazione Temporale
START_SEC = 0          # Offset iniziale (ritardo fMRI)
TOTAL_TR = 750         # ESATTAMENTE 750 volumi
TR_DURATION = 0.8      # Durata di un volume fMRI (HBN standard)
FPS_SAMPLING = 8       # Chunk campionati per ogni TR (finestra 0.8s)
SCENE_CUT_THRESHOLD = 0.90 # Soglia similarità coseno per i cambi di suono

CLAP_ID = "laion/clap-htsat-unfused"
AUDIO_SR = 48000       # Sampling rate nativo per CLAP

os.makedirs(DATA_DIR, exist_ok=True)

# === 1. CARICAMENTO MODELLO ED ESTRAZIONE AUDIO ===
print("🚀 Caricamento Modello CLAP e Traccia Audio...")

clap_processor = ClapProcessor.from_pretrained(CLAP_ID)
clap_model = ClapModel.from_pretrained(CLAP_ID, use_safetensors=True).to(DEVICE).eval()

print("🎵 Estrazione traccia audio dal file (potrebbe richiedere qualche secondo)...")
# librosa estrarrà solo l'audio dal file mp4
audio_waveform, _ = librosa.load(VIDEO_PATH, sr=AUDIO_SR, mono=True)
total_audio_samples = len(audio_waveform)

# === 2. FUNZIONI DI ESTRAZIONE ===

def get_clap_emb(audio_chunk):
    """Estrae embedding CLAP [1, 512] da un array audio numpy"""
    with torch.no_grad():
        inputs = clap_processor(audio=audio_chunk, sampling_rate=AUDIO_SR, return_tensors="pt").to(DEVICE)
        
        # WORKAROUND: Calcoliamo l'embedding esplicito aggirando il bug dell'oggetto BaseModelOutputWithPooling
        audio_outputs = clap_model.audio_model(**inputs)
        emb = clap_model.audio_projection(audio_outputs.pooler_output)
        
    return emb.cpu()

def get_dominant_index(embeddings_list):
    """
    Trova l'INDICE del chunk dominante nella lista (audio più stabile).
    """
    if len(embeddings_list) <= 1: return 0

    # Stack: [N, 512]
    stack = torch.cat(embeddings_list, dim=0).to(DEVICE) 
    if len(stack.shape) == 3: stack = stack.squeeze(1)

    # Similarità tra chunk consecutivi
    sims = F.cosine_similarity(stack[:-1], stack[1:])
    
    # Trova i "tagli" audio
    cut_indices = torch.where(sims < SCENE_CUT_THRESHOLD)[0] + 1
    boundaries = [0] + cut_indices.tolist() + [len(embeddings_list)]
    
    # Trova segmento più lungo
    max_len = -1
    best_segment_mid_idx = 0

    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i+1]
        length = end - start
        
        if length > max_len:
            max_len = length
            best_segment_mid_idx = start + (length // 2)
            
    return best_segment_mid_idx

# === 3. LOOP DI ESTRAZIONE ===
dataset = {
    "clap_single": [],   # Chunk centrale assoluto
    "clap_avg": [],      # Media temporale
    "clap_dominant": []  # Audio/Scena dominante
}

print(f"⏱️  Estrazione: {START_SEC}s -> {START_SEC + TOTAL_TR*TR_DURATION}s")
print(f"📊 Target: {TOTAL_TR} volumi (TR={TR_DURATION}s)")

chunk_duration = TR_DURATION / FPS_SAMPLING 
chunk_samples = int(chunk_duration * AUDIO_SR) 

for tr in tqdm(range(TOTAL_TR), desc="Processing TRs"):
    t_start = START_SEC + (tr * TR_DURATION)
    tr_clap_embeds = []
    
    for f in range(FPS_SAMPLING):
        offset = f * chunk_duration
        t_current = t_start + offset
        
        start_sample = int(t_current * AUDIO_SR)
        end_sample = start_sample + chunk_samples
        
        # Taglio e zero-padding se siamo alla fine o oltre la durata dell'audio
        if start_sample >= total_audio_samples:
            audio_chunk = np.zeros(chunk_samples, dtype=np.float32)
        else:
            audio_chunk = audio_waveform[start_sample:end_sample]
            if len(audio_chunk) < chunk_samples:
                audio_chunk = np.pad(audio_chunk, (0, chunk_samples - len(audio_chunk)), 'constant')

        c_emb = get_clap_emb(audio_chunk)
        tr_clap_embeds.append(c_emb)

    # 1. SINGLE (Frammento centrale della finestra)
    mid_idx = len(tr_clap_embeds) // 2
    dataset["clap_single"].append(tr_clap_embeds[mid_idx])
    
    # 2. AVG (Media su tutti gli 8 frammenti)
    dataset["clap_avg"].append(torch.mean(torch.stack(tr_clap_embeds), dim=0))
    
    # 3. DOMINANT (Identifica l'indice con audio più stabile)
    dom_idx = get_dominant_index(tr_clap_embeds)
    dataset["clap_dominant"].append(tr_clap_embeds[dom_idx])

# === 4. SALVATAGGIO ===
print("\n💾 Conversione e Salvataggio...")
final_path = os.path.join(DATA_DIR, "features_despicable_clap_only.pt")

for key in dataset:
    if len(dataset[key]) > 0:
        t = torch.cat(dataset[key], dim=0) 
        if len(t.shape) == 3 and t.shape[1] == 1:
            t = t.squeeze(1)
        dataset[key] = t
        print(f"  • {key}: {t.shape}")

assert dataset["clap_dominant"].shape[0] == TOTAL_TR, f"Errore: Generati {dataset['clap_dominant'].shape[0]} volumi invece di {TOTAL_TR}!"

torch.save(dataset, final_path)
print(f"\n✅ Dataset salvato: {final_path}")
print("   Contiene chiavi: clap_single, clap_avg, clap_dominant")