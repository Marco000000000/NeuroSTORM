import librosa
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# === CONFIGURAZIONE ===
VIDEO_PATH = "/home/mfinocchiaro/miccai2026/NeuroSTORM/descme_10min_frame_samecodec.mp4"
TR_FMRI = 0.8
AUDIO_SR = 48000
HOP_LENGTH = 512 # Risoluzione temporale standard per librosa

def main():
    print("🎵 Caricamento traccia audio dal video...")
    y, sr = librosa.load(VIDEO_PATH, sr=AUDIO_SR, mono=True)
    
    print("🌊 Calcolo del Flusso Spettrale (Onset Strength)...")
    # onset_strength calcola esattamente la differenza positiva tra spettrogrammi adiacenti
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP_LENGTH)
    
    # === ALLINEAMENTO ALLA fMRI (TR = 0.8s) ===
    # Calcoliamo quanti "frame" dell'onset envelope cadono esattamente dentro un TR
    frames_per_tr = int((TR_FMRI * sr) / HOP_LENGTH)
    total_trs = int(len(y) / (sr * TR_FMRI))
    
    flux_per_tr = []
    
    print(f"⏱️ Compressione della curva su {total_trs} volumi TR...")
    for i in range(total_trs):
        start_idx = i * frames_per_tr
        end_idx = start_idx + frames_per_tr
        
        # Facciamo la media della salienza in quel TR 
        # (puoi usare np.max se preferisci intercettare i picchi improvvisi)
        chunk_flux = np.mean(onset_env[start_idx:end_idx])
        flux_per_tr.append(chunk_flux)
        
    flux_per_tr = np.array(flux_per_tr)
    
    # Normalizziamo tra 0 e 1 per comodità visiva
    flux_per_tr = flux_per_tr / np.max(flux_per_tr)
    
    # Salviamo l'array numpy nel caso volessi usarlo per fare correlazioni con l'SVM
    np.save("spectral_flux_aligned.npy", flux_per_tr)
    print("💾 Array salvato come 'spectral_flux_aligned.npy'")

    # === PLOT ===
    print("📈 Generazione del grafico...")
    time_axis = np.arange(total_trs) * TR_FMRI
    
    plt.figure(figsize=(15, 5))
    plt.plot(time_axis, flux_per_tr, color='purple', linewidth=1.5, label='Flusso Spettrale (Media per TR)')
    plt.fill_between(time_axis, flux_per_tr, color='purple', alpha=0.2)
    
    # Evidenziamo i momenti di massima salienza acustica (top 10% dei TR)
    saliency_threshold = np.percentile(flux_per_tr, 90)
    plt.axhline(saliency_threshold, color='red', linestyle='--', alpha=0.7, label='Soglia di Alta Salienza (90° percentile)')
    
    plt.title("Salienza Acustica del Video (Flusso Spettrale sincronizzato alla fMRI)")
    plt.xlabel("Tempo (secondi)")
    plt.ylabel("Intensità del Flusso (Normalizzata)")
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    plot_filename = "spectral_flux_plot.png"
    plt.savefig(plot_filename, dpi=150)
    plt.close()
    print(f"✅ Grafico salvato come '{plot_filename}'")

if __name__ == "__main__":
    main()