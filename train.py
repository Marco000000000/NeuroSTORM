import os
import sys
import glob
import numpy as np
import pandas as pd  # Serve pandas per leggere i tsv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Aggiungi path NeuroSTORM
sys.path.append(os.path.abspath("NeuroSTORM"))

try:
    from models.neurostorm import NeuroSTORM as NeuroSTORM_Model
except ImportError:
    print("❌ Errore critico: Impossibile importare NeuroSTORM.")
    sys.exit(1)

# --- CONFIGURAZIONE ---
DATA_ROOT = "ucla_neurostorm_final"   # Dove sono i .pt
RAW_ROOT = "ucla_raw"                 # Dove sono i .tsv (struttura BIDS)
LABEL_ROOT = "ucla_labels"
CHECKPOINT_PATH = "NeuroSTORM/pretrained_models/fmrifound/pt_fmrifound_mae_ratio0.5.ckpt"

BATCH_SIZE = 32
LEARNING_RATE = 5e-5
EPOCHS = 30
SEQ_LEN = 20
EMBED_DIM = 36
TR = 2.0  # Tempo di Ripetizione (in secondi). Per UCLA solitamente è 2s. Verifica!

# Quali eventi ci interessano?
# Nel task BART solitamente sono: 'BALOON' (pumping), 'EXPLODE' (boom), 'CASHOUT' (stop)
TARGET_EVENTS = ['BALOON', 'EXPLODE'] 

# --- DATASET BASATO SU EVENTI ---
class EventAwareDataset(Dataset):
    def __init__(self, file_list, raw_root, label_map, seq_len=20, tr=2.0, mode='train'):
        self.seq_len = seq_len
        self.tr = tr
        self.samples = [] 
        
        print(f"🔨 Indicizzazione Eventi ({mode})...")
        
        for fpath in tqdm(file_list):
            sub_id = os.path.basename(fpath).split('.')[0] # es: sub-10159
            if sub_id not in label_map:
                continue
                
            label = label_map[sub_id]
            
            # 1. Costruisci path del TSV
            # ucla_raw/sub-10159/func/sub-10159_task-bart_events.tsv
            tsv_path = os.path.join(raw_root, sub_id, "func", f"{sub_id}_task-bart_events.tsv")
            
            if not os.path.exists(tsv_path):
                # Fallback: a volte sono in phenotype o raw data structure diversa
                # Prova a cercarlo ricorsivamente se non è nel path standard
                found = glob.glob(os.path.join(raw_root, "**", f"{sub_id}*events.tsv"), recursive=True)
                if found:
                    tsv_path = found[0]
                else:
                    # print(f"⚠️ TSV mancante per {sub_id}, salto.")
                    continue
            
            # 2. Leggi TSV e trova onsets
            try:
                events_df = pd.read_csv(tsv_path, sep='\t')
                
                # Filtra per eventi interessanti
                # Assumiamo che la colonna si chiami 'trial_type' o 'event_type'
                # Normalizziamo i nomi colonne
                events_df.columns = [c.lower() for c in events_df.columns]
                
                # Cerca colonna tipo evento
                type_col = 'trial_type' if 'trial_type' in events_df.columns else 'event_type'
                
                # Filtra
                relevant_events = events_df[events_df[type_col].isin(TARGET_EVENTS)]
                
                # Carica shape del PT per evitare out-of-bounds (senza caricare dati)
                # Hack veloce: assumiamo una durata media o carichiamo header se possibile.
                # Qui per sicurezza carichiamo il tensore (lento in init, ma sicuro)
                # Se troppo lento, ottimizzare salvando metadata.
                pt_data = torch.load(fpath, map_location='cpu')
                max_frames = pt_data.shape[0]
                del pt_data # libera memoria
                
                # 3. Converti Onset (sec) -> Frame Index
                for _, row in relevant_events.iterrows():
                    onset_sec = row['onset']
                    onset_frame = int(np.round(onset_sec / self.tr))
                    
                    # Controlla se la finestra sta dentro il video
                    if onset_frame + self.seq_len <= max_frames:
                        self.samples.append({
                            'pt_path': fpath,
                            'label': label,
                            'start_frame': onset_frame,
                            'event_type': row[type_col]
                        })
                        
            except Exception as e:
                # print(f"Errore processamento {sub_id}: {e}")
                pass

        print(f"✅ {mode.upper()}: Trovati {len(self.samples)} eventi totali.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        
        try:
            full_seq = torch.load(item['pt_path'])
            
            start = item['start_frame']
            clip = full_seq[start : start + self.seq_len]
            
            # (Time, 1, H, W, D) -> (1, H, W, D, Time)
            clip = clip.permute(1, 2, 3, 4, 0)
            
            return clip, item['label']
            
        except Exception as e:
            return torch.zeros((1, 96, 96, 96, self.seq_len)), item['label']

# --- MODELLO ---
class NeuroSTORMFineTuningClassifier(nn.Module):
    def __init__(self, num_classes=4, embed_dim=36):
        super().__init__()
        self.encoder = NeuroSTORM_Model(
            img_size=(96, 96, 96, SEQ_LEN),
            in_chans=1,
            embed_dim=embed_dim,
            window_size=[4, 4, 4, 4],
            first_window_size=[2, 2, 2, 2],
            patch_size=[6, 6, 6, 1],
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            c_multiplier=2,
            last_layer_full_MSA=False
        )
        if os.path.exists(CHECKPOINT_PATH):
            print(f"🔄 Caricamento pesi...")
            ckpt = torch.load(CHECKPOINT_PATH, map_location='cpu')
            sd = ckpt.get('state_dict', ckpt)
            clean_sd = {k.replace('model.', '').replace('net.', ''): v for k, v in sd.items()}
            self.encoder.load_state_dict(clean_sd, strict=False)
        
        # Unfreeze
        for param in self.encoder.parameters():
            param.requires_grad = True
            
        final_dim = int(embed_dim * (2 ** (len([2, 2, 6, 2]) - 1)))
        self.clf = nn.Sequential(
            nn.Linear(final_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        features = self.encoder(x)
        pooled = features.mean(dim=(2, 3, 4, 5)) 
        return self.clf(pooled)

# --- MAIN ---
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔥 Training EVENT-BASED su {device}")
    
    # Carica Mappe
    weights_path = os.path.join(LABEL_ROOT, "diagnosis_weights.pt")
    map_path = os.path.join(LABEL_ROOT, "diagnosis_map.npy")
    class_weights = torch.load(weights_path).to(device)
    label_map = np.load(map_path, allow_pickle=True).item()
    
    # File Discovery
    all_files = glob.glob(os.path.join(DATA_ROOT, "*.pt"))
    valid_files = [f for f in all_files if os.path.basename(f).split('.')[0] in label_map]
    
    # Split Soggetti
    train_files, val_files = train_test_split(valid_files, test_size=0.2, random_state=42)
    
    # Crea Dataset
    # Assumiamo TR=2.0 (standard UCLA). Se diverso, cambialo qui.
    train_ds = EventAwareDataset(train_files, RAW_ROOT, label_map, seq_len=SEQ_LEN, tr=TR, mode='train')
    val_ds = EventAwareDataset(val_files, RAW_ROOT, label_map, seq_len=SEQ_LEN, tr=TR, mode='val')
    
    if len(train_ds) == 0:
        print("❌ Nessun evento trovato! Controlla i percorsi dei file TSV.")
        return

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)

    # Modello
    model = NeuroSTORMFineTuningClassifier(num_classes=4, embed_dim=EMBED_DIM).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.05)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    best_acc = 0.0
    
    print(f"🚀 Inizio Training...")
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Ep {epoch+1}")
        
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_acc = 100 * correct / total if total > 0 else 0
        print(f"   📊 Val Accuracy (su eventi): {val_acc:.2f}%")
        
        scheduler.step(val_acc)
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_event_model.pth")
            print("   💾 Best Model Saved")

if __name__ == "__main__":
    main()