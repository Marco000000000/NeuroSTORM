import os
import sys
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
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
DATA_ROOT = "ucla_neurostorm_final"
LABEL_ROOT = "ucla_labels"
BATCH_SIZE = 64
GRAD_ACCUMULATION = 4
LEARNING_RATE = 1e-3
EPOCHS = 50   # Aumentiamo le epoche massime, tanto c'è l'Early Stopping
SEQ_LEN = 20
IMG_SIZE = (96, 96, 96, SEQ_LEN)
EMBED_DIM = 36

PRETRAINED_PATH = "NeuroSTORM/pretrained_models/fmrifound/pt_fmrifound_mae_ratio0.5.ckpt" 

# --- CLASSE EARLY STOPPING ---
class EarlyStopping:
    def __init__(self, patience=7, delta=0, path='checkpoint.pth'):
        self.patience = patience
        self.delta = delta
        self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_acc_max = -np.inf

    def __call__(self, val_acc, model):
        score = val_acc

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_acc, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'   ⏳ EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_acc, model)
            self.counter = 0

    def save_checkpoint(self, val_acc, model):
        print(f'   ✅ Validation accuracy increased ({self.val_acc_max:.2f}% --> {val_acc:.2f}%).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_acc_max = val_acc

# --- DATASET ---
class UCLADiagnosisDataset(Dataset):
    def __init__(self, data_dir, label_map_path, seq_len=20):
        self.data_dir = data_dir
        self.seq_len = seq_len
        self.label_map = np.load(label_map_path, allow_pickle=True).item()
        
        self.files = []
        self.labels_list = []
        
        all_files = glob.glob(os.path.join(data_dir, "*.pt"))
        for f in all_files:
            sub_id = os.path.basename(f).split('.')[0]
            if sub_id in self.label_map:
                self.files.append(f)
                self.labels_list.append(self.label_map[sub_id])
                
        print(f"🧠 Dataset totale: {len(self.files)} soggetti.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        sub_id = os.path.basename(path).split('.')[0]
        label = self.label_map[sub_id]
        
        try:
            full_seq = torch.load(path)
            total_time = full_seq.shape[0]
            
            if total_time > self.seq_len:
                start = np.random.randint(0, total_time - self.seq_len)
                clip = full_seq[start : start + self.seq_len]
            else:
                pad = torch.zeros((self.seq_len - total_time, *full_seq.shape[1:]))
                clip = torch.cat([full_seq, pad], dim=0)
                
            clip = clip.permute(1, 2, 3, 4, 0)
            return clip, label
            
        except Exception as e:
            print(f"Errore caricamento {sub_id}: {e}")
            raise e

class NeuroSTORMFrozenClassifier(nn.Module):
    def __init__(self, num_classes=4, embed_dim=36):
        super().__init__()
        
        # Inizializza NeuroSTORM
        self.encoder = NeuroSTORM_Model(
            img_size=IMG_SIZE,
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
        
        # --- CARICAMENTO PESI CORRETTO PER .CKPT ---
        if os.path.exists(PRETRAINED_PATH):
            print(f"🔄 Caricamento checkpoint: {PRETRAINED_PATH}")
            checkpoint = torch.load(PRETRAINED_PATH, map_location='cpu')
            
            # 1. Estrai lo state_dict se siamo in un checkpoint Lightning
            if 'state_dict' in checkpoint:
                print("   📦 Formato Lightning rilevato: estraggo 'state_dict'...")
                raw_state_dict = checkpoint['state_dict']
            else:
                raw_state_dict = checkpoint

            # 2. Pulisci le chiavi (rimuovi il prefisso "model.")
            clean_state_dict = {}
            for k, v in raw_state_dict.items():
                # Il modello pre-addestrato ha i pesi salvati come "model.encoder..." o "model..."
                if k.startswith('model.'):
                    new_key = k.replace('model.', '', 1)  # Rimuovi solo il primo 'model.'
                    clean_state_dict[new_key] = v
                else:
                    clean_state_dict[k] = v
            
            # 3. Carica i pesi puliti
            try:
                # strict=False è necessario perché il checkpoint contiene anche la 'output_head' che noi buttiamo via
                msg = self.encoder.load_state_dict(clean_state_dict)
                print(f"✅ Pesi caricati con successo!")
                print(f"   Missing keys (dovrebbe essere vuoto o trascurabile): {len(msg.missing_keys)}")
                print(f"   Unexpected keys (la vecchia head): {len(msg.unexpected_keys)}")
            except RuntimeError as e:
                print(f"❌ Errore di dimensione (Size Mismatch): {e}")
                print("   SUGGERIMENTO: Prova a cambiare EMBED_DIM a 24 o 48 nello script.")
                sys.exit(1)
        else:
            print(f"⚠️ FILE NON TROVATO: {PRETRAINED_PATH}")
            print("   Sto inizializzando l'encoder con pesi CASUALI (Sconsigliato!)")

        # Freeze dell'encoder (Prompt Tuning)
        for param in self.encoder.parameters():
            param.requires_grad = False
            
        # Calcolo dimensione output finale (per embed_dim=36 -> 288)
        final_dim = int(embed_dim * (2 ** (len([2, 2, 6, 2]) - 1)))
        
        self.clf = nn.Sequential(
            nn.Linear(final_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        with torch.no_grad():
            features = self.encoder(x)
        # Global Average Pooling su (D, H, W, T) -> dimensioni 2,3,4,5
        pooled = features.mean(dim=(2, 3, 4, 5)) 
        return self.clf(pooled)

# --- MAIN ---
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔥 Training su {device}")
    
    weights_path = os.path.join(LABEL_ROOT, "diagnosis_weights.pt")
    map_path = os.path.join(LABEL_ROOT, "diagnosis_map.npy")
    class_weights = torch.load(weights_path).to(device)
    
    dataset = UCLADiagnosisDataset(DATA_ROOT, map_path, seq_len=SEQ_LEN)
    
    train_idx, val_idx = train_test_split(
        np.arange(len(dataset)),
        test_size=0.2,
        shuffle=True,
        stratify=dataset.labels_list,
        random_state=42
    )
    
    train_ds = Subset(dataset, train_idx)
    val_ds = Subset(dataset, val_idx)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    model = NeuroSTORMFrozenClassifier(num_classes=4, embed_dim=EMBED_DIM).to(device)
    
    # Optimizer & Scheduler
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                                lr=LEARNING_RATE, 
                                weight_decay=0.05)    
    # SCHEDULER: Reduce LR quando la validation accuracy smette di salire
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, verbose=True
    )
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Early Stopping
    early_stopping = EarlyStopping(patience=8, path='best_diagnosis_model.pth')

    print(f"🚀 Inizio Training per max {EPOCHS} epoche...")
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Ep {epoch+1}")
        for i, (inputs, labels) in enumerate(pbar):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss = loss / GRAD_ACCUMULATION
            loss.backward()
            
            if (i + 1) % GRAD_ACCUMULATION == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            train_loss += loss.item() * GRAD_ACCUMULATION
            pbar.set_postfix({'loss': loss.item() * GRAD_ACCUMULATION})
            
        # Validation
        model.eval()
        correct = 0
        total = 0
        val_loss = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_acc = 100 * correct / total
        avg_val_loss = val_loss / len(val_loader)
        
        # Stampa i risultati
        print(f"   📊 Val Loss: {avg_val_loss:.4f} | Val Accuracy: {val_acc:.2f}%")
        
        # Step dello scheduler
        scheduler.step(val_acc)
        
        # Check Early Stopping
        early_stopping(val_acc, model)
        if early_stopping.early_stop:
            print("🛑 Early stopping attivato!")
            break

if __name__ == "__main__":
    main()