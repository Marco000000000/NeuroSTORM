import os
import glob
import argparse
import numpy as np
import nibabel as nib
import torch
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

def load_nii(path):
    img = nib.load(path)
    return img.get_fdata(), img.affine

def process_single_subject(args):
    """
    Wrapper per processare un singolo soggetto (thread-safe).
    """
    func_path, mask_path, save_dir, target_shape = args
    
    try:
        filename = os.path.basename(func_path)
        sub_id = filename.split('_')[0]
        save_path = os.path.join(save_dir, f"{sub_id}.pt")

        # Se esiste già, salta (utile per riprendere il lavoro)
        if os.path.exists(save_path):
            return None # Skip silenzioso

        # 1. Carica
        # Carica come float32 per risparmiare RAM
        func_img = nib.load(func_path)
        func_data = func_img.get_fdata(dtype=np.float32)
        
        mask_img = nib.load(mask_path)
        mask_data = mask_img.get_fdata(dtype=np.float32)
        
        # 2. Maschera e Normalizza
        func_data[mask_data < 0.5] = 0
        
        mean = np.mean(func_data, axis=-1, keepdims=True)
        std = np.std(func_data, axis=-1, keepdims=True)
        std[std == 0] = 1.0
        func_data = (func_data - mean) / std
        func_data[mask_data < 0.5] = 0

        # 3. Converti in Tensore (Time, 1, X, Y, Z)
        # Nota: Non usiamo .copy() se non necessario per risparmiare RAM
        data_tensor = torch.from_numpy(func_data)
        data_tensor = data_tensor.permute(3, 0, 1, 2).unsqueeze(1)
        
        # 4. Crop/Pad a target_shape (es. 96x96x96)
        T, C, H, W, D = data_tensor.shape
        final_tensor = torch.zeros((T, C, *target_shape), dtype=torch.float32)
        
        # Centra
        c_h, c_w, c_d = H // 2, W // 2, D // 2
        tc_h, tc_w, tc_d = target_shape[0] // 2, target_shape[1] // 2, target_shape[2] // 2
        
        start_h = max(0, c_h - tc_h); end_h = min(H, c_h + tc_h)
        start_w = max(0, c_w - tc_w); end_w = min(W, c_w + tc_w)
        start_d = max(0, c_d - tc_d); end_d = min(D, c_d + tc_d)
        
        t_start_h = max(0, tc_h - (c_h - start_h)); t_end_h = t_start_h + (end_h - start_h)
        t_start_w = max(0, tc_w - (c_w - start_w)); t_end_w = t_start_w + (end_w - start_w)
        t_start_d = max(0, tc_d - (c_d - start_d)); t_end_d = t_start_d + (end_d - start_d)
        
        final_tensor[:, :, t_start_h:t_end_h, t_start_w:t_end_w, t_start_d:t_end_d] = \
            data_tensor[:, :, start_h:end_h, start_w:end_w, start_d:end_d]

        # 5. Salva
        torch.save(final_tensor, save_path)
        return None

    except Exception as e:
        return f"❌ Errore {os.path.basename(func_path)}: {e}"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_root", required=True)
    parser.add_argument("--save_root", required=True)
    parser.add_argument("--dataset_name", default="ucla")
    # DEFAULT MODIFICATO: 16 core (o meno se non disponibili)
    default_workers = min(16, multiprocessing.cpu_count())
    parser.add_argument("--num_workers", type=int, default=default_workers, 
                        help="Numero di processi paralleli (Default: 16)")
    args = parser.parse_args()
    
    if not os.path.exists(args.save_root): os.makedirs(args.save_root)
        
    func_files = glob.glob(os.path.join(args.load_root, "*preproc_bold.nii.gz"))
    
    print(f"🚀 Trovati {len(func_files)} file.")
    print(f"🔥 Utilizzo {args.num_workers} core CPU (Modalità Cluster-Safe).")
    
    tasks = []
    target_shape = (96, 96, 96)

    for func_path in func_files:
        mask_path = func_path.replace("preproc_bold.nii.gz", "brain_mask.nii.gz")
        if os.path.exists(mask_path):
            tasks.append((func_path, mask_path, args.save_root, target_shape))
        else:
            print(f"⚠️ Maschera mancante per {os.path.basename(func_path)}")

    # Esecuzione Parallela
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = [executor.submit(process_single_subject, task) for task in tasks]
        
        # tqdm per monitorare
        for future in tqdm(as_completed(futures), total=len(tasks), desc="Processing"):
            result = future.result()
            if result:
                print(result)

    print(f"\n✅ Conversione completata in {args.save_root}")

if __name__ == "__main__":
    main()