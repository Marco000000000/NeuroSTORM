import argparse
import os
import pickle
import time
from typing import Tuple

import h5py
import nibabel as nib
import numpy as np
from tqdm import tqdm

MASK_DATASETS = {'hcpya', 'hcpa', 'hcpd', 'ukb', 'hcptask'}


def compute_foreground_mean_std(data: np.ndarray, background: np.ndarray) -> Tuple[float, float]:
    foreground = ~background
    if not np.any(foreground):
        values = data.ravel()
    else:
        values = data[foreground]
    return float(values.mean()), float(values.std())


def load_background(dataset_name: str, data: np.ndarray, file_path: str) -> np.ndarray:
    if dataset_name in MASK_DATASETS:
        return data == 0
    mask_path = None
    if dataset_name in {'abcd', 'cobre', 'hcpep'}:
        mask_path = file_path[:-19] + 'brain_mask.nii.gz'
    elif dataset_name == 'movie':
        mask_path = file_path[:-57] + 'space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz'
    elif dataset_name == 'transdiag':
        mask_path = file_path[:-19] + 'brainmask.nii.gz'
    elif dataset_name == 'ucla':
        mask_path = file_path[:-14] + 'brainmask.nii.gz'
    if mask_path and os.path.exists(mask_path):
        mask_img = nib.load(mask_path)
        return mask_img.get_fdata() == 0
    try:
        import ipdb; ipdb.set_trace()
    except Exception:
        pass
    return data == 0


def save_h5(data: np.ndarray, file_path: str):
    data = data.astype(np.float32, copy=False)
    if file_path.endswith('.nii.gz'):
        h5_path = file_path[:-7] + '.h5'
    elif file_path.endswith('.nii'):
        h5_path = file_path[:-4] + '.h5'
    else:
        h5_path = file_path + '.h5'
    with h5py.File(h5_path, 'w') as h5f:
        h5f.create_dataset('data', data=data, dtype='float32', compression='lzf')
    return h5_path


def process_file(dataset_name: str, file_path: str) -> Tuple[float, float, float, float]:
    t_read0 = time.time()
    img = nib.load(file_path)
    data = img.get_fdata()
    t_read = time.time() - t_read0

    t_proc0 = time.time()
    background = load_background(dataset_name, data, file_path)
    mean, std = compute_foreground_mean_std(data, background)
    data[background] = 0.0
    t_proc = time.time() - t_proc0

    t_save0 = time.time()
    save_h5(data, file_path)
    t_save = time.time() - t_save0

    os.remove(file_path)
    return mean, std, t_read, t_proc, t_save


def get_subject_id(filename: str) -> str:
    if filename.endswith('.nii.gz'):
        return filename[:-7]
    if filename.endswith('.nii'):
        return filename[:-4]
    return os.path.splitext(filename)[0]


def main():
    parser = argparse.ArgumentParser(
        description="Convert fMRI NIfTI to HDF5 (float32, no compression, background=0), "
                    "compute foreground mean/std, store stats in a pickle file.")
    parser.add_argument('--dataset_name', required=True, help='Dataset name')
    parser.add_argument('--load_root', required=True, help='Directory containing .nii or .nii.gz files')
    args = parser.parse_args()

    ds = args.dataset_name.lower()
    files = sorted([f for f in os.listdir(args.load_root) if f.endswith(('.nii', '.nii.gz'))])

    stats_dict = {}
    total_read = total_proc = total_save = 0.0

    with tqdm(total=len(files), desc='Processing') as pbar:
        for idx, fname in enumerate(files, 1):
            fpath = os.path.join(args.load_root, fname)
            mean, std, t_read, t_proc, t_save = process_file(ds, fpath)

            total_read += t_read
            total_proc += t_proc
            total_save += t_save

            sid = get_subject_id(fname)
            stats_dict[sid] = {'foreground_mean': mean, 'foreground_std': std}

            pbar.set_postfix(read=f'{total_read/idx:.2f}s',
                             proc=f'{total_proc/idx:.2f}s',
                             save=f'{total_save/idx:.2f}s')
            pbar.update(1)

    pkl_name = f'{ds}.pkl'
    with open(pkl_name, 'wb') as pf:
        pickle.dump(stats_dict, pf)


if __name__ == '__main__':
    main()
