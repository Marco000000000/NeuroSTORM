import os
import pytorch_lightning as pl
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
# IMPORTANTE: Importa dal NUOVO file dataset
from datasets.fmri_datasets_miccai import HCP1200, ABCD, UKB, Cobre, ADHD200, UCLA, HCPEP, HCPTASK, GOD, MOVIE, TransDiag
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from .parser import str2bool
from sklearn.model_selection import StratifiedKFold, train_test_split
from collections import defaultdict
import random

def select_elements(S, n):
    level_count = defaultdict(int)
    for value in S.values():
        level_count[value[1]] += 1
    total_elements = sum(level_count.values())
    level_quota = {level: int(n * count / total_elements) for level, count in level_count.items()}
    remaining = n - sum(level_quota.values())
    levels = sorted(level_count.keys(), key=lambda x: -level_count[x])
    for i in range(remaining):
        level_quota[levels[i % len(levels)]] += 1
    selected_elements = []
    for level in level_quota:
        elements_of_level = [k for k, v in S.items() if v[1] == level]
        selected_elements.extend(random.sample(elements_of_level, level_quota[level]))
    S_prime = {k: S[k] for k in selected_elements}
    return S_prime

class fMRIDataModuleMICCAI(pl.LightningDataModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        # --- 1. Estrazione Parametri Chiave ---
        self.dataset_name = kwargs.get('dataset_name', 'UCLA')
        self.image_path = kwargs.get('image_path', './data')
        self.target_diagnosis = kwargs.get('target_diagnosis', 'SCHZ')
        
        # --- 2. FIX FONDAMENTALE PER "EXPLODE,ACCEPT" ---
        raw_events = kwargs.get('target_events', 'EXPLODE')
        
        if isinstance(raw_events, str):
            if ',' in raw_events:
                # Se c'è una virgola, splitta e pulisci gli spazi
                self.target_events = [x.strip() for x in raw_events.split(',')]
            else:
                # Se è una stringa singola, mettila in una lista
                self.target_events = [raw_events]
        else:
            # Se è già una lista, usala così com'è
            self.target_events = raw_events
            
        # Aggiorniamo hparams per coerenza nei log
        self.hparams.target_events = self.target_events
        # -------------------------------------------------

        # --- 3. Parametri Cross-Validation ---
        self.use_cv = kwargs.get('use_cv', False)
        self.num_folds = kwargs.get('num_folds', 5)
        self.fold_index = kwargs.get('fold_index', 0)
        
        # --- 4. Setup path standard (solo se NON siamo in CV) ---
        # Questo serve per mantenere retro-compatibilità con il vecchio metodo
        if not self.use_cv:
            if self.hparams.pretraining:
                split_dir_path = f'./data/splits/{self.dataset_name}/pretraining'
            else:
                split_dir_path = f'./data/splits/{self.dataset_name}'
            
            os.makedirs(split_dir_path, exist_ok=True)
            self.split_file_path = os.path.join(split_dir_path, f"split_fixed_{kwargs.get('dataset_split_num', 1)}.txt")
        
        self.setup() 

    def get_dataset(self):
        if self.hparams.dataset_name == "HCP1200": return HCP1200
        elif self.hparams.dataset_name == "ABCD": return ABCD
        elif self.hparams.dataset_name == 'UKB': return UKB
        elif self.hparams.dataset_name == 'Cobre': return Cobre
        elif self.hparams.dataset_name == 'ADHD200': return ADHD200
        elif self.hparams.dataset_name == 'UCLA': return UCLA
        # ... altri dataset ...
        elif self.hparams.dataset_name == 'HCPEP': return HCPEP
        elif self.hparams.dataset_name == 'GOD': return GOD
        elif self.hparams.dataset_name == 'HCPTASK': return HCPTASK
        elif self.hparams.dataset_name == 'MOVIE': return MOVIE
        elif self.hparams.dataset_name == 'TransDiag': return TransDiag
        else: raise NotImplementedError

    def make_subject_dict(self):
        img_root = os.path.join(self.hparams.image_path, 'img')
        final_dict = dict()

        # ... (Copia pure la logica per HCP, ABCD, etc dal vecchio file se ti servono) ...
        # Per brevità metto solo UCLA che è quello che ti serve per il MICCAI

        if self.hparams.dataset_name == "UCLA":
            subject_list = [subj for subj in os.listdir(img_root)]
            meta_data = pd.read_csv(os.path.join(self.hparams.image_path, "metadata", "ucla-rest.csv"))
            
            # --- SELEZIONE DINAMICA DIAGNOSI ---
            target_diag = getattr(self.hparams, 'target_diagnosis', 'SCHZ')
            print(f"🎯 Configuring UCLA for Task: {target_diag} vs CONTROL")

            for subject in subject_list:
                if subject in meta_data['subject_id'].values:
                    raw_target = meta_data[meta_data["subject_id"]==subject]['diagnosis'].values[0]
                    sex_val = meta_data[meta_data["subject_id"]==subject]["gender"].values[0]
                    sex = 1 if sex_val == "M" else 0
                    
                    target = -1
                    
                    if target_diag == 'SCHZ':
                        if raw_target == 'CONTROL': target = 0
                        elif raw_target == 'SCHZ': target = 1
                    elif target_diag == 'BIPOLAR':
                        if raw_target == 'CONTROL': target = 0
                        elif raw_target == 'BIPOLAR': target = 1
                    elif target_diag == 'ADHD':
                        if raw_target == 'CONTROL': target = 0
                        elif raw_target == 'ADHD': target = 1
                    elif target_diag == 'MULTICLASS':
                        if raw_target == 'CONTROL': target = 0
                        elif raw_target == 'SCHZ': target = 1
                        elif raw_target == 'BIPOLAR': target = 2
                        elif raw_target == 'ADHD': target = 3
                    
                    if target != -1:
                        final_dict[subject] = [sex, target]
            
            print('Load dataset UCLA, {} subjects'.format(len(final_dict)))

        # Se usi altri dataset nel paper, copia qui i loro blocchi if/elif
        
        return final_dict

    def setup_kfold(self, subject_dict):
        # ... (Identico a prima, copialo dal tuo cv_data_module.py) ...
        subjects, targets = [], []
        clean_dict = {}
        for subj, (sex, target) in subject_dict.items():
            if int(target) < self.hparams.num_classes:
                subjects.append(subj)
                targets.append(int(target))
                clean_dict[subj] = [sex, target]
        subjects, targets = np.array(subjects), np.array(targets)
        if len(subjects) == 0: raise ValueError("CV Setup: Nessun soggetto trovato.")

        skf = StratifiedKFold(n_splits=self.num_folds, shuffle=True, random_state=self.hparams.seed)
        splits = list(skf.split(subjects, targets))
        train_val_idx, test_idx = splits[self.fold_index]
        test_subjs = subjects[test_idx]
        X_train_val = subjects[train_val_idx]
        y_train_val = targets[train_val_idx]
        
        train_subjs_pure, val_subjs_pure, _, _ = train_test_split(
            X_train_val, y_train_val, test_size=0.10, stratify=y_train_val, random_state=self.hparams.seed
        )
        train_dict = {s: clean_dict[s] for s in train_subjs_pure}
        val_dict = {s: clean_dict[s] for s in val_subjs_pure}
        test_dict = {s: clean_dict[s] for s in test_subjs}
        
        print(f"\n⚡ [CV Fold {self.fold_index+1}/{self.num_folds}] Setup Distinto:")
        print(f"   Train: {len(train_dict)} | Val: {len(val_dict)} | Test: {len(test_dict)}")
        return train_dict, val_dict, test_dict

    def determine_split_randomly(self, S):
        # ... (Identico a prima, copialo) ...
        # Serve solo se non usi CV, ma per sicurezza copialo
        np.random.seed(self.hparams.seed)
        S_keys = list(S.keys())
        n_total = len(S_keys)
        n_train = int(n_total * self.hparams.train_split)
        n_val = int(n_total * self.hparams.val_split)
        if self.hparams.downstream_task_type == 'classification':
            S_train = select_elements(S, n_train)
            S_train_keys = list(S_train.keys())
            S_remaining = {k: v for k, v in S.items() if k not in S_train}
            S_val = select_elements(S_remaining, n_val)
            S_val_keys = list(S_val.keys())
            S_test = {k: v for k, v in S_remaining.items() if k not in S_val}
            S_test_keys = list(S_test.keys())
        else:
            S_train_keys = np.random.choice(S_keys, n_train, replace=False)
            S_remaining_keys = np.setdiff1d(S_keys, S_train_keys)
            S_remaining = {k: S[k] for k in S_remaining_keys}
            S_val_keys = np.random.choice(list(S_remaining.keys()), n_val, replace=False)
            S_test_keys = np.setdiff1d(list(S_remaining.keys()), S_val_keys)
        
        if len(S_test_keys) == 0:
            half_val = len(S_val_keys) // 2
            S_test_keys = S_val_keys[half_val:]
            S_val_keys = S_val_keys[:half_val]
        
        self.save_split({"train_subjects": S_train_keys, "val_subjects": S_val_keys, "test_subjects": S_test_keys})
        return S_train_keys, S_val_keys, S_test_keys

    def save_split(self, sets_dict):
        with open(self.split_file_path, "w+") as f:
            for name, subj_list in sets_dict.items():
                f.write(name + "\n")
                for subj_name in subj_list: f.write(str(subj_name) + "\n")
    
    def load_split(self):
        subject_order = open(self.split_file_path, "r").readlines()
        subject_order = [x[:-1] for x in subject_order]
        train_index = np.argmax(["train" in line for line in subject_order])
        val_index = np.argmax(["val" in line for line in subject_order])
        test_index = np.argmax(["test" in line for line in subject_order])
        return subject_order[train_index + 1 : val_index], subject_order[val_index + 1 : test_index], subject_order[test_index + 1 :]

    def setup(self, stage=None):
        Dataset = self.get_dataset()
        t_events = getattr(self.hparams, 'target_events', 'EXPLODE')
        if isinstance(t_events, str): t_events = [t_events]

        params = {
            "root": self.hparams.image_path, "img_size": self.hparams.img_size,
            "sequence_length": self.hparams.sequence_length, "contrastive": self.hparams.use_contrastive,
            "contrastive_type": self.hparams.contrastive_type, "mae": self.hparams.use_mae,
            "stride_between_seq": self.hparams.stride_between_seq, "stride_within_seq": self.hparams.stride_within_seq,
            "with_voxel_norm": self.hparams.with_voxel_norm, "downstream_task_id": self.hparams.downstream_task_id,
            "task_name": self.hparams.task_name, "shuffle_time_sequence": self.hparams.shuffle_time_sequence,
            "label_scaling_method": self.hparams.label_scaling_method, "num_classes": self.hparams.num_classes,
            "dtype": 'float16', "target_events": t_events
        }

        full_subject_dict = self.make_subject_dict()

        if self.use_cv:
            train_dict, val_dict, test_dict = self.setup_kfold(full_subject_dict)
        else:
            if os.path.exists(self.split_file_path):
                train_names, val_names, test_names = self.load_split()
                train_dict = {k: full_subject_dict[k] for k in train_names if k in full_subject_dict}
                val_dict = {k: full_subject_dict[k] for k in val_names if k in full_subject_dict}
                test_dict = {k: full_subject_dict[k] for k in test_names if k in full_subject_dict}
            else:
                train_names, val_names, test_names = self.determine_split_randomly(full_subject_dict)
                train_dict = {k: full_subject_dict[k] for k in train_names if k in full_subject_dict}
                val_dict = {k: full_subject_dict[k] for k in val_names if k in full_subject_dict}
                test_dict = {k: full_subject_dict[k] for k in test_names if k in full_subject_dict}

        self.train_dataset = Dataset(**params, subject_dict=train_dict, use_augmentations=False, train=True)
        self.val_dataset = Dataset(**params, subject_dict=val_dict, use_augmentations=False, train=False)
        self.test_dataset = Dataset(**params, subject_dict=test_dict, use_augmentations=False, train=False)
        
        print(f"Samples -> Train: {len(self.train_dataset.data)}, Val: {len(self.val_dataset.data)}, Test: {len(self.test_dataset.data)}")
        
        def get_params(train):
            return {
                "batch_size": self.hparams.batch_size if train else self.hparams.eval_batch_size,
                "num_workers": self.hparams.num_workers,
                "drop_last": True, "pin_memory": False, "shuffle": train,
                "persistent_workers": (train and (self.hparams.strategy == 'ddp') and (self.hparams.num_workers > 0))
            }
        self.train_loader = DataLoader(self.train_dataset, **get_params(train=True))
        self.val_loader = DataLoader(self.val_dataset, **get_params(train=False))
        self.test_loader = DataLoader(self.test_dataset, **get_params(train=False))

    def train_dataloader(self): return self.train_loader
    def val_dataloader(self):
        # Restituisce una LISTA: [0] = Validation (per Early Stopping), [1] = Test (per Monitoraggio)
        return [self.val_loader, self.test_loader]
    def test_dataloader(self): return self.test_loader
    def predict_dataloader(self): return self.test_dataloader()

    @classmethod
    def add_data_specific_args(cls, parent_parser: ArgumentParser, **kwargs) -> ArgumentParser:
        parser = ArgumentParser(parents=[parent_parser], add_help=True, formatter_class=ArgumentDefaultsHelpFormatter)
        group = parser.add_argument_group("DataModule arguments")
        # Existing
        group.add_argument("--dataset_split_num", type=int, default=1)
        group.add_argument("--label_scaling_method", default="standardization", choices=["minmax","standardization"])
        group.add_argument("--image_path", default=None)
        group.add_argument("--bad_subj_path", default=None)
        group.add_argument("--train_split", default=0.9, type=float)
        group.add_argument("--val_split", default=0.1, type=float)
        group.add_argument("--batch_size", type=int, default=4)
        group.add_argument("--eval_batch_size", type=int, default=8)
        group.add_argument("--img_size", nargs="+", default=[96, 96, 96, 20], type=int)
        group.add_argument("--sequence_length", type=int, default=20)
        group.add_argument("--stride_between_seq", type=int, default=1)
        group.add_argument("--stride_within_seq", type=int, default=1)
        group.add_argument("--num_workers", type=int, default=8)
        group.add_argument("--with_voxel_norm", type=str2bool, default=False)
        group.add_argument("--shuffle_time_sequence", action='store_true')
        group.add_argument("--limit_training_samples", type=float, default=None)
        # NEW
        group.add_argument("--target_diagnosis", type=str, default="SCHZ", choices=["SCHZ", "BIPOLAR", "ADHD", "MULTICLASS"])
        group.add_argument("--target_events", type=str, default="EXPLODE")
        return parser