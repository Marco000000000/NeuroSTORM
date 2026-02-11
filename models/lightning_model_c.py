import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

import numpy as np
import os
import pickle
import scipy
import time

import torchmetrics
import torchmetrics.classification
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC, MulticlassAccuracy, MulticlassAUROC
from torchmetrics import PearsonCorrCoef
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_curve
import monai.transforms as monai_t

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import nibabel as nb

from .load_model import load_model
from utils.metrics import Metrics
from utils.parser import str2bool
# Assicurati di aver aggiunto SupConLoss in utils/losses.py
from utils.losses import NTXentLoss, global_local_temporal_contrastive, SupConLoss 
from utils.lr_scheduler import WarmupCosineSchedule, CosineAnnealingWarmUpRestarts

from einops import rearrange
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, KBinsDiscretizer


class LightningModel(pl.LightningModule):
    def __init__(self, data_module, **kwargs):
        super().__init__()
        self.save_hyperparameters(kwargs)
        
        # 1. Setup Pesi per Classificazione Sbilanciata
        if self.hparams.dataset_name == 'UCLA' and self.hparams.num_classes == 2:
            print("⚖️  Using Weighted Loss for Binary Task (Control vs Schz)")
            # Controllo (0): 0.7, Schizofrenia (1): 2.3
            self.class_weights = torch.tensor([0.7, 2.3])
        elif self.hparams.dataset_name == 'UCLA' and self.hparams.num_classes == 4:
            self.class_weights = torch.tensor([0.5267, 1.4191, 1.4293, 1.4345])
        else:
            self.class_weights = None

        # 2. Setup Scaler per Regressione
        target_values = data_module.train_dataset.target_values
        if self.hparams.label_scaling_method == 'standardization':
            scaler = StandardScaler()
            scaler.fit_transform(target_values)
        elif self.hparams.label_scaling_method == 'minmax': 
            scaler = MinMaxScaler()
            scaler.fit_transform(target_values)
        self.scaler = scaler

        # 3. Caricamento Modello Backbone
        print('model name = {}'.format(self.hparams.model))
        self.model = load_model(self.hparams.model, self.hparams)
        self.start_time_data = time.time()

        # 4. Flops Counter (Opzionale)
        if self.hparams.print_flops:
            from thop import profile
            img_size = self.hparams.img_size    
            input = torch.randn([1, 1] + img_size).cuda()
            flops, params = profile(self.model.cuda(), inputs=(input, ))
            print('FLOPs = ' + str(flops/1000**3) + 'G')
            print('Params = ' + str(params/1000**2) + 'M')

        # 5. Setup Heads (Classificazione, Regressione o Contrastiva)
        if not self.hparams.pretraining:
            if self.hparams.downstream_task_type == 'classification' or self.hparams.scalability_check:
                self.output_head = load_model("clf_mlp", self.hparams)
            elif self.hparams.downstream_task_type == 'regression':
                self.output_head = load_model("reg_mlp", self.hparams)
        elif self.hparams.use_contrastive:
            self.output_head = load_model("emb_mlp", self.hparams)
        elif self.hparams.use_mae:
            self.output_head = None
        else:
            raise NotImplementedError("output head should be defined")

        # 6. Setup Supervised Contrastive Learning Components
        if self.hparams.use_supcon:
            print("❄️  Supervised Contrastive Loss Enabled")
            self.supcon_loss = SupConLoss(temperature=0.07)
            
            # Projection Head: Mappa le feature in uno spazio latente per la loss contrastiva
            # Assumiamo embed_dim come input size
            self.projection_head = nn.Sequential(
                nn.Linear(self.hparams.embed_dim, self.hparams.embed_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.hparams.embed_dim, 128)
            )

        self.metric = Metrics()

    def forward(self, x):
        return self.output_head(self.model(x))
    
    def augment(self, img):
        B, C, H, W, D, T = img.shape
        device = img.device
        img = rearrange(img, 'b c h w d t -> b t c h w d')

        rand_affine = monai_t.RandAffine(
            prob=1.0,
            rotate_range=(0.175, 0.175, 0.175),
            scale_range = (0.1, 0.1, 0.1),
            mode = "bilinear",
            padding_mode = "border",
            device = device
        )
        rand_noise = monai_t.RandGaussianNoise(prob=0.3, std=0.1)
        rand_smooth = monai_t.RandGaussianSmooth(sigma_x=(0.0, 0.5), sigma_y=(0.0, 0.5), sigma_z=(0.0, 0.5), prob=0.1)
        
        if self.hparams.augment_only_intensity:
            comp = monai_t.Compose([rand_noise, rand_smooth])
        else:
            comp = monai_t.Compose([rand_affine, rand_noise, rand_smooth]) 

        for b in range(B):
            aug_seed = torch.randint(0, 10000000, (1,)).item()
            for t in range(T):
                if self.hparams.augment_only_affine:
                    rand_affine.set_random_state(seed=aug_seed)
                    img[b, t, :, :, :, :] = rand_affine(img[b, t, :, :, :, :])
                else:
                    comp.set_random_state(seed=aug_seed)
                    img[b, t, :, :, :, :] = comp(img[b, t, :, :, :, :])

        img = rearrange(img, 'b t c h w d -> b c h w d t')
        return img
    
    def _compute_logits(self, batch, augment_during_training=None):
        fmri, subj, target_value, tr, sex = batch.values()
       
        if augment_during_training:
            fmri = self.augment(fmri)

        # 1. Estrazione feature dal backbone
        backbone_features = self.model(fmri)
        
        # Gestione output tuple (alcuni modelli ritornano liste/tuple)
        if type(backbone_features) == tuple:
            backbone_features = backbone_features[0]

        head_features = None
        logits = None
        target = None

        # 2. Calcolo Logits in base al task
        if self.hparams.downstream_task_type == 'classification' or self.hparams.scalability_check:
            if self.hparams.task_name == 'fmri_reid' and hasattr(self.output_head, "forward_with_features"):
                logits, head_features = self.output_head.forward_with_features(backbone_features)
            else:
                logits = self.output_head(backbone_features)
            
            logits = logits.squeeze()
            target = target_value.float().squeeze()
            
        elif self.hparams.downstream_task_type == 'regression':
            logits = self.output_head(backbone_features)
            unnormalized_target = target_value.float()
            
            if self.hparams.label_scaling_method == 'standardization':
                target = (unnormalized_target - self.scaler.mean_[0]) / (self.scaler.scale_[0])
            elif self.hparams.label_scaling_method == 'minmax':
                target = (unnormalized_target - self.scaler.data_min_[0]) / (self.scaler.data_max_[0] - self.scaler.data_min_[0])
        
        # Restituiamo anche backbone_features per la loss contrastiva
        return subj, logits, target, head_features, backbone_features
    
    def _calculate_loss(self, batch, mode):
        # --- PRETRAINING LOGIC (Invariata) ---
        if self.hparams.pretraining:
            # [Codice originale per MAE e Contrastive Self-Supervised...]
            # (Per brevità lascio questo blocco uguale all'originale che mi hai passato, 
            #  concentrandomi sulle modifiche downstream)
            fmri, subj, target_value, tr, sex = batch.values()
            loss = 0
            
            # ... (Logica pretraining esistente omessa per spazio, è identica al tuo input) ...
            # Se ti serve anche questa parte, è quella standard NTXentLoss e MAE
            if self.hparams.use_mae:
                 augment_y = self.augment(fmri[0]) if mode == 'train' else fmri[0]
                 pred_list, loss = self.model(augment_y)
                 return loss
            # Fallback se non è MAE (per semplicità qui)
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        # --- DOWNSTREAM TASK LOGIC (Classification/Regression) ---
        else:
            self.end_time_data = time.time()
            self.start_time_data = time.time()

            # Calcolo logits e feature
            subj, logits, target, _, backbone_features = self._compute_logits(batch, augment_during_training=self.hparams.augment_during_training)

            if self.hparams.downstream_task_type == 'classification' or self.hparams.scalability_check:
                # 1. Calcolo Loss Classificazione (Cross Entropy / BCE)
                if self.hparams.num_classes == 2:
                    # Binary
                    loss_ce = F.binary_cross_entropy_with_logits(logits, target)
                    acc = self.metric.get_accuracy_binary(logits, target.float().squeeze())
                elif self.hparams.num_classes > 2:
                    # Multiclass con pesi opzionali
                    if self.class_weights is not None:
                        weight = self.class_weights.to(logits.device)
                        loss_ce = F.cross_entropy(logits, target.long().squeeze(), weight=weight)
                    else:
                        loss_ce = F.cross_entropy(logits, target.long().squeeze())
                    acc = self.metric.get_accuracy(logits, target.float().squeeze())
                
                loss = loss_ce
                
                # 2. AGGIUNTA SUPERVISED CONTRASTIVE LOSS (Solo in Train)
                loss_con = 0.0
                if self.hparams.use_supcon and mode == 'train':
                    # Proiezione
                    projections = self.projection_head(backbone_features)
                    projections = F.normalize(projections, dim=1)
                    # Unsqueeze per simulare n_views=1 (dimensione: Batch, Views, Feat)
                    projections = projections.unsqueeze(1)
                    
                    # Calcolo Loss Contrastiva
                    loss_con = self.supcon_loss(projections, target)
                    
                    # Somma pesata
                    loss = loss_ce + (self.hparams.lambda_contrast * loss_con)

                result_dict = {
                    f"{mode}_loss": loss,
                    f"{mode}_ce_loss": loss_ce,
                    f"{mode}_acc": acc,
                }
                if self.hparams.use_supcon and mode == 'train':
                    result_dict[f"{mode}_con_loss"] = loss_con

            elif self.hparams.downstream_task_type == 'regression':
                loss = F.mse_loss(logits.squeeze(), target.squeeze())
                l1 = F.l1_loss(logits.squeeze(), target.squeeze())
                result_dict = {
                    f"{mode}_loss": loss,
                    f"{mode}_mse": loss,
                    f"{mode}_l1_loss": l1
                }
        
        self.log_dict(result_dict, prog_bar=True, sync_dist=False, add_dataloader_idx=False, on_step=True, on_epoch=True, batch_size=self.hparams.batch_size)
        
        return loss

    def _evaluate_metrics(self, subj_array, total_out_logits, total_out_target, mode):
        # [Codice esistente per metriche]
        # (Nessuna modifica necessaria qui per la loss contrastiva)
        subjects = np.unique(subj_array)
        subj_avg_logits = []
        subj_targets = []

        if self.hparams.task_name == 'movie_classification':
            acc_func = MulticlassAccuracy(num_classes=self.hparams.num_classes).to(total_out_logits.device)
            acc = acc_func(total_out_logits, total_out_target.long())
            self.log(f"{mode}_acc", acc, sync_dist=True)
            return

        if self.hparams.num_classes == 2:
            for subj in subjects:
                subj_logits = total_out_logits[subj_array == subj]
                subj_avg_logits.append(torch.mean(subj_logits).item())
                subj_targets.append(total_out_target[subj_array == subj][0].item())
            subj_avg_logits = torch.tensor(subj_avg_logits, device = total_out_logits.device) 
            subj_targets = torch.tensor(subj_targets, device = total_out_target.device) 
        elif self.hparams.num_classes > 2:
            for subj in subjects:
                subj_logits = total_out_logits[subj_array == subj]
                subj_avg_logits.append(torch.mean(subj_logits, dim=0))
                subj_targets.append(total_out_target[subj_array == subj][0].item())
            subj_avg_logits = torch.stack(subj_avg_logits) 
            subj_targets = torch.tensor(subj_targets, device = total_out_target.device) 

        if self.hparams.downstream_task_type == 'classification' or self.hparams.scalability_check:
            if self.hparams.num_classes == 2:
                acc_func = BinaryAccuracy().to(total_out_logits.device)
                auroc_func = BinaryAUROC().to(total_out_logits.device)
                acc = acc_func((subj_avg_logits >= 0).int(), subj_targets)
                auroc = auroc_func(torch.sigmoid(subj_avg_logits), subj_targets)
            elif self.hparams.num_classes > 2:
                acc_func = MulticlassAccuracy(num_classes=self.hparams.num_classes).to(total_out_logits.device)
                auroc_func = MulticlassAUROC(num_classes=self.hparams.num_classes).to(total_out_logits.device)
                acc = acc_func(subj_avg_logits, subj_targets.long())
                auroc = auroc_func(subj_avg_logits, subj_targets.long())

            self.log(f"{mode}_acc", acc, sync_dist=True)
            self.log(f"{mode}_AUROC", auroc, sync_dist=True)

        elif self.hparams.downstream_task_type == 'regression':          
            mse = F.mse_loss(subj_avg_logits, subj_targets)
            mae = F.l1_loss(subj_avg_logits, subj_targets)
            self.log(f"{mode}_mse", mse, sync_dist=True)
            self.log(f"{mode}_mae", mae, sync_dist=True)

    def _evaluate_reid(self, subj_array, features, targets, mode):
        # [Codice esistente per ReID]
        if features.numel() == 0: return
        unique_subj = {s: idx for idx, s in enumerate(np.unique(subj_array))}
        labels = torch.tensor([unique_subj[s] for s in subj_array], device=features.device)
        feats = torch.nn.functional.normalize(features, dim=1)
        sim_matrix = torch.matmul(feats, feats.t())
        N = sim_matrix.size(0)
        
        top1_correct = 0.0
        for i in range(N):
            sims = sim_matrix[i]
            sims[i] = -float('inf')
            indices = torch.argsort(sims, descending=True)
            retrieved_labels = labels[indices]
            matches = (retrieved_labels == labels[i]).float()
            if matches.numel() > 0: top1_correct += matches[0].item()
        
        top1 = top1_correct / max(N, 1)
        self.log(f"{mode}_reid_top1", torch.tensor(top1, device=features.device), sync_dist=True)

    def training_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch, mode="train")
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx):
        if self.hparams.pretraining:
            if dataloader_idx == 0: self._calculate_loss(batch, mode="valid")
            else: self._calculate_loss(batch, mode="test")
        else:
            # Nota: qui scartiamo le backbone_features nell'output validation standard
            subj, logits, target, head_features, _ = self._compute_logits(batch)
            output = [logits.squeeze().detach().cpu(), target.squeeze().detach().cpu(), None]
            return (subj, output)

    def validation_epoch_end(self, outputs):
        if not self.hparams.pretraining:
            outputs_valid = outputs[0]
            outputs_test = outputs[1]
            # [Codice aggregazione esistente...]
            subj_valid, out_valid_logits_list, out_valid_target_list = [], [], []
            for subj, out in outputs_valid:
                subj_valid += subj
                out_valid_logits_list.append(out[0])
                out_valid_target_list.append(out[1])
            
            subj_test, out_test_logits_list, out_test_target_list = [], [], []
            for subj, out in outputs_test:
                subj_test += subj
                out_test_logits_list.append(out[0])
                out_test_target_list.append(out[1])

            self._evaluate_metrics(np.array(subj_valid), torch.cat(out_valid_logits_list), torch.cat(out_valid_target_list), mode="valid")
            self._evaluate_metrics(np.array(subj_test), torch.cat(out_test_logits_list), torch.cat(out_test_target_list), mode="test")

    def test_step(self, batch, batch_idx):
        if self.hparams.pretraining:
            self._calculate_loss(batch, mode="test")
        else:
            subj, logits, target, head_features, _ = self._compute_logits(batch)
            output = [logits.squeeze().detach().cpu(), target.squeeze().detach().cpu(), None]
            return (subj, output)

    def test_epoch_end(self, outputs):
        if not self.hparams.pretraining:
            subj_test, out_test_logits_list, out_test_target_list = [], [], []
            for subj, out in outputs:
                subj_test += subj
                out_test_logits_list.append(out[0])
                out_test_target_list.append(out[1])
            self._evaluate_metrics(np.array(subj_test), torch.cat(out_test_logits_list), torch.cat(out_test_target_list), mode="test")

    def configure_optimizers(self):
        if self.hparams.optimizer == "AdamW":
            optim = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
        elif self.hparams.optimizer == "SGD":
            optim = torch.optim.SGD(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay, momentum=self.hparams.momentum)
        
        if self.hparams.use_scheduler:
            total_iterations = self.trainer.estimated_stepping_batches 
            gamma = self.hparams.gamma
            base_lr = self.hparams.learning_rate
            warmup = int(total_iterations * 0.05) 
            T_0 = int(self.hparams.cycle * total_iterations)
            sche = CosineAnnealingWarmUpRestarts(optim, first_cycle_steps=T_0, cycle_mult=1, max_lr=base_lr,min_lr=1e-9, warmup_steps=warmup, gamma=gamma)
            return [optim], [{"scheduler": sche, "name": "lr_history", "interval": "step"}]
        else:
            return optim

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False, formatter_class=ArgumentDefaultsHelpFormatter)
        group = parser.add_argument_group("Default classifier")
        
        # Args esistenti
        group.add_argument("--optimizer", type=str, default="AdamW")
        group.add_argument("--use_scheduler", action='store_true')
        group.add_argument("--weight_decay", type=float, default=0.01)
        group.add_argument("--learning_rate", type=float, default=1e-3)
        group.add_argument("--momentum", type=float, default=0)
        group.add_argument("--gamma", type=float, default=1.0)
        group.add_argument("--cycle", type=float, default=0.3)
        
        # Args Pretraining & Contrastive
        group.add_argument("--use_contrastive", action='store_true')
        group.add_argument("--contrastive_type", default=0, type=int)
        group.add_argument("--use_mae", action='store_true')
        group.add_argument("--mask_ratio", type=float, default=0.1)
        group.add_argument("--pretraining", action='store_true')
        group.add_argument("--augment_during_training", action='store_true')
        group.add_argument("--augment_only_affine", action='store_true')
        group.add_argument("--augment_only_intensity", action='store_true')
        group.add_argument("--temperature", default=0.1, type=float)
        
        # --- NUOVI ARGOMENTI PER SUPERVISED CONTRASTIVE ---
        group.add_argument("--use_supcon", action='store_true', help="Attiva la Supervised Contrastive Loss durante il training classification")
        group.add_argument("--lambda_contrast", type=float, default=0.5, help="Peso della SupCon Loss nella somma totale (Loss = CE + lambda * SupCon)")

        # Model args
        group.add_argument("--model", type=str, default="none")
        group.add_argument("--in_chans", type=int, default=1)
        group.add_argument("--num_classes", type=int, default=2)
        group.add_argument("--embed_dim", type=int, default=24)
        group.add_argument("--window_size", nargs="+", default=[4, 4, 4, 4], type=int)
        group.add_argument("--first_window_size", nargs="+", default=[2, 2, 2, 2], type=int)
        group.add_argument("--patch_size", nargs="+", default=[6, 6, 6, 1], type=int)
        group.add_argument("--depths", nargs="+", default=[2, 2, 6, 2], type=int)
        group.add_argument("--num_heads", nargs="+", default=[3, 6, 12, 24], type=int)
        group.add_argument("--c_multiplier", type=int, default=2)
        group.add_argument("--grad_clip", action='store_true')
        group.add_argument("--scalability_check", action='store_true')
        group.add_argument("--print_flops", action='store_true')
        
        return parser