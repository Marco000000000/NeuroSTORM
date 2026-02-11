import os
import torch
import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from collections import OrderedDict
import pytorch_lightning as pl
from pytorch_lightning.loggers.neptune import NeptuneLogger
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
import neptune
# Ensure utils.data_module has the updated fMRIDataModule with setup_kfold
from utils.cv_data_module import fMRIDataModule 
from utils.parser import str2bool
from models.lightning_model import LightningModel
from huggingface_hub import hf_hub_download

# Metrics for final report
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix

def cli_main():

    # ==========================================
    # 1. ARGUMENT PARSING (From main.py)
    # ==========================================
    parser = ArgumentParser(add_help=False, formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--seed", default=1234, type=int, help="random seeds.")
    parser.add_argument("--dataset_name", type=str, choices=["HCP1200", "ABCD", "UKB", "Cobre", "ADHD200", "HCPA", "HCPD", "UCLA", "HCPEP", "HCPTASK", "GOD", "NSD", "BOLD5000", "MOVIE", "TransDiag"], default="HCP1200")
    parser.add_argument("--downstream_task_id", type=int, default="1", help="downstream task id")
    parser.add_argument("--downstream_task_type", type=str, default="classification", help="select either classification or regression")
    parser.add_argument("--task_name", type=str, default="sex", help="specify the task name")
    parser.add_argument("--loggername", default="tensorboard", type=str, help="A name of logger")
    parser.add_argument("--project_name", default="default", type=str, help="A name of project")
    
    # Checkpoint / Resume args
    parser.add_argument("--auto_resume", action='store_true', help="Whether to find the last checkpoint and resume")
    parser.add_argument("--resume_ckpt_path", type=str, help="A path to previous checkpoint")
    parser.add_argument("--load_model_path", type=str, help="A path to the pre-trained model weight file")
    parser.add_argument("--test_only", action='store_true', help="specify when you want to test")
    parser.add_argument("--test_ckpt_path", type=str, help="A path to the previous checkpoint")
    parser.add_argument("--freeze_feature_extractor", action='store_true', help="Whether to freeze the feature extractor")
    parser.add_argument("--print_flops", action='store_true', help="Whether to print FLOPs")

    # --- CV SPECIFIC ARGUMENT ---
    parser.add_argument("--num_folds", type=int, default=5, help="Number of Cross Validation folds")

    # Set dataset class
    Dataset = fMRIDataModule

    # Add specific args
    parser = LightningModel.add_model_specific_args(parser)
    parser = Dataset.add_data_specific_args(parser)

    _, _ = parser.parse_known_args()
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    
    # Path setup
    if args.model == "neurostorm": category_dir = "neurostorm"
    elif args.model in ["swift", "tff"]: category_dir = "volume-based"
    elif args.model in ["braingnn", "bnt"]: category_dir = "roi-based"
    else: category_dir = "other"

    setattr(args, "default_root_dir", os.path.join('output', category_dir, args.project_name))
    os.makedirs(args.default_root_dir, exist_ok=True)

    # Seed (Critical for deterministic splits across folds/GPUs)
    pl.seed_everything(args.seed)

    # ==========================================
    # 2. CROSS VALIDATION LOOP
    # ==========================================
    print("\n" + "="*50)
    print(f"🚀 STARTING {args.num_folds}-FOLD CROSS VALIDATION")
    print(f"   Task: {args.task_name} | Dataset: {args.dataset_name}")
    print("="*50)

    fold_results = []
    
    # Loop over folds
    for fold_idx in range(args.num_folds):
        print(f"\n⚡ PROCESSING FOLD {fold_idx+1}/{args.num_folds}")
        
        # --------------------------------------
        # A. DATA MODULE (The Magic Part)
        # --------------------------------------
        # We pass use_cv=True and the current fold_index.
        # The DataModule handles the splitting logic internally.
        data_module = fMRIDataModule(
            use_cv=True,
            fold_index=fold_idx,
            **vars(args)
        )
        
        # --------------------------------------
        # B. LOGGER (Separate per fold)
        # --------------------------------------
        fold_version = f"fold_{fold_idx}"
        if args.loggername == "tensorboard":
            logger = TensorBoardLogger(args.default_root_dir, name="cv_logs", version=fold_version)
        elif args.loggername == "neptune":
            # For CV, usually cleaner to disable Neptune or use separate runs. 
            # Here we default to TensorBoard structure or None to avoid chaos.
            logger = None 
        else:
            raise Exception("Wrong logger name.")

       # --- D. Callbacks ---
        monitor_metric = "valid_AUROC" if args.downstream_task_type == "classification" else "valid_mse"
        mode_metric = "max" if args.downstream_task_type == "classification" else "min"
        
        # 1. Salva il modello migliore
        checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.join(args.default_root_dir, fold_version, "checkpoints"),
            monitor=monitor_metric,
            filename="best_fold",
            save_top_k=1,
            mode=mode_metric,
        )
        
        # 2. Interrompi se non migliora (NUOVO)
        early_stop_callback = EarlyStopping(
            monitor=monitor_metric,
            patience=5,      # Aspetta 5 epoche senza miglioramenti prima di stoppare
            verbose=True,
            mode=mode_metric
        )

        lr_monitor = LearningRateMonitor(logging_interval="step")
        
        # Aggiungi alla lista
        callbacks = [checkpoint_callback, early_stop_callback, lr_monitor]

        # --------------------------------------
        # D. TRAINER
        # --------------------------------------
        if args.grad_clip:
            print('using gradient clipping')
            trainer = pl.Trainer.from_argparse_args(
                args, logger=logger, callbacks=callbacks,
                gradient_clip_val=0.5, gradient_clip_algorithm="norm", track_grad_norm=-1,
                check_val_every_n_epoch=1, num_sanity_val_steps=0
            )
        else:
            print('not using gradient clipping')
            trainer = pl.Trainer.from_argparse_args(
                args, logger=logger, callbacks=callbacks,
                check_val_every_n_epoch=1, num_sanity_val_steps=0
            )

        # --------------------------------------
        # E. MODEL (Re-init every fold)
        # --------------------------------------
        model = LightningModel(data_module=data_module, **vars(args))

        # Load Pretrained Weights (Logic from main.py)
        path = None
        if args.load_model_path is not None:
            if os.path.exists(args.load_model_path):
                print(f'   📥 Loading model from {args.load_model_path}')
                path = args.load_model_path
            else:
                print('   ⚠️ Cannot find ckpt file. Trying huggingface...')
                repo_id = "zxcvb20001/fMRI-GPT"
                if args.model == 'neurostorm':
                    filename = "neurostorm/{}".format(os.path.basename(args.load_model_path))
                elif args.model in ['swift']:
                    filename = "volume-based/{}/{}".format(args.model, os.path.basename(args.load_model_path))
                
                try:
                    path = hf_hub_download(repo_id=repo_id, filename=filename)
                except:
                    print('   ❌ Download failed. Training from scratch.')
        
        if path is not None:
            ckpt = torch.load(path)
            new_state_dict = OrderedDict()
            for k, v in ckpt['state_dict'].items():
                if 'model.' in k:
                    new_state_dict[k.removeprefix("model.")] = v
            model.model.load_state_dict(new_state_dict, strict=False)

        # Partial Freezing Logic (Main.py)
        if args.freeze_feature_extractor:
            print("\n   ❄️  PARTIAL FREEZING ACTIVE")
            print("      🔒 Freezing entire encoder...")
            for name, param in model.model.named_parameters():
                param.requires_grad = False
            
            if hasattr(model.model, 'norm') and model.model.norm is not None:
                print("      🔓 Unfreezing Final Normalization Only")
                for param in model.model.norm.parameters():
                    param.requires_grad = True

        # --------------------------------------
        # F. RUN FOLD
        # --------------------------------------
        if args.test_only:
            # Note: In CV mode, test_dataloader returns the Fold Validation set
            trainer.test(model, datamodule=data_module, ckpt_path=args.test_ckpt_path)
        else:
            # Train
            trainer.fit(model, datamodule=data_module)
            
            # Test Best Model of this fold
            print(f"   ✅ Fold {fold_idx+1} Finished. Evaluating Best Model...")
            # 'best' automatically loads the best checkpoint from ModelCheckpoint
            res = trainer.test(model, datamodule=data_module, ckpt_path="best")
            
            # Collect Metric
            metric_key = f"test_AUROC" if args.downstream_task_type == "classification" else "test_mse"
            # Lightning returns a list of dicts
            score = res[0].get(metric_key, 0.0)
            fold_results.append(score)
            print(f"   🏁 Fold {fold_idx+1} Result ({metric_key}): {score:.4f}")

    # ==========================================
    # 3. FINAL REPORT
    # ==========================================
    if not args.test_only:
        mean_score = np.mean(fold_results)
        std_score = np.std(fold_results)
        
        print("\n" + "🏆"*20)
        print(f" {args.num_folds}-FOLD CV COMPLETED")
        print("🏆"*20)
        print(f"Individual Scores: {fold_results}")
        print(f"MEAN SCORE: {mean_score:.4f} ± {std_score:.4f}")
        
        # Save summary to file
        with open(os.path.join(args.default_root_dir, "cv_results_summary.txt"), "w") as f:
            f.write(f"Mean: {mean_score:.4f}\nStd: {std_score:.4f}\nScores: {fold_results}")

if __name__ == "__main__":
    cli_main()