import os
import torch
import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from collections import OrderedDict
import pytorch_lightning as pl
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from huggingface_hub import hf_hub_download

# --- IMPORTS MICCAI ---
from utils.cv_data_module_miccai import fMRIDataModuleMICCAI
from models.lightning_model_miccai import LightningModelMICCAI
from utils.parser import str2bool
import functools
print = functools.partial(print, flush=True)

def cli_main():
    parser = ArgumentParser(add_help=False, formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--seed", default=1234, type=int)
    parser.add_argument("--dataset_name", type=str, default="UCLA")
    parser.add_argument("--downstream_task_id", type=int, default="1")
    parser.add_argument("--downstream_task_type", type=str, default="classification")
    parser.add_argument("--task_name", type=str, default="sex")
    parser.add_argument("--loggername", default="tensorboard", type=str)
    parser.add_argument("--project_name", default="default", type=str)
    parser.add_argument("--auto_resume", action='store_true')
    parser.add_argument("--resume_ckpt_path", type=str)
    parser.add_argument("--load_model_path", type=str)
    parser.add_argument("--test_only", action='store_true')
    parser.add_argument("--test_ckpt_path", type=str)
    parser.add_argument("--freeze_feature_extractor", action='store_true')
    # parser.add_argument("--print_flops", action='store_true')
    parser.add_argument("--num_folds", type=int, default=5)

    # Usa le classi MICCAI
    Dataset = fMRIDataModuleMICCAI
    parser = LightningModelMICCAI.add_model_specific_args(parser)
    parser = Dataset.add_data_specific_args(parser)

    _, _ = parser.parse_known_args()
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    
    if args.model == "neurostorm": category_dir = "neurostorm"
    else: category_dir = "other"
    setattr(args, "default_root_dir", os.path.join('output', category_dir, args.project_name))
    os.makedirs(args.default_root_dir, exist_ok=True)
    pl.seed_everything(args.seed)

    print(f"\n🚀 STARTING {args.num_folds}-FOLD CV (MICCAI PIPELINE)")
    
    fold_results = []
    for fold_idx in range(args.num_folds):
        print(f"\n⚡ FOLD {fold_idx+1}/{args.num_folds}")
        
        data_module = fMRIDataModuleMICCAI(use_cv=True, fold_index=fold_idx, **vars(args))
        
        fold_version = f"fold_{fold_idx}"
        logger = TensorBoardLogger(args.default_root_dir, name="cv_logs", version=fold_version)

        monitor_metric = "valid_AUROC" if args.downstream_task_type == "classification" else "valid_mse"
        mode_metric = "max" if args.downstream_task_type == "classification" else "min"
        
        checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.join(args.default_root_dir, fold_version, "checkpoints"),
            monitor=monitor_metric, filename="best_fold", save_top_k=1, mode=mode_metric,
        )
        early_stop_callback = EarlyStopping(
            monitor=monitor_metric, patience=5, verbose=True, mode=mode_metric
        )
        callbacks = [checkpoint_callback, early_stop_callback, LearningRateMonitor("step")]

        trainer = pl.Trainer.from_argparse_args(
            args, logger=logger, callbacks=callbacks,
            gradient_clip_val=0.5 if args.grad_clip else None,
            check_val_every_n_epoch=1, num_sanity_val_steps=0
        )

        model = LightningModelMICCAI(data_module=data_module, **vars(args))

        if args.load_model_path and os.path.exists(args.load_model_path):
            print(f'   📥 Loading pretrained: {args.load_model_path}')
            ckpt = torch.load(args.load_model_path)
            new_state_dict = OrderedDict()
            for k, v in ckpt['state_dict'].items():
                if 'model.' in k: new_state_dict[k.removeprefix("model.")] = v
            model.model.load_state_dict(new_state_dict, strict=False)

        if args.freeze_feature_extractor:
            for name, param in model.model.named_parameters(): param.requires_grad = False
            if hasattr(model.model, 'norm') and model.model.norm:
                for param in model.model.norm.parameters(): param.requires_grad = True

        if args.test_only:
            trainer.test(model, datamodule=data_module, ckpt_path=args.test_ckpt_path)
        else:
            trainer.fit(model, datamodule=data_module)
            res = trainer.test(model, datamodule=data_module, ckpt_path="best")
            metric_key = f"test_AUROC" if args.downstream_task_type == "classification" else "test_mse"
            score = res[0].get(metric_key, 0.0)
            fold_results.append(score)
            print(f"   🏁 Fold {fold_idx+1} Result ({metric_key}): {score:.4f}")

    if not args.test_only:
        with open(os.path.join(args.default_root_dir, "cv_results_summary.txt"), "w") as f:
            f.write(f"Mean: {np.mean(fold_results):.4f}\nStd: {np.std(fold_results):.4f}\nScores: {fold_results}")

if __name__ == "__main__":
    cli_main()