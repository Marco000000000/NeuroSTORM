#!/bin/bash
#SBATCH --job-name=ucla_mae_adapt
#SBATCH --partition=gpu
#SBATCH --qos=train
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

# Carica le variabili d'ambiente necessarie (se serve)
export NCCL_P2P_DISABLE=1

# Lancia il comando usando l'ambiente 'neurostorm'
# Nota: Ho incluso --task_name "diagnosis" per evitare il crash
conda run -n neurostorm python main.py \
  --accelerator gpu \
  --max_epochs 50 \
  --num_nodes 1 \
  --strategy ddp \
  --loggername tensorboard \
  --dataset_name UCLA \
  --image_path ./data/UCLA_MNI_to_TRs_minmax \
  --batch_size 4 \
  --num_workers 8 \
  --project_name "ucla_mae_domain_adaptation" \
  --limit_training_samples 1.0 \
  --c_multiplier 2 \
  --last_layer_full_MSA False \
  --pretraining \
  --use_mae \
  --mask_ratio 0.50 \
  --spatial_mask window \
  --time_mask random \
  --dataset_split_num 1 \
  --seed 1 \
  --learning_rate 1e-4 \
  --model neurostorm \
  --depth 2 2 6 2 \
  --embed_dim 36 \
  --sequence_length 20 \
  --img_size 96 96 96 20 \
  --first_window_size 4 4 4 4 \
  --window_size 4 4 4 4 \
  --load_model_path ./output/neurostorm/pt_neurostorm_mae_ratio0.5.ckpt \
  --task_name "diagnosis"