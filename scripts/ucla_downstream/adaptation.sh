#!/bin/bash
# bash scripts/ucla_downstream/pt_neurostorm_ucla_mae.sh

# Usa tutti i dati, batch size più grande possibile per MAE
batch_size="4" 

export CUDA_VISIBLE_DEVICES=0
export NCCL_P2P_DISABLE=1

# Nome progetto: Adattamento al dominio UCLA
project_name="ucla_mae_domain_adaptation"

python main.py \
  --accelerator gpu \
  --max_epochs 50 \
  --num_nodes 1 \
  --strategy ddp \
  --loggername tensorboard \
  --dataset_name UCLA \
  --image_path ./data/UCLA_MNI_to_TRs_minmax \
  --batch_size "$batch_size" \
  --num_workers 8 \
  --project_name "$project_name" \
  --limit_training_samples 1.0 \
  --c_multiplier 2 \
  --last_layer_full_MSA False \
  --pretraining \
  --use_mae \
  --mask_ratio 0.50 \
  --spatial_mask window \
  --task_name "diagnosis" \
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
  --load_model_path ./output/neurostorm/pt_neurostorm_mae_ratio0.5.ckpt