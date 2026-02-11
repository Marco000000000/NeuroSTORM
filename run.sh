#!/bin/bash

# Parametri comuni
PROJECT="ucla_ft_neurostorm_task3_dx_train1.0_cv"
DATA_PATH="./data/UCLA_MNI_to_TRs_minmax"

# Loop per i 5 fold
for i in {0..4}
do
   echo "--------------------------------------"
   echo "🚀 AVVIO FOLD $i (Processo Isolato)"
   echo "--------------------------------------"
   
   python confusion_matrix_sb.py \
      --project_name "$PROJECT" \
      --image_path "$DATA_PATH" \
      --dataset_name UCLA \
      --num_classes 2 \
      --dataset_split_num 2 \
      --train_split 0.8 \
      --val_split 0.1 \
      --batch_size 4 \
      --num_workers 0 \
      --model neurostorm \
      --task_name diagnosis \
      --seed 1 \
      --single_fold_index $i  # <--- NUOVO ARGOMENTO (vedi sotto)
      
   # Se c'è un errore, fermati
   if [ $? -ne 0 ]; then
       echo "❌ Errore nel Fold $i. Stop."
       exit 1
   fi
done

echo "✅ Tutti i fold completati!"