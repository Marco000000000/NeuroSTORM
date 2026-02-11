import torch
from utils.data_module import BaseDataModule
from argparse import Namespace

# Configurazione minima per caricare il modulo
args = Namespace(
    dataset_name='UCLA',
    batch_size=4,
    num_workers=0,
    image_path='./data/UCLA_MNI_to_TRs_minmax', # Controlla che il path sia giusto
    dataset_split_num=1,
    seed=1,
    sequence_length=20,
    img_size=[96, 96, 96, 20],
    limit_training_samples=1.0,
    downstream_task_type='classification',
    num_classes=4,
    task_name='diagnosis', # Fondamentale
    sample_duration=20,    # O quello che usi
    target="diagnosis"
)

def check_leakage():
    print("🕵️‍♂️ ISPEZIONE SPLIT DATI...")
    
    # Inizializza DataModule
    dm = BaseDataModule(**vars(args))
    dm.setup()
    
    # Estrai i soggetti
    # Nota: dipendentemente da come è fatto il dataset, potrebbe essere .subjects o .data
    # Accediamo direttamente al dataset interno
    
    train_subs = set(dm.train_dataset.data_frame['subject'] if hasattr(dm.train_dataset, 'data_frame') else [x[1] for x in dm.train_dataset.data])
    val_subs = set([x[1] for x in dm.val_dataset.data])
    test_subs = set([x[1] for x in dm.test_dataset.data])

    print(f"\nNUMERO SOGGETTI:")
    print(f"👉 Train: {len(train_subs)}")
    print(f"👉 Val:   {len(val_subs)}")
    print(f"👉 Test:  {len(test_subs)}")

    print("\nCONTROLLO SOVRAPPOSIZIONI (LEAKAGE):")
    
    # 1. Val == Test? (Il tuo sospetto)
    intersection_val_test = val_subs.intersection(test_subs)
    if len(intersection_val_test) == len(val_subs) and len(val_subs) > 0:
        print("🚨 CRITICO: Validation e Test SONO IDENTICI! (Stessi soggetti)")
        print(f"   Soggetti: {list(intersection_val_test)[:5]} ...")
    elif len(intersection_val_test) > 0:
        print(f"⚠️ ATTENZIONE: Ci sono {len(intersection_val_test)} soggetti condivisi tra Val e Test!")
    else:
        print("✅ Val e Test sono disgiunti (Bravi!)")

    # 2. Train vs Val/Test
    leak_train_val = train_subs.intersection(val_subs)
    if len(leak_train_val) > 0:
        print(f"❌ ERRORE: {len(leak_train_val)} soggetti del Train sono anche nel Val!")
        
    leak_train_test = train_subs.intersection(test_subs)
    if len(leak_train_test) > 0:
        print(f"❌ ERRORE: {len(leak_train_test)} soggetti del Train sono anche nel Test!")

if __name__ == "__main__":
    check_leakage()