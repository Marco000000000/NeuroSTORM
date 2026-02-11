import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, roc_curve, auc, confusion_matrix, 
    accuracy_score, balanced_accuracy_score, 
    classification_report, recall_score
)
from argparse import ArgumentParser

def plot_roc_curve(y_true, y_probs, auc_score, title, save_path):
    """Genera una curva ROC professionale per il paper"""
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(alpha=0.3)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"   📈 Curva ROC salvata: {save_path}")

def plot_normalized_cm(y_true, y_pred, title, save_path, labels=['Control', 'Schizo']):
    """Genera matrice di confusione normalizzata (%)"""
    cm = confusion_matrix(y_true, y_pred, normalize='true') # Normalizza sulle righe (True)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='.1%', cmap='Blues', 
                xticklabels=labels, yticklabels=labels, vmin=0, vmax=1)
    plt.title(title, fontsize=14)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"   🖼️  Matrice Normalizzata salvata: {save_path}")

def print_advanced_metrics(y_true, y_pred, y_prob, level_name):
    """Stampa metriche cliniche avanzate"""
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    
    # Specificity = Recall della classe 0
    # Sensitivity = Recall della classe 1
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)
    
    auc_score = roc_auc_score(y_true, y_prob)
    
    print(f"\n📊 {level_name} METRICS")
    print("-" * 30)
    print(f"   • AUC:               {auc_score:.4f}")
    print(f"   • Accuracy:          {acc:.4f}")
    print(f"   • Balanced Accuracy: {bal_acc:.4f}  <-- IL NUMERO CHE CERCAVI")
    print(f"   • Sensitivity (Recall Schizo): {sensitivity:.4f}")
    print(f"   • Specificity (Recall Ctrl):   {specificity:.4f}")
    
    return auc_score

def main():
    parser = ArgumentParser()
    parser.add_argument("--project_name", type=str, required=True)
    parser.add_argument("--csv_file", type=str, default="results_progressive.csv", 
                        help="Nome del file CSV generato dallo script precedente")
    args = parser.parse_args()

    # Setup percorsi
    # Cerca in neurostorm o other
    base_dir = os.path.join('output', 'neurostorm', args.project_name)
    if not os.path.exists(base_dir):
        base_dir = os.path.join('output', 'other', args.project_name)
    
    csv_path = os.path.join(base_dir, args.csv_file)
    if not os.path.exists(csv_path):
        # Prova a cercare il file merged se l'altro non c'è
        csv_path = os.path.join(base_dir, "final_results_merged.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"❌ Impossibile trovare il file dei risultati in {base_dir}")

    print(f"📂 Analisi File: {csv_path}")
    df = pd.read_csv(csv_path)

    # ==========================================
    # 1. EVENT LEVEL ANALYSIS
    # ==========================================
    # Ricostruiamo le predizioni usando la soglia salvata nel CSV
    df['pred'] = df.apply(lambda row: 1 if row['prob'] >= row['thresh'] else 0, axis=1)
    
    auc_event = print_advanced_metrics(df['true'], df['pred'], df['prob'], "EVENT-LEVEL")
    
    # Plot ROC Eventi
    plot_roc_curve(df['true'], df['prob'], auc_event, 
                   "Event-Level ROC Curve", 
                   os.path.join(base_dir, "Paper_ROC_Event.png"))
    
    # Plot CM Eventi Normalizzata
    plot_normalized_cm(df['true'], df['pred'], 
                       "Event-Level Confusion Matrix (Normalized)", 
                       os.path.join(base_dir, "Paper_CM_Event_Norm.png"))

    # ==========================================
    # 2. SUBJECT LEVEL ANALYSIS
    # ==========================================
    if 'unknown' not in df['subject'].values:
        # Aggregazione per soggetto
        subj_df = df.groupby('subject').agg({
            'true': 'first', 
            'prob': 'mean', 
            'thresh': 'mean' # Usiamo la media delle soglie dei fold
        }).reset_index()
        
        subj_df['pred'] = subj_df.apply(lambda row: 1 if row['prob'] >= row['thresh'] else 0, axis=1)
        
        auc_subj = print_advanced_metrics(subj_df['true'], subj_df['pred'], subj_df['prob'], "SUBJECT-LEVEL")
        
        # Plot ROC Soggetti
        plot_roc_curve(subj_df['true'], subj_df['prob'], auc_subj, 
                       "Subject-Level ROC Curve", 
                       os.path.join(base_dir, "Paper_ROC_Subject.png"))
        
        # Plot CM Soggetti Normalizzata
        plot_normalized_cm(subj_df['true'], subj_df['pred'], 
                           "Subject-Level Confusion Matrix (Normalized)", 
                           os.path.join(base_dir, "Paper_CM_Subject_Norm.png"))
        
        # Salvataggio metriche su file
        with open(os.path.join(base_dir, "FINAL_PAPER_METRICS.txt"), "w") as f:
            f.write("METRICHE FINALI PER IL PAPER\n")
            f.write("============================\n\n")
            f.write(f"SUBJECT LEVEL:\n")
            f.write(f"AUC: {auc_subj:.4f}\n")
            f.write(f"Balanced Accuracy: {balanced_accuracy_score(subj_df['true'], subj_df['pred']):.4f}\n")
            f.write(f"Standard Accuracy: {accuracy_score(subj_df['true'], subj_df['pred']):.4f}\n")
            
            tn, fp, fn, tp = confusion_matrix(subj_df['true'], subj_df['pred']).ravel()
            f.write(f"Sensitivity (Recall Schizo): {tp / (tp + fn):.4f}\n")
            f.write(f"Specificity (Recall Control): {tn / (tn + fp):.4f}\n")

    else:
        print("⚠️ Analisi per soggetto saltata (ID non trovati).")

if __name__ == "__main__":
    main()