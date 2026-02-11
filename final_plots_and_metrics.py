import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score, roc_auc_score, f1_score, classification_report
import argparse
import os
import ast

# Impostiamo lo stile dei grafici
sns.set_style("whitegrid")
plt.rcParams.update({'font.size': 12})

def parse_probs(prob_str):
    """Converte stringhe '[0.1, 0.9]' in array numpy."""
    try:
        if isinstance(prob_str, (float, int)):
            return [1-prob_str, prob_str]
        val = ast.literal_eval(prob_str)
        return val
    except:
        return prob_str

def plot_cm(y_true, y_pred, labels, output_path, title, normalize=None):
    """Genera e salva la Confusion Matrix (Assoluta o Percentuale)."""
    cm = confusion_matrix(y_true, y_pred, normalize=normalize)
    
    plt.figure(figsize=(7, 6))
    fmt = '.1%' if normalize else 'd'
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', 
                xticklabels=labels, yticklabels=labels, cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_name", type=str, required=True)
    parser.add_argument("--csv_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    # Percorsi
    if args.output_dir is None:
        if "neurostorm" in args.project_name: category = "neurostorm" 
        else: category = "other"
        output_dir = os.path.join("output", category, args.project_name)
    else:
        output_dir = args.output_dir

    file_path = os.path.join(output_dir, args.csv_file)
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return

    print(f"📊 Generazione Report Avanzato: {args.project_name}")
    df = pd.read_csv(file_path)

    # --- Pre-processing ---
    first_prob = df['prob'].iloc[0]
    is_multiclass = False
    
    if isinstance(first_prob, str) and '[' in first_prob:
        df['prob'] = df['prob'].apply(parse_probs)
        if len(df['prob'].iloc[0]) > 2:
            is_multiclass = True
            num_classes = len(df['prob'].iloc[0])
            class_labels = [f"Class {i}" for i in range(num_classes)] # Placeholder
        else:
            df['prob'] = df['prob'].apply(lambda x: x[1])
            class_labels = ["Control", "Case"]
    elif isinstance(first_prob, (float, int)):
        class_labels = ["Control", "Case"]

    report_file = os.path.join(output_dir, "final_metrics_report_advanced.txt")
    
    with open(report_file, "w") as f:
        f.write(f"🔬 DETAILED ANALYSIS REPORT: {args.project_name}\n")
        f.write("==================================================\n\n")

        # ======================================================================
        # 1. ANALISI BINARIA (Clip-Level & Subject-Level)
        # ======================================================================
        if not is_multiclass:
            y_true = df['true'].values
            y_prob = df['prob'].values

            # --- Calcolo Soglia Ottimale ---
            fpr, tpr, thresholds = roc_curve(y_true, y_prob)
            roc_auc = auc(fpr, tpr)
            J = tpr - fpr
            best_thresh = thresholds[np.argmax(J)]
            
            # --- Metriche: Default (0.5) vs Optimized ---
            preds_05 = (y_prob >= 0.5).astype(int)
            preds_opt = (y_prob >= best_thresh).astype(int)

            acc_05 = accuracy_score(y_true, preds_05)
            acc_opt = accuracy_score(y_true, preds_opt)
            
            # Scrittura Report Clip-Level
            f.write(f"🎬 CLIP-LEVEL METRICS (Single Windows)\n")
            f.write(f"   Total Samples: {len(df)}\n")
            f.write(f"   Global AUC:    {roc_auc:.4f}\n")
            f.write(f"   ----------------------------------\n")
            f.write(f"   THRESHOLD 0.5 (Default):\n")
            f.write(f"     Accuracy: {acc_05:.4f}\n")
            f.write(f"     F1 Score: {f1_score(y_true, preds_05):.4f}\n")
            f.write(f"   ----------------------------------\n")
            f.write(f"   THRESHOLD {best_thresh:.4f} (Optimized):\n")
            f.write(f"     Accuracy: {acc_opt:.4f}  <-- BEST\n")
            f.write(f"     F1 Score: {f1_score(y_true, preds_opt):.4f}\n\n")

            # --- Grafici Clip-Level ---
            # ROC
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve (Clip Level)')
            plt.legend()
            plt.savefig(os.path.join(output_dir, "roc_curve_clip.png"))
            plt.close()

            # Confusion Matrix (Standard + Normalized)
            plot_cm(y_true, preds_opt, class_labels, 
                    os.path.join(output_dir, "cm_clip_counts.png"), 
                    f"Confusion Matrix (Clip, Thresh {best_thresh:.2f})")
            
            plot_cm(y_true, preds_opt, class_labels, 
                    os.path.join(output_dir, "cm_clip_normalized.png"), 
                    f"Confusion Matrix % (Clip, Thresh {best_thresh:.2f})", normalize='true')


            # ==================================================================
            # 2. ANALISI PER SOGGETTO (Subject-Level Aggregation)
            # ==================================================================
            f.write(f"👤 SUBJECT-LEVEL METRICS (Aggregated)\n")
            f.write(f"   (Media delle probabilità di tutte le clip di un soggetto)\n")
            
            # Raggruppa per soggetto e calcola media probabilità
            subj_df = df.groupby(['subject', 'true', 'fold']).agg({'prob': 'mean'}).reset_index()
            
            y_sub_true = subj_df['true'].values
            y_sub_prob = subj_df['prob'].values
            
            # Soglia ottimale anche per soggetti
            fpr_s, tpr_s, thresh_s = roc_curve(y_sub_true, y_sub_prob)
            auc_sub = auc(fpr_s, tpr_s)
            best_thresh_sub = thresh_s[np.argmax(tpr_s - fpr_s)]
            
            y_sub_pred = (y_sub_prob >= best_thresh_sub).astype(int)
            acc_sub = accuracy_score(y_sub_true, y_sub_pred)
            
            f.write(f"   Total Subjects: {len(subj_df)}\n")
            f.write(f"   Subject AUC:    {auc_sub:.4f}\n")
            f.write(f"   Subject ACC:    {acc_sub:.4f} (at thresh {best_thresh_sub:.2f})\n\n")
            
            # Confusion Matrix Soggetti
            plot_cm(y_sub_true, y_sub_pred, class_labels, 
                    os.path.join(output_dir, "cm_subject_normalized.png"), 
                    f"Confusion Matrix % (Subject Level)", normalize='true')
            
            # Salva CSV Subject Level per ispezione
            subj_df['pred_class'] = y_sub_pred
            subj_df.to_csv(os.path.join(output_dir, "subject_level_results.csv"), index=False)
            f.write(f"   -> Dettagli per soggetto salvati in 'subject_level_results.csv'\n\n")

            # ==================================================================
            # 3. ANALISI PER TASK (Se disponibile)
            # ==================================================================
            # Nota: Al momento il CSV non ha la colonna 'task' o 'event'. 
            # Se la aggiungiamo in futuro, questo codice funzionerà.
            if 'event' in df.columns:
                f.write(f"🧩 TASK-SPECIFIC ANALYSIS\n")
                for task in df['event'].unique():
                    task_df = df[df['event'] == task]
                    if len(task_df) > 0:
                        t_acc = accuracy_score(task_df['true'], (task_df['prob'] >= best_thresh).astype(int))
                        f.write(f"   Task '{task}': Accuracy = {t_acc:.4f} (n={len(task_df)})\n")
            else:
                f.write(f"🧩 TASK ANALYSIS: Not available (missing 'event' column in CSV).\n")

        # ======================================================================
        # 4. ANALISI MULTICLASSE
        # ======================================================================
        else:
            y_true = df['true'].values
            y_prob = np.array(df['prob'].tolist())
            y_pred = np.argmax(y_prob, axis=1)
            
            acc = accuracy_score(y_true, y_pred)
            
            f.write(f"🚦 MULTICLASS METRICS\n")
            f.write(f"   Accuracy: {acc:.4f}\n")
            f.write(classification_report(y_true, y_pred) + "\n")
            
            # Confusion Matrix %
            plot_cm(y_true, y_pred, range(num_classes), 
                    os.path.join(output_dir, "cm_multiclass_normalized.png"), 
                    f"Multiclass CM %", normalize='true')

            # Per soggetto
            subj_df = df.groupby(['subject', 'true', 'fold']).agg({'prob': lambda x: np.mean(np.vstack(x), axis=0).tolist()}).reset_index()
            y_s_true = subj_df['true'].values
            y_s_prob = np.array(subj_df['prob'].tolist())
            y_s_pred = np.argmax(y_s_prob, axis=1)
            acc_s = accuracy_score(y_s_true, y_s_pred)
            
            f.write(f"\n👤 SUBJECT-LEVEL MULTICLASS\n")
            f.write(f"   Accuracy: {acc_s:.4f}\n")

    print(f"✅ Report completo generato in: {output_dir}")

if __name__ == "__main__":
    main()