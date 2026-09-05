import os
import glob
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import torch.optim as optim
import re


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


BATCH_SIZE = 1
EPOCHS = 100
LEARNING_RATE = 3e-3
EARLY_STOP_PATIENCE = 20
DROPOUT = 0.2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


CONDITION_NAME = "Baseline_Classical_1D_CNN"


ROI_FOLDER_NAME = "ROIs"

MAIN_RESULTS_FOLDER = os.path.join(SCRIPT_DIR, "Results")
CONDITION_FOLDER = os.path.join(MAIN_RESULTS_FOLDER, CONDITION_NAME)
PLOTS_SUBFOLDER = os.path.join(CONDITION_FOLDER, "plots")


ROI_FOLDER = os.path.join(SCRIPT_DIR, ROI_FOLDER_NAME)

print(f"Script directory: {SCRIPT_DIR}")
print(f"ROI folder: {ROI_FOLDER}")
print(f"Main results folder: {MAIN_RESULTS_FOLDER}")
print(f"Condition folder: {CONDITION_FOLDER}")
print(f"Plots folder: {PLOTS_SUBFOLDER}")


os.makedirs(PLOTS_SUBFOLDER, exist_ok=True)
print(f"✓ Folders created successfully!")


class EarlyStopping:
    def __init__(self, patience=EARLY_STOP_PATIENCE, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score):
        if self.best_score is None or score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


class Classical1DCNNModel(nn.Module):
    def __init__(self, input_channels=1, output_dim=2):
        super().__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Sequential(
            nn.Linear(128, 96),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(96, 48),
            nn.ReLU(),
            nn.Linear(48, output_dim)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.global_pool(x)          
        x = x.squeeze(-1)                
        logits = self.classifier(x)
        return logits


def get_confusion_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = fp = fn = tp = None
    return tn, fp, fn, tp


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y.long())
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == y).sum().item()
        total += X.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())
    bal_acc = balanced_accuracy_score(all_labels, all_preds)
    tn, fp, fn, tp = get_confusion_metrics(all_labels, all_preds)
    avg_loss = total_loss / total
    return avg_loss, correct/total, bal_acc, tn, fp, fn, tp


def validate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            loss = criterion(logits, y.long())
            total_loss += loss.item() * X.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y).sum().item()
            total += X.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    bal_acc = balanced_accuracy_score(all_labels, all_preds)
    tn, fp, fn, tp = get_confusion_metrics(all_labels, all_preds)
    avg_loss = total_loss / total
    return avg_loss, correct/total, bal_acc, tn, fp, fn, tp


def plot_training_curves(train_losses, test_losses, train_accuracies, test_accuracies,
                        train_bal_accuracies, test_bal_accuracies, roi_number, fold, plots_folder):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    ax1.plot(train_losses, label='Train Loss', color='blue', linewidth=2)
    ax1.plot(test_losses, label='Test Loss', color='red', linewidth=2)
    ax1.set_title(f'ROI {roi_number:02d} - Fold {fold}: Loss Curves', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(train_accuracies, label='Train Accuracy', color='blue', linewidth=2)
    ax2.plot(test_accuracies, label='Test Accuracy', color='red', linewidth=2)
    ax2.set_title(f'ROI {roi_number:02d} - Fold {fold}: Accuracy Curves', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax3.plot(train_bal_accuracies, label='Train Balanced Accuracy', color='blue', linewidth=2)
    ax3.plot(test_bal_accuracies, label='Test Balanced Accuracy', color='red', linewidth=2)
    ax3.set_title(f'ROI {roi_number:02d} - Fold {fold}: Balanced Accuracy Curves', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Balanced Accuracy')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    epochs = range(1, len(train_losses) + 1)
    ax4.plot(epochs, train_losses, label='Train Loss', color='blue', linestyle='-', linewidth=2)
    ax4.plot(epochs, test_losses, label='Test Loss', color='red', linestyle='-', linewidth=2)
    ax4.plot(epochs, train_bal_accuracies, label='Train Bal Acc', color='green', linestyle='--', linewidth=2)
    ax4.plot(epochs, test_bal_accuracies, label='Test Bal Acc', color='orange', linestyle='--', linewidth=2)
    ax4.set_title(f'ROI {roi_number:02d} - Fold {fold}: All Metrics', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Metric Value')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    plot_filename = f"ROI_{roi_number:02d}_Fold_{fold}_training_curves.png"
    plot_path = os.path.join(plots_folder, plot_filename)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Training curves saved to: {plot_path}")


def extract_roi_number(filename):
    match = re.search(r'ROI_(\d+)', filename)
    if match:
        return int(match.group(1))
    else:
        return 0


def run_roi_file(file_path, roi_number, n_splits=3):
    df = pd.read_csv(file_path)
    y = df['research_group'].values
    X = df.drop(columns=['research_group']).values

    if X.shape[1] != 140:
        print(f"Warning: Expected 140 time points, got {X.shape[1]}. Using first 140 time points.")
        X = X[:, :140]

    SEQ_LEN, FEATURES = X.shape[1], 1
    X = X.reshape(X.shape[0], SEQ_LEN, FEATURES)
    print(f"Dataset shape: {X.shape}, Classes: {np.unique(y)}")
    print(f"Processing {SEQ_LEN} time series with {FEATURES} feature(s) per time point")
    print(f"BASELINE: Classical 1D-CNN on full {SEQ_LEN}-time-point signal (no quantum component)")

    skf = StratifiedKFold(n_splits=min(n_splits, len(np.unique(y)) * 2),
                          shuffle=True, random_state=42)
    fold_results = []
    all_fold_metrics = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        print(f"\n--- Fold {fold}/{skf.n_splits} ---")
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        train_unique, train_counts = np.unique(y_train, return_counts=True)
        test_unique, test_counts = np.unique(y_test, return_counts=True)
        print(f"Before balancing - Train: {dict(zip(train_unique, train_counts))}, Test: {dict(zip(test_unique, test_counts))}")

        if len(np.unique(y_train)) > 1:
            X_train_flat = X_train.reshape(len(X_train), -1)

            smote = SMOTE(random_state=42, k_neighbors=min(5, np.sum(y_train == 1) - 1, np.sum(y_train == 0) - 1))
            X_train_flat, y_train = smote.fit_resample(X_train_flat, y_train)
            X_train = X_train_flat.reshape(-1, SEQ_LEN, FEATURES)

            train_unique_after, train_counts_after = np.unique(y_train, return_counts=True)
            print(f"After balancing  - Train: {dict(zip(train_unique_after, train_counts_after))}")

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train.reshape(len(X_train), -1)).reshape(-1, SEQ_LEN, FEATURES)
        X_test = scaler.transform(X_test.reshape(len(X_test), -1)).reshape(-1, SEQ_LEN, FEATURES)

        classes = np.unique(y_train)
        if len(classes) > 1:
            cw = compute_class_weight("balanced", classes=classes, y=y_train)
            weight_vec = np.ones(int(classes.max()) + 1, dtype=float)
            for c, w in zip(classes, cw):
                weight_vec[int(c)] = w
            class_weights = torch.tensor(weight_vec, dtype=torch.float32).to(DEVICE)
            print(f"Class weights: Class 0: {weight_vec[0]:.3f}, Class 1: {weight_vec[1]:.3f}")
        else:
            class_weights = torch.tensor([1.0, 1.0], dtype=torch.float32).to(DEVICE)
            print("Warning: Only one class present in training data")

        train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                      torch.tensor(y_train, dtype=torch.float32))
        test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                                     torch.tensor(y_test, dtype=torch.float32))
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

        output_dim = 2
        model = Classical1DCNNModel(input_channels=FEATURES, output_dim=output_dim).to(DEVICE)

        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

        best_bal_acc = 0.0
        best_metrics = None
        train_losses, test_losses = [], []
        train_accuracies, test_accuracies = [], []
        train_bal_accuracies, test_bal_accuracies = [], []
        early_stopping = EarlyStopping()

        print("Epoch | Train Loss | Test Loss | Train Acc | Test Acc | Train BalAcc | Test BalAcc")
        print("----- | ---------- | --------- | --------- | -------- | ------------- | ------------")

        for epoch in range(EPOCHS):
            train_loss, train_acc, train_bal_acc, train_tn, train_fp, train_fn, train_tp = train_epoch(
                model, train_loader, criterion, optimizer, DEVICE)
            test_loss, test_acc, test_bal_acc, test_tn, test_fp, test_fn, test_tp = validate(
                model, test_loader, criterion, DEVICE)

            train_losses.append(train_loss)
            test_losses.append(test_loss)
            train_accuracies.append(train_acc)
            test_accuracies.append(test_acc)
            train_bal_accuracies.append(train_bal_acc)
            test_bal_accuracies.append(test_bal_acc)

            if test_bal_acc > best_bal_acc:
                best_bal_acc = test_bal_acc
                best_metrics = (train_loss, test_loss, train_acc, test_acc,
                                train_bal_acc, test_bal_acc,
                                train_tn, train_fp, train_fn, train_tp,
                                test_tn, test_fp, test_fn, test_tp)

            print(f"{epoch+1:4d} | {train_loss:10.4f} | {test_loss:9.4f} | "
                  f"{train_acc:9.4f} | {test_acc:8.4f} | {train_bal_acc:13.4f} | {test_bal_acc:12.4f}")

            early_stopping(test_bal_acc)
            if early_stopping.early_stop:
                print(f"Early stopping at epoch {epoch+1}")
                break

        plot_training_curves(train_losses, test_losses, train_accuracies, test_accuracies,
                           train_bal_accuracies, test_bal_accuracies, roi_number, fold, PLOTS_SUBFOLDER)

        print(f"\nBest Balanced Accuracy (fold {fold}): {best_bal_acc:.4f}")
        if best_metrics:
            (train_loss, test_loss, train_acc, test_acc,
             train_bal_acc, test_bal_acc,
             train_tn, train_fp, train_fn, train_tp,
             test_tn, test_fp, test_fn, test_tp) = best_metrics

            test_sensitivity = test_tp / (test_tp + test_fn) if (test_tp + test_fn) > 0 else 0.0
            test_specificity = test_tn / (test_tn + test_fp) if (test_tn + test_fp) > 0 else 0.0

            print(f"Train -> Loss: {train_loss:.4f}, TN: {train_tn}, FP: {train_fp}, FN: {train_fn}, TP: {train_tp}")
            print(f"Test  -> Loss: {test_loss:.4f}, TN: {test_tn}, FP: {test_fp}, FN: {test_fn}, TP: {test_tp}")
            print(f"Test Sensitivity: {test_sensitivity:.4f}, Specificity: {test_specificity:.4f}")

            fold_metrics = {
                'ROI_Number': roi_number,
                'Fold': fold,
                'Best_Balanced_Accuracy': best_bal_acc,
                'Best_Train_Loss': train_loss,
                'Best_Test_Loss': test_loss,
                'Best_Train_Accuracy': train_acc,
                'Best_Test_Accuracy': test_acc,
                'Test_Sensitivity': test_sensitivity,
                'Test_Specificity': test_specificity,
                'Train_TN': train_tn,
                'Train_FP': train_fp,
                'Train_FN': train_fn,
                'Train_TP': train_tp,
                'Test_TN': test_tn,
                'Test_FP': test_fp,
                'Test_FN': test_fn,
                'Test_TP': test_tp
            }
            all_fold_metrics.append(fold_metrics)

        fold_results.append(best_bal_acc)

    avg_bal_acc = np.mean(fold_results)
    print(f"\nAverage Balanced Accuracy: {avg_bal_acc:.4f}")

    roi_results = {
        'ROI_Number': roi_number,
        'Avg_Balanced_Accuracy': avg_bal_acc,
        'Fold_1_Balanced_Accuracy': fold_results[0] if len(fold_results) > 0 else 0,
        'Fold_2_Balanced_Accuracy': fold_results[1] if len(fold_results) > 1 else 0,
        'Fold_3_Balanced_Accuracy': fold_results[2] if len(fold_results) > 2 else 0,
        'Std_Balanced_Accuracy': np.std(fold_results) if len(fold_results) > 1 else 0
    }

    return roi_results, all_fold_metrics


if __name__ == "__main__":

    if not os.path.exists(ROI_FOLDER):
        print(f"✗ Error: ROI folder not found at: {ROI_FOLDER}")
        print(f"Please create a folder named '{ROI_FOLDER_NAME}' in the same directory as this script")
        exit(1)

    csv_files = glob.glob(os.path.join(ROI_FOLDER, "*.csv"))

    if not csv_files:
        print(f"✗ Error: No CSV files found in ROI folder: {ROI_FOLDER}")
        exit(1)

    csv_files_sorted = sorted(csv_files, key=lambda x: extract_roi_number(os.path.basename(x)))

    all_results = []
    all_detailed_metrics = []

    for idx, fpath in enumerate(csv_files_sorted, 1):
        filename = os.path.basename(fpath)
        roi_number = extract_roi_number(filename)

        print(f"\n=== Processing ROI {roi_number:02d} ({idx}/{len(csv_files_sorted)}): {filename} ===")
        try:
            roi_results, fold_metrics = run_roi_file(fpath, roi_number, n_splits=3)
            all_results.append(roi_results)
            all_detailed_metrics.extend(fold_metrics)

            roi_results['Status'] = "Success"
            roi_results['Filename'] = filename

        except Exception as e:
            print(f"Error processing {fpath}: {e}")
            all_results.append({
                'ROI_Number': roi_number,
                'Avg_Balanced_Accuracy': 0.0,
                'Fold_1_Balanced_Accuracy': 0.0,
                'Fold_2_Balanced_Accuracy': 0.0,
                'Fold_3_Balanced_Accuracy': 0.0,
                'Std_Balanced_Accuracy': 0.0,
                'Status': f"Error: {str(e)}",
                'Filename': filename
            })

    try:
        results_df = pd.DataFrame(all_results)
        results_path = os.path.join(CONDITION_FOLDER, "baseline_classical_1dcnn_results.csv")
        results_df.to_csv(results_path, index=False)
        print(f"\n✓ Results saved to: {results_path}")
    except Exception as e:
        print(f"✗ Error saving results: {e}")

    try:
        if all_detailed_metrics:
            detailed_df = pd.DataFrame(all_detailed_metrics)
            detailed_path = os.path.join(CONDITION_FOLDER, "detailed_baseline_classical_1dcnn_metrics.csv")
            detailed_df.to_csv(detailed_path, index=False)
            print(f"✓ Detailed metrics saved to: {detailed_path}")
    except Exception as e:
        print(f"✗ Error saving detailed metrics: {e}")

    if not results_df.empty:
        successful_runs = results_df[results_df['Status'] == 'Success']
        if not successful_runs.empty:
            print(f"\n=== Overall Statistics ===")
            print(f"Total ROIs processed: {len(results_df)}")
            print(f"Successful runs: {len(successful_runs)}")
            print(f"Average Balanced Accuracy: {successful_runs['Avg_Balanced_Accuracy'].mean():.4f}")
            print(f"Best ROI: ROI_{successful_runs.loc[successful_runs['Avg_Balanced_Accuracy'].idxmax(), 'ROI_Number']:02d} "
                  f"({successful_runs['Avg_Balanced_Accuracy'].max():.4f})")

    print(f"\n=== Analysis Complete ===")
    print(f"Model Architecture:")
    print("- BASELINE: Classical 1D-CNN (no quantum component)")
    print("- 3 Conv1d blocks (32/64/128 channels) + BatchNorm + ReLU + MaxPool")
    print("- Global Average Pooling -> FC classifier (96 -> 48 -> 2)")
    print("- Processes full 140-time-point signal")
    print(f"- All training plots saved to: {PLOTS_SUBFOLDER}")
