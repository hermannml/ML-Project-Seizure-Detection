import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from tqdm import tqdm
import mne
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import spectrogram
from collections import defaultdict
import gc
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
import scipy.stats

WINDOW_SIZE = 10
OVERLAP = 0.5
CONTEXT_WINDOWS = 3
CACHE_DIR = "cached"
os.makedirs(CACHE_DIR, exist_ok=True)


def load_and_window_edf(path, seizure_times, window_size=10, sfreq=128, overlap=0.5, context_windows=3, max_channels=128):
    try:
        raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
        raw.rename_channels(lambda ch: ch.replace('EEG ', '').replace('-REF', '').strip())
        raw.pick("eeg")
        raw.resample(sfreq, npad="auto", window="boxcar")

        raw.filter(l_freq=0.5, h_freq=40.0)
        
        data = raw.get_data()
        orig_channel_count = data.shape[0]
        
        if orig_channel_count < 16 or orig_channel_count > max_channels:
            return np.array([]), np.array([])

        padded_data = np.zeros((max_channels, data.shape[1]))
        padded_data[:orig_channel_count] = data
        data = padded_data

        n_samples = data.shape[1]
        w_len = int(window_size * sfreq)
        step = int(w_len * (1 - overlap))
        segments, labels = [], []

        for start in range(0, n_samples - w_len, step):
            end = start + w_len
            win = data[:, start:end]
            label = 0
            
            if seizure_times:
                t_start = start / sfreq
                t_end = end / sfreq
                
                for s, e in seizure_times:
                    
                    if (min(t_end, e) - max(t_start, s)) > (window_size * 0.5):
                        label = 1
                        break

            segments.append(win)
            labels.append(label)

        segments = np.array(segments)
        labels = np.array(labels)

        if context_windows > 1:
            pad = context_windows // 2
            segments = np.pad(segments, ((pad, pad), (0, 0), (0, 0)), mode="edge")
            context_segments = []
            
            for i in range(pad, len(segments) - pad):
                ctx = segments[i - pad: i + pad + 1].reshape(-1, segments.shape[-1])
                context_segments.append(ctx)
            
            segments = np.stack(context_segments)
        
        return segments, labels
    
    except Exception as e:
        print(f"Error processing {path}: {str(e)}")
        return np.array([]), np.array([])


def zscore_normalize(windows):
    mean = windows.mean(axis=2, keepdims=True)
    std = windows.std(axis=2, keepdims=True)
    std[std == 0] = 1
    return (windows - mean) / std


def extract_features(X, sfreq=128):
    features = []
    
    # for window in X:
    for window in tqdm(X, desc="Extracting features"):
        window_features = []
        
        for channel in window:
            mean = np.mean(channel)
            std = np.std(channel) + 1e-10
            max_val = np.max(channel)
            min_val = np.min(channel)
            rms = np.sqrt(np.mean(np.square(channel)))
            kurtosis = scipy.stats.kurtosis(channel)
            skewness = scipy.stats.skew(channel)
            
            zero_crossings = np.sum(np.diff(np.signbit(channel).astype(int)) != 0)
            zero_crossing_rate = zero_crossings / (len(channel) - 1)
            
            line_length = np.sum(np.abs(np.diff(channel)))
            
            fft_vals = np.abs(np.fft.rfft(channel))
            fft_freq = np.fft.rfftfreq(len(channel), 1.0/sfreq)
            
            delta_mask = (fft_freq >= 0.5) & (fft_freq < 4)
            theta_mask = (fft_freq >= 4) & (fft_freq < 8)
            alpha_mask = (fft_freq >= 8) & (fft_freq < 13)
            beta_mask = (fft_freq >= 13) & (fft_freq < 30)
            gamma_mask = (fft_freq >= 30) & (fft_freq <= 40)
            
            delta_power = np.sum(fft_vals[delta_mask]**2) + 1e-10
            theta_power = np.sum(fft_vals[theta_mask]**2) + 1e-10
            alpha_power = np.sum(fft_vals[alpha_mask]**2) + 1e-10
            beta_power = np.sum(fft_vals[beta_mask]**2) + 1e-10
            gamma_power = np.sum(fft_vals[gamma_mask]**2) + 1e-10
            total_power = delta_power + theta_power + alpha_power + beta_power + gamma_power
            
            delta_ratio = delta_power / total_power
            theta_ratio = theta_power / total_power
            alpha_ratio = alpha_power / total_power
            beta_ratio = beta_power / total_power
            gamma_ratio = gamma_power / total_power
            
            power_sum = np.sum(fft_vals**2) + 1e-10
            power_norm = fft_vals**2 / power_sum
            spectral_entropy = -np.sum(np.where(power_norm > 0, power_norm * np.log2(power_norm), 0))
            
            channel_features = [
                mean, std, max_val, min_val, rms, 
                kurtosis, skewness, zero_crossing_rate, line_length,
                delta_ratio, theta_ratio, alpha_ratio, beta_ratio, gamma_ratio,
                spectral_entropy
            ]
            
            window_features.extend(channel_features)
            
        features.append(window_features)
    
    return np.array(features)


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio, bias=False),
            nn.ReLU(),
            nn.Linear(in_channels // reduction_ratio, in_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        b, c, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        out = avg_out + max_out
        return self.sigmoid(out).view(b, c, 1)



class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv1d(2, 1, kernel_size=kernel_size, padding=(kernel_size-1)//2)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return self.sigmoid(out)



class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.ca = ChannelAttention(out_channels)
        self.sa = SpatialAttention()
        
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )
            
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out = self.ca(out) * out
        out = self.sa(out) * out
        
        if self.downsample is not None:
            residual = self.downsample(x)
            
        out += residual
        out = self.relu(out)
        
        return out


def plot_learning_curves(history, title):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'])
    plt.title(f'{title} - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(history['acc'], label='Accuracy')
    plt.plot(history['prec'], label='Precision')
    plt.plot(history['rec'], label='Recall')
    plt.plot(history['f1'], label='F1 Score')
    plt.title(f'{title} - Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{title.lower().replace(' ', '_')}_learning_curves.png")
    # plt.show()
    plt.close()


def plot_conf_matrix(cm, name):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="viridis", xticklabels=["Non-seizure", "Seizure"], yticklabels=["Non-seizure", "Seizure"])
    plt.title(f"Confusion Matrix: {name}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(f"{name.lower().replace(' ', '_')}_confusion_matrix.png")
    # plt.show()
    plt.close()


def compare_models_performance(model_metrics, std_metrics):
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    
    plt.figure(figsize=(12, 8))
    
    x = np.arange(len(metrics))
    width = 0.2
    offsets = np.linspace(-width * len(model_metrics) / 2, width * len(model_metrics) / 2, len(model_metrics))
    
    for i, (model_name, model_metric) in enumerate(model_metrics.items()):
        values = [model_metric[m] for m in metrics]
        errors = [std_metrics[model_name][m] for m in metrics]
        
        
        plt.bar(x + offsets[i], values, width, label=model_name, yerr=errors, capsize=5)
    
    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.title('Performance Comparison Across Models (Mean ± Std)')
    plt.xticks(x, metrics)
    plt.legend()
    plt.ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig("model_comparison.png")
    # plt.show()


def load_files(x_files, y_files, cache_dir, use_features=False):
    X_list, y_list, feat_list = [], [], []

    # for xf, yf in zip(x_files, y_files):
    for xf, yf in tqdm(zip(x_files, y_files), total=len(x_files), desc="Loading files"):
        X = np.load(os.path.join(cache_dir, xf))
        y = np.load(os.path.join(cache_dir, yf))
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)

        X_list.append(X)
        y_list.append(y)

        if use_features:
            feats = extract_features(X.numpy())
            feat_list.append(torch.tensor(feats, dtype=torch.float32))

    X_all = torch.cat(X_list, dim=0)
    y_all = torch.cat(y_list, dim=0)

    if use_features:
        feats_all = torch.cat(feat_list, dim=0)
        dataset = TensorDataset(X_all, feats_all, y_all)
    else:
        dataset = TensorDataset(X_all, y_all)

    return dataset

def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for data in test_loader:
            if len(data) == 3:
                inputs, features, labels = [d.to(device) for d in data]
                if features is not None:
                    features = features.mean(dim=1)
                outputs = model(inputs, features)
            else:
                inputs, labels = [d.to(device) for d in data]
                outputs = model(inputs)

            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

    all_preds = postprocess_predictions(np.array(all_preds))

    return {
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, zero_division=0),
        "recall": recall_score(all_labels, all_preds, zero_division=0),
        "f1": f1_score(all_labels, all_preds, zero_division=0),
        "roc_auc": roc_auc_score(all_labels, all_probs)
    }, confusion_matrix(all_labels, all_preds)
    


def create_better_sampler(labels):
    class_counts = np.bincount(labels)
    seizure_count = class_counts[1]
    nonseizure_count = class_counts[0]
    
    weight_seizure = nonseizure_count / (seizure_count * 3)
    weights = np.ones_like(labels, dtype=float)
    weights[labels == 1] = weight_seizure
    
    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True
    )
    
    return sampler


class EnhancedSeizureDetector(nn.Module):
    def __init__(self, n_channels, seq_len, dropout_rate=0.3):
        super().__init__()
        
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(n_channels, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout_rate),
            
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ELU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout_rate),
            
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ELU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout_rate),
        )
        
        self.attention = nn.MultiheadAttention(embed_dim=256, num_heads=8, dropout=dropout_rate)
        self.norm = nn.LayerNorm(256)
        
        self.lstm = nn.LSTM(256, 128, num_layers=2, batch_first=True, 
                          bidirectional=True, dropout=dropout_rate)
        
        reduced_seq_len = seq_len // 8
        self.classifier = nn.Sequential(
            nn.AdaptiveMaxPool1d(8),
            nn.Flatten(),
            nn.Linear(256 * 8, 128),
            nn.ELU(),
            nn.Dropout(dropout_rate + 0.1),
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Dropout(dropout_rate + 0.1),
        )
    
        input_feature_dim = n_channels * 15
        self.feature_net = nn.Sequential(
            nn.BatchNorm1d(input_feature_dim),
            nn.Linear(input_feature_dim, 128),
            nn.ELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ELU()
        )
        
        self.fusion = nn.Linear(64 + 64, 2)
        
    def forward(self, x, additional_features=None):
        features = self.feature_extractor(x)
        
        attn_features = features.permute(2, 0, 1)
        attn_out, _ = self.attention(attn_features, attn_features, attn_features)
        attn_out = attn_out.permute(1, 2, 0)
        
        lstm_in = features.permute(0, 2, 1)
        lstm_out, _ = self.lstm(lstm_in)
        lstm_out = lstm_out.permute(0, 2, 1)
        
        combined = features + attn_out + lstm_out[:, :256, :]
        
        deep_features = self.classifier(combined)
        
        if additional_features is not None:
            try:
                if len(additional_features.shape) == 3:
                    batch_size, seq_len, feat_dim = additional_features.shape
                    additional_features = additional_features.mean(dim=1)
                
                eng_features = self.feature_net(additional_features)
                
                concat_features = torch.cat([deep_features, eng_features], dim=1)
                output = self.fusion(concat_features)
            except Exception as e:
                print(f"Warning: Feature processing error, using deep features only. Error: {e}")
                output = nn.Linear(64, 2).to(deep_features.device)(deep_features)
        else:
            output = nn.Linear(64, 2).to(deep_features.device)(deep_features)
            
        return output
            

def train_optimized_model(model, train_loader, val_loader, device, epochs=15, class_weight=None):
    model.to(device)
    
    class FocalLoss(nn.Module):
        def __init__(self, weight=None, gamma=2.0, reduction='mean'):
            super().__init__()
            self.weight = weight
            self.gamma = gamma
            self.reduction = reduction   

        def forward(self, input, target):
            ce_loss = F.cross_entropy(input, target, weight=self.weight, reduction='none')
            pt = torch.exp(-ce_loss)
            loss = (1 - pt) ** self.gamma * ce_loss
            
            if self.reduction == 'mean':
                return loss.mean()
            elif self.reduction == 'sum':
                return loss.sum()
            else:
                return loss
    
    if class_weight is not None:
        class_weights_tensor = torch.tensor([class_weight[0], class_weight[1]], dtype=torch.float32).to(device)
    else:
        class_weights_tensor = None
        
    criterion = FocalLoss(weight=class_weights_tensor, gamma=2.0)
    
    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)
    
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=0.001,
        steps_per_epoch=len(train_loader),
        epochs=epochs,
        pct_start=0.3,
        div_factor=25,
        final_div_factor=1000
    )
    
    best_val_f1, best_model_state = 0, None
    patience, patience_counter = 5, 0
    history = {'loss': [], 'val_loss': [], 'acc': [], 'prec': [], 'rec': [], 'f1': []}
    
    scaler = torch.cuda.amp.GradScaler() if device == 'cuda' else None
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        all_preds, all_labels = [], []
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for data in progress_bar:
            if len(data) == 3:
                inputs, features, labels = [d.to(device) for d in data]
            else:
                inputs, labels = [d.to(device) for d in data]
                features = None
            
            optimizer.zero_grad()
            
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = model(inputs, features) if features is not None else model(inputs)
                    loss = criterion(outputs, labels)
                
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

            else:
                outputs = model(inputs, features) if features is not None else model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
            scheduler.step()
        
        progress_bar.close()

        val_metrics, val_loss = validate_model(model, val_loader, device, criterion)
        
        acc = accuracy_score(all_labels, all_preds)
        prec = precision_score(all_labels, all_preds, zero_division=0)
        rec = recall_score(all_labels, all_preds, zero_division=0)
        f1 = f1_score(all_labels, all_preds, zero_division=0)
        
        history['loss'].append(running_loss / len(train_loader))
        history['val_loss'].append(val_loss)
        history['acc'].append(acc)
        history['prec'].append(prec)
        history['rec'].append(rec)
        history['f1'].append(f1)
        
        print(f"Epoch {epoch+1} — train_loss: {running_loss/len(train_loader):.4f}, val_loss: {val_loss:.4f}, "
              f"acc: {acc:.4f}, prec: {prec:.4f}, rec: {rec:.4f}, f1: {f1:.4f}, val_f1: {val_metrics['f1']:.4f}")
        
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0

        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    if best_model_state:
        model.load_state_dict(best_model_state)
        print(f"Loaded best model with validation F1: {best_val_f1:.4f}")
    
    return model, history


def validate_model(model, val_loader, device, criterion):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for data in val_loader:
            if len(data) == 3:
                inputs, features, labels = [d.to(device) for d in data]
                if len(features.shape) == 3 and features.shape[1] > 1:
                    features = features.mean(dim=1)
                outputs = model(inputs, features)
            
            else:
                inputs, labels = [d.to(device) for d in data]
                outputs = model(inputs)

            loss = criterion(outputs, labels)
            running_loss += loss.item()
            
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

    post_preds = postprocess_predictions(np.array(all_preds))
    
    acc = accuracy_score(all_labels, post_preds)
    prec = precision_score(all_labels, post_preds, zero_division=0)
    rec = recall_score(all_labels, post_preds, zero_division=0)
    f1 = f1_score(all_labels, post_preds, zero_division=0)
    
    try:
        roc = roc_auc_score(all_labels, all_probs)
    except:
        roc = 0.5

    metrics = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": roc
    }
    val_loss = running_loss / len(val_loader)

    return metrics, val_loss


def postprocess_predictions(predictions, threshold=3):
    """
    Apply temporal smoothing to reduce false alarms
    """
    smooth_predictions = np.copy(predictions)
    for i in range(len(predictions)-threshold+1):
        if np.sum(predictions[i:i+threshold]) > 0:
            smooth_predictions[i:i+threshold] = 1
    return smooth_predictions


def main():
    with open("real_seizure_annotations.pkl", "rb") as f:
        annots = pickle.load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if not os.listdir(CACHE_DIR):
        print("Caching EEG features...")
        for i, (path, seizures) in enumerate(tqdm(annots.items())):
            X, y = load_and_window_edf(path, seizures)
            if len(X) == 0 or len(y) == 0:
                continue
            
            if np.all(y == 0) and np.random.random() > 0.3:
                continue
                
            X = zscore_normalize(X)
            np.save(os.path.join(CACHE_DIR, f"X_{i}.npy"), X)
            np.save(os.path.join(CACHE_DIR, f"y_{i}.npy"), y)
            
            del X, y
            gc.collect()

    X_files = sorted([f for f in os.listdir(CACHE_DIR) if f.startswith("X_")])
    y_files = sorted([f for f in os.listdir(CACHE_DIR) if f.startswith("y_")])
    
    total_samples = 0
    class_counts = np.zeros(2, dtype=int)
    for yf in y_files:
        y = np.load(os.path.join(CACHE_DIR, yf))
        total_samples += len(y)
        class_counts += np.bincount(y, minlength=2)
        del y
        gc.collect()

    weight_for_0 = 1.0
    weight_for_1 = (class_counts[0] / class_counts[1]) * 2
    class_weight = {0: float(weight_for_0), 1: float(weight_for_1)}
    print(f"Class weights: {class_weight}")

    sample_X = np.load(os.path.join(CACHE_DIR, X_files[0]))
    n_channels = sample_X.shape[1]
    seq_len = sample_X.shape[2]
    del sample_X
    gc.collect()

    all_labels = []
    for yf in y_files:
        y = np.load(os.path.join(CACHE_DIR, yf))
        has_seizure = 1 if np.any(y == 1) else 0
        all_labels.append(has_seizure)
        del y
        gc.collect()

    results = {
        'Enhanced Model': [],
        'Enhanced Model with Features': []
    }

    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    for fold, (train_file_idx, test_file_idx) in enumerate(skf.split(range(len(X_files)), all_labels)):
        print(f"\nFold {fold+1}/{n_splits}:")
        
        train_x = [X_files[i] for i in train_file_idx]
        train_y = [y_files[i] for i in train_file_idx]
        test_x = [X_files[i] for i in test_file_idx]
        test_y = [y_files[i] for i in test_file_idx]
        
        val_size = len(train_x) // 5
        val_x = train_x[:val_size]
        val_y = train_y[:val_size]
        train_x = train_x[val_size:]
        train_y = train_y[val_size:]
        
        train_dataset = load_files(train_x, train_y, CACHE_DIR, use_features=False)
        val_dataset = load_files(val_x, val_y, CACHE_DIR, use_features=False)
        test_dataset = load_files(test_x, test_y, CACHE_DIR, use_features=False)
        
        train_labels = np.array([y.item() for _, y in train_dataset])
        train_sampler = create_better_sampler(train_labels)
        
        train_loader = DataLoader(train_dataset, batch_size=32, sampler=train_sampler)
        val_loader = DataLoader(val_dataset, batch_size=64)
        test_loader = DataLoader(test_dataset, batch_size=64)
        
        print("\nTraining Enhanced Model:")
        model = EnhancedSeizureDetector(n_channels, seq_len)
        model, history = train_optimized_model(
            model, train_loader, val_loader, device, 
            epochs=15, class_weight=class_weight
        )
        
        metrics, cm = evaluate_model(model, test_loader, device)
        results['Enhanced Model'].append(metrics)
        
        plot_learning_curves(history, f"Enhanced Model - Fold {fold+1}")
        plot_conf_matrix(cm, f"Enhanced Model - Fold {fold+1}")
        
        del model, train_loader, val_loader, test_loader
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()
        
        train_dataset_feats = load_files(train_x, train_y, CACHE_DIR, use_features=True)
        val_dataset_feats = load_files(val_x, val_y, CACHE_DIR, use_features=True)
        test_dataset_feats = load_files(test_x, test_y, CACHE_DIR, use_features=True)
        
        train_loader_feats = DataLoader(train_dataset_feats, batch_size=32, sampler=train_sampler)
        val_loader_feats = DataLoader(val_dataset_feats, batch_size=64)
        test_loader_feats = DataLoader(test_dataset_feats, batch_size=64)
        
        print("\nTraining Enhanced Model with Feature Engineering:")
        model_feats = EnhancedSeizureDetector(n_channels, seq_len)
        model_feats, history_feats = train_optimized_model(
            model_feats, train_loader_feats, val_loader_feats, device, 
            epochs=15, class_weight=class_weight
        )
        
        metrics_feats, cm_feats = evaluate_model(model_feats, test_loader_feats, device)
        results['Enhanced Model with Features'].append(metrics_feats)
        
        plot_learning_curves(history_feats, f"Enhanced Model with Features - Fold {fold+1}")
        plot_conf_matrix(cm_feats, f"Enhanced Model with Features - Fold {fold+1}")
        
        del model_feats, train_loader_feats, val_loader_feats, test_loader_feats
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()

    avg_results = {}
    std_results = {}

    for model_name, model_results in results.items():
        avg_results[model_name] = {
            metric: np.mean([res[metric] for res in model_results])
            for metric in model_results[0].keys()
        }
        std_results[model_name] = {
            metric: np.std([res[metric] for res in model_results])
            for metric in model_results[0].keys()
        }

        print(f"\n{model_name} - Average Results")
        for metric, value in avg_results[model_name].items():
            print(f"{metric:<10}: {value:.4f} ± {std_results[model_name][metric]:.4f}")

    compare_models_performance(avg_results, std_results)
    
    best_model = max(avg_results.items(), key=lambda x: x[1]['f1'])
    print(f"\n=== Best Model: {best_model[0]} ===")
    print(f"F1 Score: {best_model[1]['f1']:.4f} ± {std_results[best_model[0]]['f1']:.4f}")
    print(f"Accuracy: {best_model[1]['accuracy']:.4f} ± {std_results[best_model[0]]['accuracy']:.4f}")
    print(f"ROC AUC: {best_model[1]['roc_auc']:.4f} ± {std_results[best_model[0]]['roc_auc']:.4f}")



if __name__ == "__main__":
    main()
