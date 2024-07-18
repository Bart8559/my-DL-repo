import os
import numpy as np
import torch
from typing import Tuple
from termcolor import cprint

def preprocess_eeg(data):
    """
    EEGデータの前処理を行う関数
    Args:
        data (numpy.ndarray): EEGデータ（形状: チャネル x 系列長）
    Returns:
        numpy.ndarray: 前処理されたEEGデータ
    """
    # ベースライン補正
    baseline = np.mean(data[:, :50], axis=1, keepdims=True)  # 最初の50サンプルの平均を使用
    data_baseline_corrected = data - baseline

    # スケーリング（標準化）
    data_scaled = (data_baseline_corrected - np.mean(data_baseline_corrected, axis=1, keepdims=True)) / np.std(data_baseline_corrected, axis=1, keepdims=True)

    return data_scaled

class ThingsMEGDataset(torch.utils.data.Dataset):
    def __init__(self, split: str, data_dir: str = "data") -> None:
        super().__init__()
        
        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        self.split = split
        self.num_classes = 1854
        
        self.X = torch.load(os.path.join(data_dir, f"{split}_X.pt"))
        self.subject_idxs = torch.load(os.path.join(data_dir, f"{split}_subject_idxs.pt"))
        
        if split in ["train", "val"]:
            self.y = torch.load(os.path.join(data_dir, f"{split}_y.pt"))
            assert len(torch.unique(self.y)) == self.num_classes, "Number of classes do not match."

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i):
        X = self.X[i].numpy()  # テンソルからnumpy配列に変換
        X = preprocess_eeg(X)  # 前処理を適用
        X = torch.tensor(X, dtype=torch.float32)  # テンソルに戻す
        
        if hasattr(self, "y"):
            return X, self.y[i], self.subject_idxs[i]
        else:
            return X, self.subject_idxs[i]
        
    @property
    def num_channels(self) -> int:
        return self.X.shape[1]
    
    @property
    def seq_len(self) -> int:
        return self.X.shape[2]
