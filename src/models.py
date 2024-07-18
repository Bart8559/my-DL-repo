import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from torchvision.models import resnet18

class BasicConvClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        seq_len: int,
        in_channels: int,
        hid_dim: int = 128
    ) -> None:
        super().__init__()

        self.blocks = nn.Sequential(
            ConvBlock(in_channels, hid_dim),
            ConvBlock(hid_dim, hid_dim),
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            Rearrange("b d 1 -> b d"),
            nn.Linear(hid_dim, num_classes),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """_summary_
        Args:
            X ( b, c, t ): _description_
        Returns:
            X ( b, num_classes ): _description_
        """
        X = self.blocks(X)

        return self.head(X)


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        kernel_size: int = 3,
        p_drop: float = 0.1,
    ) -> None:
        super().__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.conv0 = nn.Conv1d(in_dim, out_dim, kernel_size, padding="same")
        self.conv1 = nn.Conv1d(out_dim, out_dim, kernel_size, padding="same")
        # self.conv2 = nn.Conv1d(out_dim, out_dim, kernel_size) # , padding="same")
        
        self.batchnorm0 = nn.BatchNorm1d(num_features=out_dim)
        self.batchnorm1 = nn.BatchNorm1d(num_features=out_dim)

        self.dropout = nn.Dropout(p_drop)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if self.in_dim == self.out_dim:
            X = self.conv0(X) + X  # skip connection
        else:
            X = self.conv0(X)

        X = F.gelu(self.batchnorm0(X))

        X = self.conv1(X) + X  # skip connection
        X = F.gelu(self.batchnorm1(X))

        # X = self.conv2(X)
        # X = F.glu(X, dim=-2)

        return self.dropout(X)
    
class ImprovedConvBlock(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=3, p_drop=0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_dim, out_dim, kernel_size, padding='same')
        self.bn1 = nn.BatchNorm1d(out_dim)
        self.conv2 = nn.Conv1d(out_dim, out_dim, kernel_size, padding='same')
        self.bn2 = nn.BatchNorm1d(out_dim)
        self.dropout = nn.Dropout(p_drop)
        self.shortcut = nn.Sequential()
        if in_dim != out_dim:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_dim, out_dim, kernel_size=1),
                nn.BatchNorm1d(out_dim)
            )

    def forward(self, X):
        out = F.relu(self.bn1(self.conv1(X)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(X)
        out = F.relu(out)
        return self.dropout(out)

class HybridModel(nn.Module):
    def __init__(self, num_classes, seq_len, num_channels):
        super(HybridModel, self).__init__()
        self.conv_block1 = ImprovedConvBlock(num_channels, 64, kernel_size=7, p_drop=0.2)
        self.conv_block2 = ImprovedConvBlock(64, 128, kernel_size=5, p_drop=0.2)
        self.lstm = nn.LSTM(128, 128, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(128*2, num_classes)

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = x.transpose(1, 2)  # LSTMに渡すために次元を入れ替え
        x, _ = self.lstm(x)
        x = x[:, -1, :]  # 最後のタイムステップの出力を使用
        x = self.fc(x)
        return x
    
class SimpleConvModel(nn.Module):
    def __init__(self, num_classes, seq_len, num_channels):
        super(SimpleConvModel, self).__init__()
        
        # 畳み込み層1
        self.conv1 = nn.Conv1d(num_channels, 64, kernel_size=3, padding='same')
        self.bn1 = nn.BatchNorm1d(64)
        
        # 畳み込み層2
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding='same')
        self.bn2 = nn.BatchNorm1d(128)
        
        # プーリング層
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        # 全結合層1
        self.fc1 = nn.Linear(128, 64)
        
        # 全結合層2
        self.fc2 = nn.Linear(64, num_classes)
        
    def forward(self, x):
        # 畳み込み層1
        x = F.relu(self.bn1(self.conv1(x)))
        
        # 畳み込み層2
        x = F.relu(self.bn2(self.conv2(x)))
        
        # プーリング層
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        
        # 全結合層1
        x = F.relu(self.fc1(x))
        
        # 全結合層2
        x = self.fc2(x)
        
        return x