# import torch.nn as nn
# from einops import rearrange

# class PoseTSM(nn.Module):
#     def __init__(self, embed_dim=128, num_classes=2):
#         super().__init__()
#         self.fc_in = nn.Linear(17*2, embed_dim)
#         self.relu  = nn.ReLU()
#         self.fc_mid = nn.Linear(embed_dim, embed_dim)
#         self.fc_out = nn.Linear(embed_dim, num_classes)

#     def temporal_shift(self, x):
#         B, T, C = x.shape
#         fold = C // 4
#         out = x.clone()

#         out[:, 1:, :fold] = x[:, :-1, :fold]
#         out[:, 0, :fold]  = 0
#         out[:, :-1, fold:2*fold] = x[:, 1:, fold:2*fold]
#         out[:, -1, fold:2*fold] = 0

#         return out

#     def forward(self, clip):
#         x = rearrange(clip, "b t k c -> b t (k c)")
#         x = self.fc_in(x)
#         x = self.relu(x)
#         x = self.temporal_shift(x)
#         x = x.mean(dim=1)
#         x = self.fc_mid(x)
#         x = self.relu(x)
#         return self.fc_out(x)

import torch
import torch.nn as nn
from einops import rearrange

class PoseTSM(nn.Module):
    def __init__(self, embed_dim=128, num_classes=2):
        super().__init__()
        self.fc_in = nn.Linear(17 * 2, embed_dim)
        self.relu = nn.ReLU()
        self.fc_mid = nn.Linear(embed_dim, embed_dim)
        self.fc_out = nn.Linear(embed_dim, num_classes)

    def temporal_shift(self, x):
        B, T, C = x.shape
        fold = C // 4

        out = x.clone()

        out[:, 1:, :fold] = x[:, :-1, :fold]
        out[:, 0, :fold] = 0

        out[:, :-1, fold:2*fold] = x[:, 1:, fold:2*fold]
        out[:, -1, fold:2*fold] = 0

        return out

    def forward(self, clip):
        # (B,16,17,2)
        x = rearrange(clip, "b t k c -> b t (k c)")  # (B,16,34)

        x = self.fc_in(x)
        x = self.relu(x)

        x = self.temporal_shift(x)

        x = x.mean(dim=1)   # EXACTLY SAME AS COLAB

        x = self.fc_mid(x)
        x = self.relu(x)

        return self.fc_out(x)
