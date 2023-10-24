# -*- coding: utf-8 -*-
# @Author: xiegr
# @Date:   2020-08-30 15:58:51
# @Last Modified by:   xiegr
# @Last Modified time: 2020-09-18 14:22:48
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import math

from torch.autograd import Variable

torch.manual_seed(2020)
torch.cuda.manual_seed_all(2020)
np.random.seed(2020)
random.seed(2020)
torch.backends.cudnn.deterministic = True


class SelfAttention(nn.Module):
    """docstring for SelfAttention"""

    def __init__(self, d_dim=256, dropout=0.1):
        super(SelfAttention, self).__init__()
        # for query, key, value, output
        self.dim = d_dim
        self.linears = nn.ModuleList([nn.Linear(d_dim, d_dim) for _ in range(4)])
        self.dropout = nn.Dropout(p=dropout)

    def attention(self, query, key, value):
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.dim)
        scores = F.softmax(scores, dim=-1)
        return scores

    def forward(self, query, key, value):
        # 1) query, key, value
        query, key, value = \
            [l(x) for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention
        scores = self.attention(query, key, value)  # 算自相关矩阵 sij表示第i个位置和第j个位置的相关程度
        x = torch.matmul(scores,
                         value)  # torch.Size([16384, 50, 50]) * torch.Size([16384, 50, 256])  得到50 * 256   每个特征通道上所有位置的加权平均

        # 3) apply the final linear
        x = self.linears[-1](x.contiguous())  # torch.Size([16384, 50, 256])过linear得到torch.Size([16384, 50, 256])
        # sum keepdim=False
        return self.dropout(x), torch.mean(scores, dim=-2)  # torch.Size([16384, 50, 256])   torch.Size([16384, 50])


class OneDimCNN(nn.Module):
    """docstring for OneDimCNN"""

    # https://blog.csdn.net/sunny_xsc1994/article/details/82969867
    def __init__(self, max_byte_len, d_dim=256, \
                 kernel_size=[3, 4], filters=256, dropout=0.1):
        super(OneDimCNN, self).__init__()
        self.kernel_size = kernel_size
        self.convs = nn.ModuleList([
            nn.Sequential(nn.Conv1d(in_channels=d_dim,
                                    out_channels=filters,
                                    kernel_size=h),
                          # nn.BatchNorm1d(num_features=config.feature_size),
                          nn.ReLU(),
                          # MaxPool1d:
                          # stride – the stride of the window. Default value is kernel_size
                          nn.MaxPool1d(kernel_size=max_byte_len - h + 1))
            for h in self.kernel_size
        ]
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        out = [conv(x.transpose(-2, -1)) for conv in
               self.convs]  # 两个一维卷积单独运算得到两个结果 [torch.Size([16384, 256, 1]), torch.Size([16384, 256, 1])]
        out = torch.cat(out, dim=1)
        out = out.view(-1, out.size(1))  # torch.Size([16384, 512])
        return self.dropout(out)


# # raw
# class SAM(nn.Module):
#     """docstring for SAM"""
#
#     # total header bytes 24
#     def __init__(self, num_class, max_byte_len, kernel_size=[3, 4], \
#                  d_dim=256, dropout=0.1, filters=256):
#         super(SAM, self).__init__()
#         self.posembedding = nn.Embedding(num_embeddings=max_byte_len,
#                                          embedding_dim=d_dim)  # nn.Embedding是onehot接矩阵乘法 onehot*weight    3*3 * 3*2
#         self.byteembedding = nn.Embedding(num_embeddings=300,
#                                           embedding_dim=d_dim)
#         self.attention = SelfAttention(d_dim, dropout)
#         self.cnn = OneDimCNN(max_byte_len, d_dim, kernel_size, filters, dropout)
#         self.fc = nn.Linear(in_features=256 * len(kernel_size),
#                             out_features=num_class)
#
#     def forward(self, x, y):
#         out = self.byteembedding(x) + self.posembedding(y)  # torch.Size([16384, 50, 256])
#         out, score = self.attention(out, out, out)  # torch.Size([16384, 50, 256])		torch.Size([16384, 50])
#         out = self.cnn(out)  # 每个流量生成512的特征
#         out = self.fc(out)  # 分类器
#         if not self.training:
#             return F.softmax(out, dim=-1).max(1)[1], score
#         return out


# 转2d 64*64
class SAMap(nn.Module):
    """docstring for SelfAttention"""
    def __init__(self, d_dim=50):
        super(SAMap, self).__init__()
        # for query, key, value, output
        self.dim = d_dim
        self.t_dim = 64
        self.linears = nn.ModuleList([nn.Linear(d_dim, self.t_dim) for _ in range(2)])

    def attention(self, query, key):

        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.dim)
        scores = torch.tanh(scores)
        return scores

    def forward(self, query, key):

        # 1) query, key, value
        query, key = [l(x) for l, x in zip(self.linears, (query, key))]

        query = query.permute(0, 2, 1)
        key = key.permute(0, 2, 1)

        # 2) Apply attention
        scores = self.attention(query, key)			# 算自相关矩阵 sij表示第i个位置和第j个位置的相关程度

        return scores.unsqueeze(1)

class SAM(nn.Module):
    """docstring for SAM"""
    # total header bytes 24
    def __init__(self, num_class, max_byte_len, kernel_size = [3, 4], \
        d_dim=256, dropout=0.1, filters=256):
        super(SAM, self).__init__()
        self.posembedding = nn.Embedding(num_embeddings=max_byte_len,
                                embedding_dim=d_dim)					# nn.Embedding是onehot接矩阵乘法 onehot*weight    3*3 * 3*2
        self.byteembedding = nn.Embedding(num_embeddings=300,
                                embedding_dim=d_dim)

        self.sam = SAMap(256)    # L的值

        img_layers, in_features = self.get_img_layers('resnet18', feat_size=16)
        self.img_model = nn.Sequential(*img_layers)
        self.final_fc = nn.Sequential(nn.Linear(in_features, in_features),
                                      nn.BatchNorm1d(in_features),
                                      nn.ReLU(inplace=True),
                                      nn.Dropout(dropout),
                                      nn.Linear(in_features, num_class),
                                      )

    @staticmethod
    def get_img_layers(backbone, feat_size):
        """
        Return layers for the image model
        """

        from resnet import _resnet, BasicBlock
        assert backbone == 'resnet18'
        layers = [1, 1, 1, 1]
        block = BasicBlock
        backbone_mod = _resnet(
            arch=None,
            block=block,
            layers=layers,
            pretrained=False,
            progress=False,
            feature_size=feat_size,
            zero_init_residual=True)

        all_layers = [x for x in backbone_mod.children()]
        in_features = all_layers[-1].in_features

        # all layers except the final fc layer and the initial conv layers
        # WARNING: this is checked only for resnet models
        main_layers = all_layers[4:-1]
        img_layers = [
            nn.Conv2d(1, feat_size, kernel_size=(3, 3), stride=(1, 1),
                      padding=(1, 1), bias=False),
            nn.BatchNorm2d(feat_size, eps=1e-05, momentum=0.1,
                           affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            *main_layers,
        ]

        return img_layers, in_features

    def forward(self, x, y):
        out = self.byteembedding(x) + self.posembedding(y)			# torch.Size([16384, 50, 256])
        img = self.sam(out, out)
        feat = self.img_model(img).squeeze()
        out = self.final_fc(feat)

        if not self.training:
            return F.softmax(out, dim=-1).max(1)[1], img
        return out

if __name__ == '__main__':
    x = np.random.randint(0, 255, (10, 20))
    y = np.random.randint(0, 20, (10, 20))
    sam = SAM(num_class=5, max_byte_len=20)
    out = sam(torch.from_numpy(x).long(), torch.from_numpy(y).long())
    print(out[0])

    sam.eval()
    out, score = sam(torch.from_numpy(x).long(), torch.from_numpy(y).long())
    print(out[0], score[0])
