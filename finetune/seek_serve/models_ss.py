# input, seek trajectories or serving trajectories
# seek trajectories, label 0, serve trajectories, label 1
# binary clasification

import torch
import torch.nn as nn
from torchvision import models
import torch.nn.init as init

# Binary classification for seek and serve using LSTM
class binaryLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=1):
        super(binaryLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size = self.input_size,
            hidden_size = self.hidden_size,
            num_layers = self.num_layers,
            batch_first=True,
        )

        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        # initialize weights
        for layer in self.fc:
            if isinstance(layer, nn.Linear):
                init.xavier_normal_(layer.weight)
                init.zeros_(layer.bias)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])
        return x

    
class binaryCNN(nn.Module):
    def __init__(self, input_size):
        super(binaryCNN, self).__init__()
        self.input_size = input_size
        
        # # Large CNN
        # self.conv = nn.Sequential(
        #     # nn.Conv1d(input_size, 640, 4),
        #     nn.Conv1d(input_size, 896, 3),
        #     nn.Dropout(0.1),
        #     nn.ReLU(),
        #     nn.MaxPool1d(3),
        #     nn.Conv1d(896, 256, 3),
        #     nn.Dropout(0.1),
        #     nn.ReLU(),
        #     nn.MaxPool1d(3),
        #     nn.Conv1d(256, 128, 3),
        #     nn.Dropout(0.1),
        #     nn.ReLU(),
        #     nn.MaxPool1d(3),
        #     nn.AdaptiveAvgPool1d(1)
        # )

        # Base CNN
        self.conv = nn.Sequential(
            nn.Conv1d(input_size, 256, 3),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.MaxPool1d(3),
            nn.Conv1d(256, 128, 3),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.MaxPool1d(3),
            nn.AdaptiveAvgPool1d(1)
        )

        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        # initialize weights
        for layer in self.fc:
            if isinstance(layer, nn.Linear):
                init.xavier_normal_(layer.weight)
                init.zeros_(layer.bias)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x ,



class AttentionBlock(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_heads, dropout=0.0):
        """
        Inputs:
            embed_dim - Dimensionality of input and attention feature vectors
            hidden_dim - Dimensionality of hidden layer in feed-forward network
                         (usually 2-4x larger than embed_dim)
            num_heads - Number of heads to use in the Multi-Head Attention block
            dropout - Amount of dropout to apply in the feed-forward network
        """
        super().__init__()

        self.layer_norm_1 = nn.LayerNorm(embed_dim)
        # self.attn = SelfAttention(embed_dim, num_heads)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.layer_norm_2 = nn.LayerNorm(embed_dim)
        self.linear = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # print('x1', x.shape)
        inp_x = self.layer_norm_1(x)
        # print('inp_x', inp_x.shape)
        # print(self.attn(inp_x, inp_x, inp_x, None).shape)
        # inp_x = x
        x = x + self.attn(inp_x, inp_x, inp_x, None)[0]
        # print('x2', x.shape)
        # print(self.layer_norm_2)
        x = x + self.linear(self.layer_norm_2(x))
        return x
    

class VisionTransformer(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_heads, num_layers, dropout=0.0):
        """
        Inputs:
            embed_dim - Dimensionality of the input feature vectors to the Transformer
            hidden_dim - Dimensionality of the hidden layer in the feed-forward networks
                         within the Transformer
            num_heads - Number of heads to use in the Multi-Head Attention block
            num_layers - Number of layers to use in the Transformer
            num_classes - Number of classes to predict
            dropout - Amount of dropout to apply in the feed-forward network and
                      on the input encoding
        """
        super().__init__()

        # Layers/Networks
        # in our  case, the  input size  is  (32,100,11), 32  is  the batch  size
        self.input_layer = nn.Linear(3, embed_dim)
        self.transformer = nn.Sequential(*[AttentionBlock(embed_dim, hidden_dim, num_heads, dropout=dropout) for _ in range(num_layers)])
        self.fc = nn.Sequential(
            # nn.Dropout(0.2),
            nn.Linear(embed_dim, 64),
            # nn.BatchNorm1d(64),
            # nn.ReLU(),
            # nn.Linear(64, 32),
            # nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
        
        for layer in self.fc:
            if isinstance(layer, nn.Linear):
                init.xavier_uniform_(layer.weight)
                init.zeros_(layer.bias)

        self.dropout = nn.Dropout(dropout)

        # Parameters/Embeddings
        self.cls_token = nn.Parameter(torch.randn(1,1,embed_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, 61, embed_dim))
        # another way to  initialize  the  parameters
        # self.cls_token = nn.Parameter(torch.zeros(1,1,embed_dim))
        # self.pos_embedding = nn.Parameter(torch.zeros(1, 101, embed_dim))

    def forward(self, x):
        # Preprocess input
        B, T, _ = x.shape
        # print(B, T)
        x = self.input_layer(x)
        # Add CLS token and positional encoding
        cls_token = self.cls_token.repeat(B, 1, 1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.pos_embedding[:,:T+1]

        # Apply Transforrmer
        x = self.dropout(x)
        x = x.transpose(0, 1)
        x = self.transformer(x)
        # Perform classification prediction
        # print(x.shape)
        # cls = x[0]
        # print(x.shape)
        # flatten, (batchsize, seuqence_len, embed_dim) -> (batchsize, seuqence_len * embed_dim)
        # cls = x.transpose(0, 1).reshape(B, -1)
        cls = x[0]
        # print(cls.shape)

        # print(cls.shape)
        # assert False
        out = self.fc(cls)
        return out
    

class binaryViT(nn.Module):
    ''' Build model for classification '''
    def __init__(self, input_size):
        super(binaryViT, self).__init__()
        self.input_size = input_size
        self.model = VisionTransformer(embed_dim=128, hidden_dim=128, num_heads=16, num_layers=8, dropout=0.1)
        # self.model = VisionTransformer(embed_dim=128, hidden_dim=64, num_heads=8, num_layers=1, dropout=0.1)
        # self.model = VisionTransformer(embed_dim=256, hidden_dim=128, num_heads=16, num_layers=1, dropout=0.1)
    
    def forward(self, x):
        prediction = self.model(x)
        return prediction

    def predict(self, x):
        x = self.forward(x)
        _, pred = torch.max(x.data, 1)
        return pred

    def save(self, path, weights_only=False):
        if weights_only:
            torch.save(self.state_dict(), path)
        else:
            torch.save(self, path)



