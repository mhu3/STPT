
import torch
import torch.nn as nn
from torchvision import models
import torch.nn.init as init

class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=3):
        super(SimpleLSTM, self).__init__()
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
            # nn.Dropout(0.2),
            # nn.Linear(embed_dim * 2000, 128),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            # nn.Dropout(0.2),
            # nn.ReLU(),
            # nn.Linear(128, 128),
            # nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(64, 8)
        )
                

    def forward(self, x):
        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])
        return x

    def predict(self, x):
        x = self.forward(x)
        _, pred = torch.max(x.data, 1)
        return pred

    def save(self, path, weights_only=False):
        if weights_only:
            torch.save(self.state_dict(), path)
        else:
            torch.save(self, path)


class SimpleCNN(nn.Module):
    def __init__(self, input_size):
        super(SimpleCNN, self).__init__()
        self.input_size = input_size
        
        # self.conv = nn.Sequential(
        #     # nn.Conv1d(input_size, 640, 4),
        #     nn.Conv1d(input_size, 896, 4),
        #     nn.Dropout(0.1),
        #     nn.ReLU(),
        #     nn.MaxPool1d(4),
        #     nn.Conv1d(896, 256, 4),
        #     nn.Dropout(0.1),
        #     nn.ReLU(),
        #     nn.MaxPool1d(4),
        #     nn.Conv1d(256, 128, 4),
        #     nn.Dropout(0.2),
        #     nn.ReLU(),
        #     nn.MaxPool1d(4),
        #     nn.AdaptiveAvgPool1d(1)
        # )
        self.conv = nn.Sequential(
            nn.Conv1d(input_size, 640, 4),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(640, 128, 4),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.AdaptiveAvgPool1d(1)
          )

        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(64, 8)
        )

        for layer in self.fc:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
    def predict(self, x):
        x = self.forward(x)
        _, pred = torch.max(x.data, 1)
        return pred
    
    def save(self, path, weights_only=False):
        if weights_only:
            torch.save(self.state_dict(), path)
        else:
            torch.save(self, path)


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
        inp_x = self.layer_norm_1(x)
        x = x + self.attn(inp_x, inp_x, inp_x, None)[0]
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
        self.input_layer = nn.Linear(4, embed_dim)
        self.transformer = nn.Sequential(*[AttentionBlock(embed_dim, hidden_dim, num_heads, dropout=dropout) for _ in range(num_layers)])
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(64, 8)
        )

        for layer in self.fc:
            if isinstance(layer, nn.Linear):
                init.xavier_uniform_(layer.weight)
                init.zeros_(layer.bias)

        self.dropout = nn.Dropout(dropout)

        # Parameters/Embeddings
        self.cls_token = nn.Parameter(torch.randn(1,1,embed_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, 2001, embed_dim))
        # another way to  initialize  the  parameters
        # self.cls_token = nn.Parameter(torch.zeros(1,1,embed_dim))
        # self.pos_embedding = nn.Parameter(torch.zeros(1, 101, embed_dim))

    def forward(self, x):
        # Preprocess input
        B, T, _ = x.shape
        # print(B, T)
        x = self.input_layer(x)
        # # Add CLS token and positional encoding
        cls_token = self.cls_token.repeat(B, 1, 1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.pos_embedding[:,:T+1]

        # Apply Transforrmer
        x = self.dropout(x)
        x = x.transpose(0, 1)
        x = self.transformer(x)
        cls = x[0]
        out = self.fc(cls)
        return out
    

class SimpleViT(nn.Module):
    ''' Build model for classification '''
    def __init__(self, input_size):
        super(SimpleViT, self).__init__()
        self.input_size = input_size
        # self.model = VisionTransformer(embed_dim=128, hidden_dim=128, num_heads=16, num_layers=8, dropout=0.1)
        self.model = VisionTransformer(embed_dim=128, hidden_dim=64, num_heads=8, num_layers=1, dropout=0.1)
    
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
