import torch
import torch.cuda.memory as memory

from tqdm import tqdm

# Set the fraction of GPU memory to be allocated
memory.set_per_process_memory_fraction(1.0)  # Use 80% of the available GPU memory

import torch
import torch.nn as nn
from torchvision import models
import torch.nn.init as init

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
    def __init__(self, embed_dim, hidden_dim, num_heads, num_layers, dropout=0.0, sub_len = 100):
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
        super(VisionTransformer, self).__init__()
        self.sub_len = sub_len
        # Layers/Networks
        self.input_layer = nn.Linear(4, embed_dim)
        self.transformer = nn.Sequential(*[AttentionBlock(embed_dim, hidden_dim, num_heads, dropout=dropout) for _ in range(num_layers)])
        
        self.fc = nn.Sequential(
            # nn.Linear(128 * 100, 64),
            nn.Linear(embed_dim * 100, 64),
            nn.Dropout(0.1),
            nn.ReLU(),
        )

        self.sim = nn.Sequential(
            # nn.Linear(64 * 2, 64),
            # nn.Dropout(0.1),
            # nn.ReLU(),
            # nn.Linear(64, 32),
            # nn.Dropout(0.1),
            # nn.ReLU(),
            # nn.Linear(32, 8),
            nn.Linear(64 * 2, 1),
            nn.Dropout(0.1),
            nn.ReLU(),
            # nn.Linear(8, 1),
            nn.Sigmoid(),
        )

        for layer in self.fc:
            if isinstance(layer, nn.Linear):
                init.xavier_normal_(layer.weight)
                init.zeros_(layer.bias)

        for layer in self.sim:
            if isinstance(layer, nn.Linear):
                init.xavier_normal_(layer.weight)
                init.zeros_(layer.bias)

        self.dropout = nn.Dropout(dropout)

        # Parameters/Embeddings
        self.cls_token = nn.Parameter(torch.randn(1,1,embed_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, 2001, embed_dim))

    def forward(self, x):
        # Preprocess input
        x1, x2 = x[:, 0, :, :], x[:, 1, :, :]
        B, T, _ = x1.shape
        x1 = self.input_layer(x1)
        x2 = self.input_layer(x2)

        # # Add CLS token and positional encoding
        cls_token = self.cls_token.repeat(B, 1, 1)
        x1 = torch.cat([cls_token, x1], dim=1)
        x1= x1 + self.pos_embedding[:,:T+1]
        x2 = torch.cat([cls_token, x2], dim=1)
        x2= x2 + self.pos_embedding[:,:T+1]

        # x1 = x1 + self.pos_embedding[:,:T]
        # x2 = x2 + self.pos_embedding[:,:T]

        # Apply Transforrmer
        # x = self.dropout(x)
        # x = x.transpose(0, 1)
        # x = self.transformer(x)
        x1 = self.dropout(x1)
        x1 = x1.transpose(0, 1)
        x1 = self.transformer(x1)
        x2 = self.dropout(x2)
        x2 = x2.transpose(0, 1)
        x2 = self.transformer(x2)

        # # flatten
        # x1 = x1.transpose(0, 1).reshape(B, -1)
        # x2 = x2.transpose(0, 1).reshape(B, -1)
        
        # TODO fix this part
        # Perform classification prediction
        outputs = []
        x1_list = []
        x2_list = []

        for i in range(x1.shape[0] // self.sub_len):
            sub_x1 = x1[(1+i * self.sub_len):(1+(i + 1) * self.sub_len), :, :]
            sub_x2 = x2[(1+i * self.sub_len):(1+(i + 1) * self.sub_len), :, :]
            # the shape of sub_x1 and sub_x2 is (sub_len, batch_size, hidden_size)

            # flatten 
            embedding_x1 = sub_x1.transpose(0, 1).reshape(B, -1)
            embedding_x2 = sub_x2.transpose(0, 1).reshape(B, -1)

            # print(embedding_x1.shape, embedding_x2.shape)

            # the shape of embedding_x1 and embedding_x2 is (batch_size, hidden_size)
            x1_list.append(embedding_x1)
            x2_list.append(embedding_x2)
        
        for i in range(len(x1_list)):
            for j in range(i+1, len(x1_list)):
                outputs.append(self.sim((torch.cat((self.fc(x1_list[i]), self.fc(x1_list[j])), dim=1))))
            for k in range(len(x2_list)):
                outputs.append(self.sim((torch.cat((self.fc(x1_list[i]), self.fc(x2_list[k])), dim=1))))
        for i in range(len(x2_list)):
            for j in range(i+1, len(x2_list)):
                outputs.append(self.sim((torch.cat((self.fc(x2_list[i]), self.fc(x2_list[j])), dim=1))))
        
        outputs = torch.cat(outputs, dim=1)
        # outputs shape: (batch_size, num_subtrajectories*(num_subtrajectories-1)/2 + num_subtrajectories*num_subtrajectories + num_subtrajectories*(num_subtrajectories-1)/2, 1)
        return outputs