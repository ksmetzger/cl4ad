import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

#Implement transformer (only encoder w/o positional encoding) from "Attention is all you need": https://arxiv.org/pdf/1706.03762.pdf
#using the implemenation from pytorch, similar to "JetCLR": https://github.com/bmdillon/JetCLR/blob/main/scripts/modules/transformer.py
class TransformerEncoder(nn.Module):
    def __init__(self,input_dim=3, model_dim=512, output_dim=512, n_heads=8, dim_feedforward=2048, n_layers=6, n_head_layers=2, head_norm=False, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.input_dim = input_dim
        self.model_dim = model_dim
        self.output_dim = output_dim
        self.n_heads = n_heads
        self.dim_feedforward = dim_feedforward
        self.n_layers = n_layers
        self.n_head_layers = n_head_layers
        self.head_norm = head_norm
        self.dropout = dropout
        #encoder part from pytorch
        self.embedding = nn.Linear(input_dim, model_dim)
        #Get a pre-norm tranformer encoder from pytorch
        self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(model_dim, n_heads, dim_feedforward=dim_feedforward, dropout=dropout, norm_first=True), n_layers)
        self.dino_head = DINOHead(in_dim=self.model_dim, out_dim=self.output_dim)
        #Define [CLS] token for aggregate result where head attaches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, model_dim))
    
    #Embedding output without the DINO head
    def representation(self, x):
        self.batch_size = x.shape[0]
        #Create embedding for tranformer input
        x = self.embedding(x)
        #Add CLS token
        cls_tokens = self.cls_token.expand(self.batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        #Transpose (bsz, n_const, model_dim) -> (n_const, bsz, model_dim) for transformer input
        x = torch.transpose(x,0,1)
        #Input to transformer encoder
        x = self.transformer(x)
        #Only take [CLS] token and input into DINO head
        x = x[0,...]
        return x
    
    #For now the implementation does not use masking like in JetCLR, might be revisited later.
    def forward(self, x, mask=None, use_mask=False, use_continuous_mask=False, mult_reps=False):
        x = self.representation(x)
        x = self.dino_head(x)
        return x


#Import the projection head for the DINO loss from https://github.com/facebookresearch/dino/blob/main/vision_transformer.py
class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True, nlayers=3, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x