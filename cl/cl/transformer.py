import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt

#Implement transformer (only encoder w/o positional encoding) from "Attention is all you need": https://arxiv.org/pdf/1706.03762.pdf
#using the implemenation from pytorch, similar to "JetCLR": https://github.com/bmdillon/JetCLR/blob/main/scripts/modules/transformer.py
class TransformerEncoder(nn.Module):
    def __init__(self,input_dim=3, model_dim=512, output_dim=512, embed_dim=64, n_heads=8, dim_feedforward=2048, n_layers=6, hidden_dim_dino_head=2048, bottleneck_dim_dino_head=256, head_norm=False, 
                 dropout=0.1, norm_last_layer=True, pos_encoding=False, use_mask=False, mode='cls'):
        super(TransformerEncoder, self).__init__()
        self.input_dim = input_dim
        self.model_dim = model_dim
        self.output_dim = output_dim
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.dim_feedforward = dim_feedforward
        self.n_layers = n_layers
        self.head_norm = head_norm
        self.dropout = dropout
        self.hidden_dim_dino_head = hidden_dim_dino_head
        self.bottleneck_dim_dino_head =bottleneck_dim_dino_head
        self.norm_last_layer = norm_last_layer
        self.pos_encoding = pos_encoding
        self.use_mask = use_mask
        self.num_classes = 4 #Just background classes for AD Delphes
        self.mode = mode

        #encoder part from pytorch
        self.embedding = nn.Linear(input_dim, model_dim)
        #Get a pre-norm tranformer encoder from pytorch
        self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(model_dim, n_heads, dim_feedforward=dim_feedforward, dropout=dropout, norm_first=True), n_layers)
        self.dino_head = DINOHead(in_dim=self.embed_dim, out_dim=self.output_dim, hidden_dim=self.hidden_dim_dino_head, bottleneck_dim=self.bottleneck_dim_dino_head, norm_last_layer=self.norm_last_layer)
        #Define [CLS] token for aggregate result where head attaches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, model_dim))
        #Add a classification head (just a linear layer)
        #self.classification_head = nn.Linear(embed_dim, self.num_classes)
        #If pos_encoding = True, create and apply the positional encoding to the input
        self.pos_encoder = PositionalEncoding(self.model_dim, self.dropout)
        #In order to downsize to a smaller embedding without lowering the model_dim, append an MLP to the CLS token
        self.downsize = MLP(input_dim=model_dim, embed_dim=embed_dim, hidden_channels=[32])
    
    #Embedding output without the DINO head
    def representation(self, x, mask=None):
        if self.use_mask==True: #Mask zero padded pT (no particles) because of the softmax in the attention mechanism
            mask = self.make_mask(x, mode=self.mode)
        self.batch_size = x.shape[0]
        #Create embedding for tranformer input
        x = self.embedding(x)
        #Add CLS token
        if self.mode=='cls':
            cls_tokens = self.cls_token.expand(self.batch_size, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
        #Transpose (bsz, n_const, model_dim) -> (n_const, bsz, model_dim) for transformer input
        x = torch.transpose(x,0,1)
        #If pos_encoding = True, apply positional encoding
        if self.pos_encoding:
            x = self.pos_encoder(x * math.sqrt(self.model_dim))
        #Input to transformer encoder
        x = self.transformer(x, mask=mask)
        #Only take [CLS] token and input into DINO head
        if self.mode == 'cls':
            x = x[0,...]
        elif self.mode == 'avg':
            x = x.mean(dim=0)
        elif self.mode == 'max':
            x = x.max(dim=0)[0]
        #If downsize to lower dimension run throught the MLP
        x = self.downsize(x)
        return x
    
    #Adding a classification head to the embedding (for pot. sup. cross entropy)
    """ def classify(self, x):
        self.representation(x)
        self.classification_head(x) """

    #For now the implementation does not use masking like in JetCLR, might be revisited later.
    def forward(self, x):
        x = self.representation(x)
        x = self.dino_head(x)
        return x
    
    def make_mask(self, x, mode):
        ''' Input: x w/ shape (bsz, n_const=19, n_feat=3)
            output: mask w/ pT zero (no particles) masked for attention w/ shape (bsz*n_heads, n_const, n_const)
                    where 0 is not masked and -np.inf is masked
        '''
        n_const = x.shape[1]

        pT_zero = x[:,:,0] == 0 #Checks where pT is zero -> no particles/jets
        pT_zero = torch.repeat_interleave(pT_zero, self.n_heads, axis=0)
        pT_zero = torch.repeat_interleave(pT_zero[:,None], n_const, axis=1)

        mask = torch.zeros(pT_zero.size(0), n_const, n_const, device=x.device)
        mask[pT_zero] = -np.inf
        if mode=='cls':
            #Add extra const for CLS token, not masked (0)
            mask_with_cls = torch.zeros(pT_zero.size(0), n_const+1, n_const+1, device=x.device)
            mask_with_cls[:,1:,1:] = mask
        else:
            mask_with_cls = mask
        return mask_with_cls


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
    
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 1000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        self.max_len = max_len

        position = torch.arange(self.max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2) * (-math.log(10000.0) / self.d_model))
        pe = torch.zeros(self.max_len, 1, self.d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
    
    def visualize(self):
        plt.imshow(self.pe.reshape(-1,self.d_model), aspect="auto")
        plt.title("Positional Encoding")
        plt.xlabel("Encoding Dimension")
        plt.ylabel("Position Index")

        # set the tick marks for the axes
        if self.d_model < 10:
            plt.xticks(torch.arange(0, self.d_model))
    
        plt.colorbar()
        plt.show()

class MLP(torch.nn.Module):
    '''
    Define MLP in order to downsize embedding from transformer model_dim (input_dim) to embed_dim. 
    Don't use batchnorm normalization as it's also not used in DINO head.
    '''
    def __init__(self, input_dim = 64, embed_dim = 6, hidden_channels=None, act_layer=nn.ReLU, bias=True):
        super(MLP, self).__init__()
        layers = []
        in_dim = input_dim
        for hidden_dim in hidden_channels:
            layers.append(nn.Linear(in_dim, hidden_dim, bias=bias))
            layers.append(act_layer())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, embed_dim, bias=bias))
        
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class Identity(torch.nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        self.identity = nn.Identity()
    def representation(self, x):
        return self.identity(x)
    def forward(self, x):
        return self.representation(x)
    
if __name__ == "__main__":
    transformer_args_standard = dict(
        input_dim=3, 
        model_dim=64, 
        output_dim=64, 
        embed_dim=12,
        n_heads=8, 
        dim_feedforward=256, 
        n_layers=4,
        hidden_dim_dino_head=256,
        bottleneck_dim_dino_head=64,
        pos_encoding = True,
        use_mask = False,
    )
    transformer_args_jetclass = dict(
        input_dim=4, 
        model_dim=128, 
        output_dim=64,
        embed_dim=6,   #Only change embed_dim without describing new transformer architecture
        n_heads=8, 
        dim_feedforward=256, 
        n_layers=4,
        hidden_dim_dino_head=256,
        bottleneck_dim_dino_head=64,
        pos_encoding = True,
        use_mask = True,
        mode='cls',
    )
    #TransformerEncoder(**transformer_args_standard)
    plot_pos_encoding = PositionalEncoding(d_model=64)
    plot_pos_encoding.visualize()

