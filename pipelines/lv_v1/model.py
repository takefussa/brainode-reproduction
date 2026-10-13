import torch.nn as nn
import torch

class ConditionalLongiFlowODE(nn.Module):
    def __init__(self, latent_dim, condition_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + 1 + condition_dim, 256),  # V + t + c
            nn.LayerNorm(normalized_shape=256),               
            nn.GELU(),
            nn.Linear(256, 512),
            nn.LayerNorm(normalized_shape=512),              
            nn.GELU(),
            nn.Linear(512, 1024),
            nn.LayerNorm(normalized_shape=1024),             
            nn.GELU(),
            nn.Linear(1024, latent_dim)
        )
        self.condition_dim = condition_dim
        
    def forward(self, t, V, c):
        
        c = c.unsqueeze(1)
        # print(t.shape, V.shape, c.shape, '!!!!!')
        if t.dim() == 0:
            t = t.unsqueeze(0)
        t = t.view(-1, 1).repeat(V.shape[0], 1)
        c = c.view(-1, self.condition_dim).repeat(V.shape[0], 1)
        input = torch.cat([V, t, c], dim=-1)
        dVdt = self.net(input)
        return dVdt
class ODEFunc(nn.Module):
    def __init__(self, latent_dim, condition_dim):
        super().__init__()
        input_dim = latent_dim + 1 + condition_dim
        self.layer1 = nn.Linear(input_dim, 256)
        self.layer2 = nn.Linear(256, 512)
        self.layer3 = nn.Linear(512, 512)
        self.layer4 = nn.Linear(512, latent_dim)
        self.gelu = nn.GELU()
        # Residual layer
        self.residual = nn.Linear(input_dim, latent_dim)  
        self.condition_dim = condition_dim

    def forward(self, t, V, c):
        c = c.unsqueeze(1)
        # print(t.shape, V.shape, c.shape, '!!!!!')
        c = c.unsqueeze(1)
        # print(t.shape, V.shape, c.shape, '!!!!!')
        if t.dim() == 0:
            t = t.unsqueeze(0)
        t = t.view(-1, 1).repeat(V.shape[0], 1)
        c = c.view(-1, self.condition_dim).repeat(V.shape[0], 1)
        x = torch.cat([V, t, c], dim=-1)
        h = self.gelu(self.layer1(x))
        h = self.gelu(self.layer2(h))
        # h = self.gelu(self.layer3(h))
        out = self.layer4(h)
        residual = self.residual(x)  
        return out + residual
    
class ODEFuncWithAttention(nn.Module):
    def __init__(self, latent_dim, condition_dim):
        super().__init__()
        input_dim = latent_dim + 1 + condition_dim
        self.query = nn.Linear(input_dim, 256)
        self.key = nn.Linear(input_dim, 256)
        self.value = nn.Linear(input_dim, 256)
        self.scale = 256 ** -0.5  # scaling
        self.fc1 = nn.Linear(256, 512)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(512, latent_dim)
        self.condition_dim = condition_dim

    def forward(self, t, V, c):
        c = c.unsqueeze(1)
        # print(t.shape, V.shape, c.shape, '!!!!!')
        if t.dim() == 0:
            t = t.unsqueeze(0)
        t = t.view(-1, 1).repeat(V.shape[0], 1)
        c = c.view(-1, self.condition_dim).repeat(V.shape[0], 1)
        x = torch.cat([V, t, c], dim=-1)
        Q = self.query(x)  # [batch, 256]
        K = self.key(x)    # [batch, 256]
        V = self.value(x)  # [batch, 256]
        attn = torch.softmax(Q @ K.T * self.scale, dim=-1)  # Self-Attention
        h = attn @ V  # [batch, 256]
        h = self.gelu(self.fc1(h))
        return self.fc2(h)
    