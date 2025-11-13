import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=64):  # 64 squares on chess board
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class ChessTransformerBlock(nn.Module):
    def __init__(self, d_model=512, nhead=8, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_output, _ = self.self_attn(x, x, x)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)
        
        ff_output = self.feed_forward(x)
        x = x + self.dropout(ff_output)
        x = self.norm2(x)
        return x

class ChessTransformer(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_layers=6, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        
        # Input embedding for chess pieces (12 piece types: 6 pieces × 2 colors)
        self.piece_embedding = nn.Linear(12, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            ChessTransformerBlock(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        
        # Policy head (predicting moves)
        self.policy_head = nn.Sequential(
            nn.Linear(d_model * 64, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 4672)  # Maximum possible moves in chess
        )
        
        # Value head (evaluating positions)
        self.value_head = nn.Sequential(
            nn.Linear(d_model * 64, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 1),
            nn.Tanh()
        )

    def forward(self, x):
        # x shape: (batch_size, 64, 12) - 64 squares, 12 piece channels
        batch_size = x.size(0)
        
        # Embed pieces
        x = self.piece_embedding(x)  # (batch_size, 64, d_model)
        
        # Add positional encoding
        x = self.positional_encoding(x)
        
        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x)
        
        # Flatten for heads
        x_flat = x.view(batch_size, -1)
        
        # Get policy and value outputs
        policy_output = self.policy_head(x_flat)  # Move probabilities
        value_output = self.value_head(x_flat)    # Position evaluation
        
        return policy_output, value_output

def create_chess_transformer():
    model = ChessTransformer(
        d_model=512,
        nhead=8,
        num_layers=6,
        dim_feedforward=2048,
        dropout=0.1
    )
    return model
