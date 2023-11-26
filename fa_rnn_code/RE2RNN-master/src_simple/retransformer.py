from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch
import torch.nn as nn
import math

# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, max_len=5000):
#         super().__init__()
#         # Compute the positional encodings once in log space.
#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         pe = pe.unsqueeze(0).transpose(0, 1)
#         self.register_buffer('pe', pe)

#     def forward(self, x):
#         x = x + self.pe[:x.size(0), :]
#         return x

# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, max_len=5000):
#         super().__init__()
#         self.d_model = d_model
#         self.max_len = int(max_len)  # Ensure max_len is an integer

#         # Compute the positional encodings once in log space.
#         pe = torch.zeros(self.max_len, self.d_model)
#         position = torch.arange(0, self.max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * -(math.log(10000.0) / self.d_model))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         pe = pe.unsqueeze(0).transpose(0, 1)
#         self.register_buffer('pe', pe)

#     def forward(self, x):
#         x = x + self.pe[:x.size(0), :]
#         return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.max_len = int(max_len)
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(self.max_len, self.d_model)
        position = torch.arange(0, self.max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * -(math.log(10000.0) / self.d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        print(f"Input size: {x.size()}")
        print(f"Positional Encoding Initialized size: {self.pe.size()}")
        print(f"Positional Encoding size: {self.pe[:x.size(0), :].size()}")
        print(f"Sliced Positional Encoding size: {self.pe[:x.size(0), :].size()}")
        pe = self.pe[:x.size(0)]
        if pe.size(0) < x.size(0):
            # Repeat or extend the positional encoding if it's shorter than input
            repeat_factor = math.ceil(x.size(0) / pe.size(0))
            pe = pe.repeat(repeat_factor, 1, 1)[:x.size(0)]
        
        x = x + pe
        return self.dropout(x)

class TransformerModel(nn.Module):
    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, nclasses, dropout=0.5):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        ninp = ninp - 1
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        print(f"The ninp: {ninp} and nhead: {nhead}")
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)
        self.embedding = nn.Embedding(ntoken, ninp)
        self.classifier = nn.Linear(ninp, nclasses)

        self.init_weights()

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_mask):
        src = self.embedding(src) * math.sqrt(self.ninp)
        print(f"Size before positional encoding: {src.size()}")
        src = self.pos_encoder(src)
        print(f"Size after positional encoding: {src.size()}")
        if src_mask is None:
            """Generate a square causal mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
            """
            src_mask = nn.Transformer.generate_square_subsequent_mask(len(src)).to(torch.device)
        output = self.transformer_encoder(src, src_mask)
        # output = self.decoder(output) # maybe linear? self.linear(output)
        output = self.classifier(output.mean(dim=0))
        return output
