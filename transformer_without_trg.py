import torch
import torch.nn as nn
from transformer import EncoderLayer


def make_src_mask(src, src_pad_idx):
    # src = [batch_size, src_len]
    src_mask = (src != src_pad_idx).unsqueeze(1).unsqueeze(2)

    return src_mask


class Decoder(nn.Module):
    def __init__(self,
                 output_dim,
                 hid_dim,
                 n_layers,
                 n_heads,
                 pf_dim,
                 dropout,
                 device
                 ):
        super().__init__()

        self.device = device

        self.layers = nn.ModuleList([EncoderLayer(hid_dim, n_heads, pf_dim, dropout, device) for _ in range(n_layers)])

        self.fc_out = nn.Linear(hid_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)

    def forward(self, enc_src, src_mask):

        for layer in self.layers:
            trg = layer(enc_src, src_mask)

        output = self.fc_out(trg)
        # output = [batch_size, trg_len, output_dim]

        return output


class Seq2Seq(nn.Module):
    def __init__(self,
                 encoder,
                 decoder,
                 src_pad_idx,
                 max_len,
                 feature_shape_size,
                 device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.device = device
        self.convert = nn.Linear(feature_shape_size, max_len).to(device)

    def forward(self, src):
        src = self.convert(src).to(self.device)
        src -= src.min(1, keepdim=True)[0]
        src /= src.max(1, keepdim=True)[0]
        src *= 255
        src = src.type(torch.LongTensor).to(self.device)

        src_mask = make_src_mask(src, self.src_pad_idx)

        enc_src = self.encoder(src, src_mask)

        output = self.decoder(enc_src, src_mask)

        return output


class Seq2SeqWithoutDecoder(nn.Module):
    def __init__(self,
                 encoder,
                 src_pad_idx,
                 hid_dim,
                 output_dim,
                 max_len,
                 feature_shape_size,
                 device):
        super().__init__()
        self.encoder = encoder
        self.src_pad_idx = src_pad_idx
        self.device = device
        self.convert = nn.Linear(feature_shape_size, max_len).to(device)
        self.fc_out = nn.Linear(hid_dim, output_dim)

    def forward(self, src):
        src = self.convert(src).to(self.device)
        src -= src.min(1, keepdim=True)[0]
        src /= src.max(1, keepdim=True)[0]
        src *= 255
        src = src.type(torch.LongTensor).to(self.device)

        src_mask = make_src_mask(src, self.src_pad_idx)

        enc_src = self.encoder(src, src_mask)
        output = self.fc_out(enc_src)
        return output
