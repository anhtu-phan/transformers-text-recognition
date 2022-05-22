import torch
import torch.nn as nn
from fast_transformers.masking import TriangularCausalMask, LengthMask


class LinearTransformer(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg):
        src_mask = TriangularCausalMask(src.shape[1], device=self.device)
        length_mask = LengthMask(torch.rand((src.shape[1])).type(torch.IntTensor), device=self.device)
        enc_src = self.encoder(src, attn_mask=src_mask, length_mask=length_mask)
        trg_mask = TriangularCausalMask(trg.shape[1], device=self.device)
        output, attention = self.decoder(trg, enc_src, trg_mask, src_mask)

        return output, attention
