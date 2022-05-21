import torch.nn as nn


class LinearTransformer(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg):
        enc_src = self.encoder(src)
        output, attention = self.decoder(trg, enc_src)

        return output, attention
