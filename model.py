import torchvision
# from fast_transformers.builders import TransformerEncoderBuilder, TransformerDecoderBuilder
from transformer import Encoder, Decoder, Seq2Seq
# from linear_transformer import LinearTransformer
import string


def load_model(transformer_model_type, feature_model_type, config, device):
    if feature_model_type == 'vgg16':
        feature_model = torchvision.models.vgg16_bn(pretrained=True).features
    else:
        raise NotImplementedError

    if transformer_model_type == 'transformer':
        enc = Encoder(256, config['HID_DIM'], config['ENC_LAYERS'], config['ENC_HEADS'], config['ENC_PF_DIM'],
                      config['ENC_DROPOUT'], device, 3 * 9)
        dec = Decoder(len(string.printable) + 1, config['HID_DIM'], config['DEC_LAYERS'], config['DEC_HEADS'],
                      config['DEC_PF_DIM'], config['DEC_DROPOUT'], device, config['OUTPUT_LEN'])
        model = Seq2Seq(enc, dec, 0, 0, device).to(device)
    # elif transformer_model_type == 'linear-transformer':
    #     enc = TransformerEncoderBuilder.from_kwargs(n_layers=config['ENC_LAYERS'], n_heads=config['ENC_HEADS'],
    #                                                 feed_forward_dimensions=config['ENC_PF_DIM'],
    #                                                 query_dimensions=config['HID_DIM'],
    #                                                 value_dimensions=config['HID_DIM'], attention_type='linear',
    #                                                 dropout=config['ENC_DROPOUT']).get()
    #     dec = TransformerDecoderBuilder.from_kwargs(n_layers=config['DEC_LAYERS'], n_heads=config['DEC_HEADS'],
    #                                                 feed_forward_dimensions=config['DEC_PF_DIM'],
    #                                                 query_dimensions=config['HID_DIM'],
    #                                                 value_dimensions=config['HID_DIM'], self_attention_type='linear',
    #                                                 cross_attention_type='linear', dropout=config['ENC_DROPOUT']).get()
    #     model = LinearTransformer(enc, dec, device).to(device)
    else:
        raise NotImplementedError

    return model, feature_model
