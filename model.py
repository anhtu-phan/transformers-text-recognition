import torchvision
import torch
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


def extract_feature(feature_model, inputs, device):
    feature = feature_model(inputs)
    feature = torch.sum(feature, dim=1)
    feature = feature.view(feature.shape[0], -1)
    feature -= feature.min(1, keepdim=True)[0]
    feature /= feature.max(1, keepdim=True)[0]
    feature *= 255
    feature = feature.type(torch.LongTensor)

    return feature.to(device)


def predict(feature, model, device, max_len):
    model.eval()

    src_mask = model.make_src_mask(feature)
    with torch.no_grad():
        enc_src = model.encoder(feature, src_mask)

    trg_indexes = [0]
    for i in range(max_len):
        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)
        trg_mask = model.make_trg_mask(trg_tensor)

        with torch.no_grad():
            output, attention = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)

        pred_token = output.argmax(2)[:, -1].item()

        trg_indexes.append(pred_token)
        if pred_token == 0:
            break

    vocab = string.printable
    output_tokens = []
    for i in trg_indexes:
        if i > 0:
            output_tokens.append(vocab[i - 1])
    return output_tokens
