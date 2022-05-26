import torchvision
import torch
from transformer import Encoder, Decoder, Seq2Seq, Seq2SeqTrgSameSrc, Seq2SeqWithAllFeature
from transformer_without_trg import Decoder as DecoderNoTrg, Seq2Seq as Seq2SeqNoTrg, Seq2SeqWithoutDecoder
from constants import MODEL_TYPE
import string


def load_model(transformer_model_type, feature_model_type, config, device):
    if feature_model_type == 'vgg16':
        feature_model = torchvision.models.vgg16_bn(pretrained=True).features
    else:
        raise NotImplementedError

    if transformer_model_type == MODEL_TYPE[0] or transformer_model_type == MODEL_TYPE[1]:
        enc = Encoder(256, config['HID_DIM'], config['ENC_LAYERS'], config['ENC_HEADS'], config['ENC_PF_DIM'],
                      config['ENC_DROPOUT'], device, 3 * 9)
        dec = Decoder(len(string.printable) + 2, config['HID_DIM'], config['DEC_LAYERS'], config['DEC_HEADS'],
                      config['DEC_PF_DIM'], config['DEC_DROPOUT'], device, config['OUTPUT_LEN'])
        model = Seq2Seq(enc, dec, 0, 0, device).to(device)
    elif transformer_model_type == MODEL_TYPE[2]:
        enc = Encoder(256, config['HID_DIM'], config['ENC_LAYERS'], config['ENC_HEADS'], config['ENC_PF_DIM'],
                      config['ENC_DROPOUT'], device, 3 * 9)
        dec = DecoderNoTrg(len(string.printable) + 2, config['HID_DIM'], config['DEC_LAYERS'], config['DEC_HEADS'],
                           config['DEC_PF_DIM'], config['DEC_DROPOUT'], device)
        model = Seq2SeqNoTrg(enc, dec, 0, config['OUTPUT_LEN'], 512*3*9, device).to(device)
    elif transformer_model_type == MODEL_TYPE[3]:
        enc = Encoder(256, config['HID_DIM'], config['ENC_LAYERS'], config['ENC_HEADS'], config['ENC_PF_DIM'],
                      config['ENC_DROPOUT'], device, 3 * 9)
        model = Seq2SeqWithoutDecoder(enc, 0, config['HID_DIM'], len(string.printable) + 1, config['OUTPUT_LEN'], 512*3*9, device).to(device)
    elif transformer_model_type == MODEL_TYPE[4]:
        enc = Encoder(256, config['HID_DIM'], config['ENC_LAYERS'], config['ENC_HEADS'], config['ENC_PF_DIM'],
                      config['ENC_DROPOUT'], device, config['OUTPUT_LEN'])
        dec = Decoder(len(string.printable) + 2, config['HID_DIM'], config['DEC_LAYERS'], config['DEC_HEADS'],
                      config['DEC_PF_DIM'], config['DEC_DROPOUT'], device, config['OUTPUT_LEN'])
        model = Seq2SeqTrgSameSrc(enc, dec, 0, 0, len(string.printable) + 2, config['OUTPUT_LEN'], 512*3*9, device).to(device)
    elif transformer_model_type == MODEL_TYPE[5]:
        enc = Encoder(256, config['HID_DIM'], config['ENC_LAYERS'], config['ENC_HEADS'], config['ENC_PF_DIM'],
                      config['ENC_DROPOUT'], device, 3 * 9)
        dec = Decoder(len(string.printable) + 2, config['HID_DIM'], config['DEC_LAYERS'], config['DEC_HEADS'],
                      config['DEC_PF_DIM'], config['DEC_DROPOUT'], device, config['OUTPUT_LEN'])
        model = Seq2SeqWithAllFeature(enc, dec, 0, 0, 3*9, 3*9*512, device).to(device)
    else:
        raise NotImplementedError

    return model, feature_model.to(device)


def extract_feature(feature_model, inputs, device):
    feature = feature_model(inputs)
    feature = torch.sum(feature, dim=1)
    feature = feature.view(feature.shape[0], -1)
    feature -= feature.min(1, keepdim=True)[0]
    feature /= feature.max(1, keepdim=True)[0]
    feature *= 255
    feature = feature.type(torch.LongTensor)

    return feature.to(device)


def predict_sequence(feature, model, device, max_len):
    model.eval()

    src_mask = model.make_src_mask(feature)
    with torch.no_grad():
        enc_src = model.encoder(feature, src_mask)

    trg_indexes = [1]
    for i in range(max_len):
        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)
        trg_mask = model.make_trg_mask(trg_tensor)

        with torch.no_grad():
            output, attention = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)

        pred_token = output.argmax(2)[:, -1].item()
        # if i == 0:
        #     trg_indexes[0] = pred_token
        trg_indexes.append(pred_token)
        # trg_indexes[i+1] = pred_token
        if pred_token == 0:
            break

    vocab = string.printable
    output_tokens = []
    for i in trg_indexes:
        if i > 2:
            output_tokens.append(vocab[i - 2])
    return output_tokens


def predict(feature, model, device, max_len):
    model.eval()

    src_mask = model.make_src_mask(feature)
    with torch.no_grad():
        enc_src = model.encoder(feature, src_mask)

    vocab = string.printable

    trg_indexes = torch.randint(1, len(vocab)+1, (max_len,))
    trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)
    trg_mask = model.make_trg_mask(trg_tensor)

    with torch.no_grad():
        output, attention = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)

    output_dim = output.shape[-1]
    output = output.contiguous().view(-1, output_dim)
    pred = output.max(1)[1]
    output_tokens = []
    for i in pred:
        if i > 0:
            output_tokens.append(vocab[i - 1])
    return output_tokens
