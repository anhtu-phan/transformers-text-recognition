import torch
import torch.nn as nn
import os
import wandb
import string
import argparse
from tqdm import tqdm
from loss import cal_performance
from load_image_data import get_data
from model import load_model, extract_feature
from optim import SchedulerOptim
from constants import MODEL_TYPE

vocab = string.printable


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)


def get_output_transformer(model, feature_model, inputs, targets, device):
    feature = extract_feature(feature_model, inputs, device)
    output, _ = model(feature, targets)
    output_dim = output.shape[-1]
    output = output.contiguous().view(-1, output_dim)
    return output


def get_output_transformer_with_all_feature(model, feature_model, inputs, targets, device):
    feature = feature_model(inputs)
    feature = feature.view(feature.shape[0], -1).to(device)
    output, _ = model(feature, targets)
    output_dim = output.shape[-1]
    output = output.contiguous().view(-1, output_dim)
    return output


def get_output_transformer_trg_same_src(model, feature_model, inputs, device):
    feature = feature_model(inputs)
    feature = feature.view(feature.shape[0], -1).to(device)
    output, _ = model(feature, feature)
    output_dim = output.shape[-1]
    output = output.contiguous().view(-1, output_dim)
    return output


def get_output_transformer_random_trg(model, feature_model, inputs, targets, device):
    feature = extract_feature(feature_model, inputs, device)
    trg_indexes = torch.LongTensor(torch.randint(1, len(vocab) + 1, targets.shape)).to(device)
    output, _ = model(feature, trg_indexes)
    output_dim = output.shape[-1]
    output = output.contiguous().view(-1, output_dim)

    return output


def get_output_transformer_no_trg(model, feature_model, inputs, device):
    feature = feature_model(inputs)
    feature = feature.view(feature.shape[0], -1).to(device)
    output = model(feature)
    output_dim = output.shape[-1]
    output = output.contiguous().view(-1, output_dim)

    return output


def get_output(model, feature_model, inputs, targets, device, model_type):
    if model_type == MODEL_TYPE[0]:
        output = get_output_transformer(model, feature_model, inputs, targets, device)
    elif model_type == MODEL_TYPE[1]:
        output = get_output_transformer_random_trg(model, feature_model, inputs, targets, device)
    elif model_type == MODEL_TYPE[2] or model_type == MODEL_TYPE[3]:
        output = get_output_transformer_no_trg(model, feature_model, inputs, device)
    elif model_type == MODEL_TYPE[4]:
        output = get_output_transformer_trg_same_src(model, feature_model, inputs, device)
    elif model_type == MODEL_TYPE[5]:
        output = get_output_transformer_with_all_feature(model, feature_model, inputs, targets, device)
    else:
        raise NotImplementedError
    return output


def train(model, feature_model, data_loader, optimizer, device, model_type):
    model.train()
    epoch_loss, epoch_total_word, epoch_n_word_correct = 0, 0, 0
    with tqdm(total=len(data_loader)) as pbar:
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            output = get_output(model, feature_model, inputs, targets[:, :-1], device, model_type)
            targets = targets[:, 1:].contiguous().view(-1)

            loss, n_correct, n_word = cal_performance(output, targets, 0, True, 0.1)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_total_word += n_word
            epoch_n_word_correct += n_correct
            pbar.update(1)

    loss_per_word = epoch_loss/epoch_total_word
    acc = epoch_n_word_correct/epoch_total_word

    return epoch_loss / len(data_loader), loss_per_word, acc


def evaluate(model, feature_model, data_loader, device, model_type):
    model.eval()
    epoch_loss, epoch_total_word, epoch_n_word_correct = 0, 0, 0
    with torch.no_grad():
        with tqdm(total=len(data_loader)) as pbar:
            for batch_idx, (inputs, targets) in enumerate(data_loader):
                inputs = inputs.to(device)
                targets = targets.to(device)

                output = get_output(model, feature_model, inputs, targets[:, :-1], device, model_type)
                targets = targets[:, 1:].contiguous().view(-1)

                loss, n_correct, n_word = cal_performance(output, targets, 0, True, 0.1)
                epoch_loss += loss.item()
                epoch_total_word += n_word
                epoch_n_word_correct += n_correct
                pbar.update(1)

    loss_per_word = epoch_loss / epoch_total_word
    acc = epoch_n_word_correct / epoch_total_word

    return epoch_loss / len(data_loader), loss_per_word, acc


def main(model_type):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _model, _feature_model = load_model(model_type, "vgg16", CONFIG, device)
    print(f"{'-' * 10}number of parameters = {count_parameters(_model)}{'-' * 10}\n")
    model_name = f'{model_type}.pt'
    wandb_name = f'{model_type}'
    saved_model_dir = './checkpoints/'
    saved_model_path = saved_model_dir + model_name
    best_valid_acc = float('inf')*-1
    saved_epoch = 0

    if not os.path.exists(saved_model_dir):
        os.makedirs(saved_model_dir)
    if os.path.exists(saved_model_path):
        print(f"Load saved model {'.' * 10}\n")
        last_checkpoint = torch.load(saved_model_path, map_location=torch.device(device))
        best_valid_acc = last_checkpoint['best_valid_acc']
        saved_epoch = last_checkpoint['epoch']
        _model.load_state_dict(last_checkpoint['state_dict'])
        CONFIG['LEARNING_RATE'] = last_checkpoint['lr']
        wandb.init(name=wandb_name, project="transformer-text-recognition", config=CONFIG,
                   resume=True)
    else:
        _model.apply(initialize_weights)
        wandb.init(name=wandb_name, project="transformer-text-recognition", config=CONFIG,
                   resume=False)

    _optimizer = SchedulerOptim(torch.optim.Adam(_model.parameters(), lr=CONFIG['LEARNING_RATE'], betas=(0.9, 0.98),
                                                 weight_decay=0.0001), 1, CONFIG['HID_DIM'], 4000, 5e-4, saved_epoch)

    wandb.watch(_model, log='all')

    train_loader, val_loader = get_data(CONFIG['BATCH_SIZE'], CONFIG['OUTPUT_LEN'])

    for epoch in tqdm(range(saved_epoch, CONFIG['N_EPOCHS'])):
        logs = dict()

        train_lr = _optimizer.optimizer.param_groups[0]['lr']
        logs['train_lr'] = train_lr
        train_loss, train_loss_per_word, train_acc = train(_model, _feature_model, train_loader, _optimizer, device, model_type)
        val_loss, val_loss_per_word, val_acc = evaluate(_model, _feature_model, val_loader, device, model_type)

        logs['train_loss'] = train_loss
        logs['val_loss'] = val_loss
        logs['train_acc'] = train_acc
        logs['val_acc'] = val_acc
        logs['train_loss_per_word'] = train_loss_per_word
        logs['val_loss_per_word'] = val_loss_per_word

        if val_acc > best_valid_acc:
            best_valid_acc = val_acc
            checkpoint = {
                'epoch': epoch+1,
                'state_dict': _model.state_dict(),
                'best_valid_acc': best_valid_acc,
                'lr': train_lr,
            }
            torch.save(checkpoint, saved_model_path)

        wandb.log(logs, step=epoch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Transformer text recognition")
    parser.add_argument("--model_type", type=int)
    args = parser.parse_args()
    _model_type = MODEL_TYPE[args.model_type]

    CONFIG = {
        'OUTPUT_LEN': 20,
        "LEARNING_RATE": 1e-4,
        "BATCH_SIZE": 32,
        "HID_DIM": 512,
        "ENC_LAYERS": 6,
        "DEC_LAYERS": 6,
        "ENC_HEADS": 4,
        "DEC_HEADS": 4,
        "ENC_PF_DIM": 1024,
        "DEC_PF_DIM": 1024,
        "ENC_DROPOUT": 0.2,
        "DEC_DROPOUT": 0.2,
        "N_EPOCHS": 1000000,
        "CLIP": 1
    }
    main(_model_type)
