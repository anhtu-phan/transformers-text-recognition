import torch
import torchvision
import torch.nn as nn
import os
# import wandb
from tqdm import tqdm
from transformer import Encoder, Decoder, Seq2Seq
from optim import SchedulerOptim
from load_image_data import get_data


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)


def train(model, feature_model, data_loader, optimizer, trg_pad_idx, device):
    model.train()
    epoch_loss, epoch_acc = 0, 0
    for batch_idx, (inputs, targets) in enumerate(data_loader):
        inputs = inputs.to(device)
        # targets = targets.to(device)

        feature = feature_model(inputs).features[-1]
        output = model(inputs, targets)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    enc = Encoder(CONFIG['IMG_INPUT_HEIGHT']*CONFIG['IMG_INPUT_WIDTH'], CONFIG['HID_DIM'], CONFIG['ENC_LAYERS'],
                  CONFIG['ENC_HEADS'], CONFIG['ENC_PF_DIM'], CONFIG['ENC_DROPOUT'], device)
    dec = Decoder(CONFIG['OUTPUT_LEN'], CONFIG['HID_DIM'], CONFIG['DEC_LAYERS'], CONFIG['DEC_HEADS'],
                  CONFIG['DEC_PF_DIM'], CONFIG['DEC_DROPOUT'], device)
    _model = Seq2Seq(enc, dec, -1, -1, device).to(device)

    print(f"{'-' * 10}number of parameters = {count_parameters(_model)}{'-' * 10}\n")
    model_name = 'transformer.pt'
    wandb_name = 'transformer-with-init'
    saved_model_dir = './checkpoints/'
    saved_model_path = saved_model_dir + model_name
    best_valid_acc = float('inf')
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
        # wandb.init(name=wandb_name, project="multi-domain-machine-translation", config=CONFIG,
        #            resume=True)
    else:
        _model.apply(initialize_weights)
        # wandb.init(name=wandb_name, project="multi-domain-machine-translation", config=CONFIG,
        #            resume=False)

    _optimizer = SchedulerOptim(torch.optim.Adam(_model.parameters(), lr=CONFIG['LEARNING_RATE'], betas=(0.9, 0.98),
                                                 weight_decay=0.0001), 1, CONFIG['HID_DIM'], 4000, 5e-4, saved_epoch)
    _feature_model = torchvision.models.vgg16_bn(pretrained=True)
    # wandb.watch(_model, log='all')

    train_loader, val_loader = get_data(CONFIG['BATCH_SIZE'])

    for epoch in tqdm(range(saved_epoch, CONFIG['N_EPOCHS'])):
        logs = dict()

        train_lr = _optimizer.optimizer.param_groups[0]['lr']
        logs['train_lr'] = train_lr
        train(_model, _feature_model, train_loader, _optimizer, -1, device)


if __name__ == '__main__':
    CONFIG = {
        'IMG_INPUT_WIDTH': 300,
        'IMG_INPUT_HEIGHT': 100,
        'OUTPUT_LEN': 10,
        "LEARNING_RATE": 1e-7,
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
    main()
