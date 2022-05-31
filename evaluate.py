import torch
from model import load_model
from load_image_data import get_test_data
from training import evaluate, model_type
from constants import MODEL_TYPE
import argparse


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, feature_model = load_model(model_type, 'vgg16', CONFIG, device)
    model_path = f'./checkpoints/{model_type}.pt'
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])

    test_loader = get_test_data(CONFIG['BATCH_SIZE'], CONFIG['OUTPUT_LEN'])
    _, _, test_acc = evaluate(model, feature_model, test_loader, device)
    print(test_acc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Transformer text recognition")
    parser.add_argument("--model_type", type=int)
    args = parser.parse_args()
    model_type = MODEL_TYPE[args.model_type]
    CONFIG = {
        'OUTPUT_LEN': 20,
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
