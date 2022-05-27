import os
import torch
import cv2
from flask import Flask, request, render_template, redirect, url_for
from model import load_model, predict, extract_feature, predict_sequence
from torchvision import transforms
import time
import string
from constants import MODEL_TYPE

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def index_post():
    file = request.files['image']
    file_save_path = os.path.join('static/input_images', file.filename)
    file.save(file_save_path)
    img = cv2.imread(os.path.join('static/input_images', file.filename))
    img = cv2.resize(img, (300, 100))
    transform = list()
    transform.append(transforms.ToTensor())
    transform.append(transforms.Normalize(mean=[0.5], std=[0.5]))
    transform = transforms.Compose(transform)
    img = transform(img)
    img = img.unsqueeze(0)

    results = []
    feature = feature_model(img)
    for i_m, model in enumerate(models):
        if i_m == 0:
            continue
        elif i_m == 1:
            feature = extract_feature(feature_model, img, device)
            start_time = time.time()
            output = predict(feature, model, device, CONFIG['OUTPUT_LEN'])
            predict_time = time.time() - start_time
            results.append({'model_type': MODEL_TYPE[1], 'result': ''.join(output), 'time': f'{predict_time:.2f}'})
        elif i_m == 2 or i_m == 3 or i_m == 4:
            feature = feature.view(feature.shape[0], -1).to(device)
            start_time = time.time()
            if i_m == 4:
                output = model(feature, feature)
            else:
                output = model(feature)
            predict_time = time.time() - start_time
            preds = torch.topk(output, CONFIG['OUTPUT_LEN'])[1]
            vocab = string.printable
            output = []
            for i in preds:
                if i > 0:
                    output.append(vocab[i - 1])
            results.append({'model_type': MODEL_TYPE[0], 'result': ''.join(output), 'time': f'{predict_time:.2f}'})
        elif i_m == 5:
            feature = feature.view(feature.shape[0], -1).to(device)
            src = model.convert_src(feature).to(device)
            src -= src.min(1, keepdim=True)[0]
            src /= src.max(1, keepdim=True)[0]
            src *= 255
            src = src.type(torch.LongTensor).to(device)
            start_time = time.time()
            output = predict_sequence(src, model, device, CONFIG['OUTPUT_LEN'])
            predict_time = time.time() - start_time
            results.append({'model_type': MODEL_TYPE[0], 'result': ''.join(output), 'time': f'{predict_time:.2f}'})
        else:
            raise NotImplementedError
    return render_template('index.html', filename=file.filename, result=results)


@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename="input_images/"+filename), code=301)


if __name__ == '__main__':
    if not os.path.exists('static/input_images'):
        os.makedirs('static/input_images')
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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    models = []
    for model_type in MODEL_TYPE:
        model, feature_model = load_model(model_type, 'vgg16', CONFIG, device)
        model_path = f'./checkpoints/{model_type}.pt'
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        models.append(model)

    app.run('0.0.0.0', port=9595)
