import os
import torch
import cv2
from flask import Flask, request, render_template, redirect, url_for
from model import load_model, predict, extract_feature, predict_sequence
from torchvision import transforms
import time

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

    feature = extract_feature(feature_model, img, device)
    start_time = time.time()
    output = predict_sequence(feature, model, device, CONFIG['OUTPUT_LEN'])
    predict_time = time.time() - start_time

    start_time = time.time()
    output2 = predict(feature, model2, device, CONFIG['OUTPUT_LEN'])
    predict_time2 = time.time() - start_time

    result = [{'model_type': 'transformer', 'result': ''.join(output), 'time': f'{predict_time:.2f}'},
              {'model_type': 'transformer2', 'result': ''.join(output2), 'time': f'{predict_time2:.2f}'}]
    return render_template('index.html', filename=file.filename, result=result)


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
    model_type = 'transformer'
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, feature_model = load_model(model_type, 'vgg16', CONFIG, device)
    model_path = f'./checkpoints/{model_type}-no-train-feature.pt'
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])

    model2, _ = load_model(model_type, 'vgg16', CONFIG, device)
    model2_path = f'./checkpoints/{model_type}.pt'
    checkpoint2 = torch.load(model2_path, map_location=device)
    model2.load_state_dict(checkpoint2['state_dict'])

    app.run('0.0.0.0', port=9595)
