# transformers-text-recognition
<p align = "center">
<img src = "https://i.imgur.com/Mq09gYR.png">
</p>
<p align = "center">
Architecture of transformer-text-recognition model
</p>
This project will try to apply transformer to recognize the text from image. The input
of model is a image and the output of the model is word taken from image. The input image feature is
extracted by convolution network and then the extracted feature is used as a input sentence
to train transformer model to translate image to text.

## How to run

- Download dataset from <a href='https://drive.google.com/file/d/16K4dSaj99JwUzqX2l6roBVV9bYixUm3-/view?usp=sharing'>here</a>
- The trained models can be downloaded from  <a href='https://drive.google.com/file/d/15kHdvRib2W1bstgbxCCedB6FKiou7n0B/view?usp=sharing'>here</a>

### Install 

    #python3.7
    pip install --upgrade pip
    pip install -r requirements.txt

### Demo
    
    python run_demo_server.py --port PORT --model_folder FOLDER_PATH

- `PORT`: port to run server (default server will run on http://localhost:9595)
- `model_folder`: folder store trained model 

### Training
    
    python training.py --model_type MODEL_TYPE

- `model_type`:
  + `1`: transformer-random-trg
  + `2`: transformer-no-trg
  + `3`: transformer-no-decoder
  + `4`: transformer-trg-same-src 
  + `5`: transformer
- The training model will be saved to `./checkpoints/{model_type}.pt`

### Eval
    
    python evaluate.py --model_type MODEL_TYPE

- `model_type`:
  + `1`: transformer-random-trg
  + `2`: transformer-no-trg
  + `3`: transformer-no-decoder
  + `4`: transformer-trg-same-src 
  + `5`: transformer