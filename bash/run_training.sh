#!/bin/bash

if [ -d "/workspace/students/anhtu/transformers-text-recognition/venv/" ]; then
  source /workspace/students/anhtu/transformers-text-recognition/venv/bin/activate
else
  python3 -m venv /workspace/students/anhtu/transformers-text-recognition/venv
  source /workspace/students/anhtu/transformers-text-recognition/venv/bin/activate
  pip install -r /workspace/students/anhtu/transformers-text-recognition/requirements.txt
  python -m spacy download en_core_web_sm
  python -m spacy download de_core_news_sm
fi
python /workspace/students/anhtu/transformers-text-recognition/training.py
deactivate