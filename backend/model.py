import pickle
import sys

def load_model(file_path):
    with open(file_path, 'rb') as f:
        model = pickle.load(f)
    print(f"Model loaded from {file_path}", file=sys.stderr)
    return model