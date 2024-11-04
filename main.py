'''NOTE: Change 'config.py' based on the kind of facial parts you want to finetune the model (UNET) on before running 'main.py'.
'''

from train import train_model
from predict import predict_and_display_sample_image

if __name__ == "__main__":
    # Finetune the model on Lapa Dataset
    # train_model()

    # Predict mask for an image
    predict_and_display_sample_image("sample_image.jpg")