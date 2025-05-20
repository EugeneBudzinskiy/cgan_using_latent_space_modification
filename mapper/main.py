# V7

from mapper.files import train

import torch_directml
DEVICE = torch_directml.device()
# import torch
# DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


TRAIN_DATASET_PATH = "dataset/latent_dataset/train.h5"
TEST_DATASET_PATH = "dataset/latent_dataset/test.h5"

NUM_CLASSES = 10
Z_DIM = 256

BATCH_SIZE = 512
EPOCHS = 300


def main():
    train.start_train(
        train_dataset_path=TRAIN_DATASET_PATH,
        test_dataset_path=TEST_DATASET_PATH,
        num_classes=NUM_CLASSES,
        z_dim=Z_DIM,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        device=DEVICE
    )


if __name__ == '__main__':
    main()
