# V2

import torch

from encoder.files import train
from mapper.files.train import Mapper

import torch_directml
DEVICE = torch_directml.device()
# DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


MAPPER_MODEL_PATH = "mapper/files/checkpoint/model_at_0217.pth"

Z_DIM = 256
NUM_CLASSES = 10
NOISE_SIZE = 256

DATASET_TRAIN_LENGTH = 80_000
DATASET_TEST_LENGTH = 20_000

BATCH_SIZE = 512
EPOCHS = 100



def load_mapper(model_path: str, z_dim: int, num_classes: int, device: torch.device | str) -> torch.nn.Module:
    mapper = Mapper(z_dim=z_dim, num_classes=num_classes)
    mapper.to(device)

    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    mapper.load_state_dict(state_dict=state_dict, strict=True)
    return mapper.eval()


def main():
    mapper = load_mapper(model_path=MAPPER_MODEL_PATH, z_dim=Z_DIM, num_classes=NUM_CLASSES, device=DEVICE)

    train.start_train(
        mapper=mapper,
        noise_size=NOISE_SIZE,
        num_classes=NUM_CLASSES,
        z_dim=Z_DIM,
        dataset_train_length=DATASET_TRAIN_LENGTH,
        dataset_test_length=DATASET_TEST_LENGTH,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        device=DEVICE
    )


if __name__ == '__main__':
    main()
