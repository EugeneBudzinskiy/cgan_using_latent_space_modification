# V7

from classificator.files import train


import torch_directml
DEVICE = torch_directml.device()
# DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main():
    train.start_train(
        train_dataset_path="data/artbench256-60k-split/train.zip",
        test_dataset_path="data/artbench256-60k-split/test.zip",
        labels_path="data/artbench256-60k-split/labels.json",
        num_classes=10,
        device=DEVICE,
        batch_size=32,
        epochs=20
    )


if __name__ == '__main__':
    main()
