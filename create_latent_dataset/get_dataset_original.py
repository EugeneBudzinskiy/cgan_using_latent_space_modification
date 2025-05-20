import torch
import torch.utils.data

from generator.files import legacy
from classificator.files.train import Classificator
from create_latent_dataset.files.factory import Factory


import torch_directml
DEVICE = torch_directml.device()
# DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


CLASSIFICATOR_MODEL_PATH = "classificator/files/checkpoint/model_at_0013.pth"
GENERATOR_MODEL_PATH = "generator/files/training-runs/network-snapshot.pkl"

LABELS_PATH = "dataset/artbench256-60k-split/labels.json"

NUM_CLASSES = 10
DATASET_PART_NUMBER = 350
DATASET_PART_SIZE = 1000
CLASSIFICATOR_BATCH_SIZE = 10


def load_generator(network_pkl: str, device: torch.device | str) -> torch.nn.Module:
    """
    Завантажує попередньо натреновану генеративну модель з pickle-файлу для подальшої генерації зображень
    :param network_pkl: str
       Шлях до pkl-файлу згенерованої мережі
    :param device: torch.device | str
       Обчислювальний пристрій, на який слід перенести генератор
    :return: torch.nn.Module
       Об'єкт генератора у режимі eval()
    """
    with open(network_pkl, mode="rb") as f:
        generator = legacy.load_network_pkl(f)['G_ema'].to(device)
    return generator.eval()


def load_classificator(model_path: str, num_classes: int, device: torch.device | str) -> torch.nn.Module:
    """
    Завантажує попередньо натреновану модель класифікатора для роботи на заданому пристрої
    :param model_path: str
        Шлях до збереженої state_dict моделі класифікатора
    :param num_classes: int
        Кількість класів (структура останнього шару моделі)
    :param device: torch.device | str
        Обчислювальний пристрій (cpu, cuda, DirectML)
    :return: torch.nn.Module
        Об'єкт класифікатора у режимі eval()
    """
    classificator = Classificator(num_classes=num_classes)
    classificator.to(device)

    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    classificator.load_state_dict(state_dict=state_dict, strict=True)
    return classificator.eval()


def main():
    generator = load_generator(network_pkl=GENERATOR_MODEL_PATH, device=DEVICE)
    classificator = load_classificator(model_path=CLASSIFICATOR_MODEL_PATH, num_classes=NUM_CLASSES, device=DEVICE)

    factory = Factory(
        generator=generator,
        classificator=classificator,
        num_classes=NUM_CLASSES,
        dataset_part_number=DATASET_PART_NUMBER,
        dataset_part_size=DATASET_PART_SIZE,
        classificator_batch_size=CLASSIFICATOR_BATCH_SIZE,
        device=DEVICE
    )

    factory.create_parts()


if __name__ == '__main__':
    main()
