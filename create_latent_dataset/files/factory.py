# V1

import os
import time

import numpy as np
import h5py
import torch
import torchvision
import torch.utils.data
from torchvision.transforms import InterpolationMode


DATASET_PARTS_FOLDER = "create_latent_dataset/data/original"


class GenerativeDataset(torch.utils.data.Dataset):
    """
    Клас, що реалізує генерацію датасету штучних зображень за допомогою генеративної моделі.
    Для кожного елемента створюється латентний вектор та відповідне зображення.
    """
    def __init__(self, generator: torch.nn.Module, length: int, device: torch.device | str):
        self._length = length
        self._device = device
        self._generator = generator

        # Ініціалізація seed для детермінованого відтворення латентних векторів
        self._seeds = np.random.randint(low=0, high=2_000_000_000, size=self._length)

        # Комбінація перетворень для приведення зображень до відповідного формату та масштабування
        self._transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize(size=(224, 224), interpolation=InterpolationMode.BICUBIC),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    @property
    def z_dim(self):
        """
        Отримати розмірність латентного простору генератора
        :return: int
            Розмірність латентного вектору генератора
        """
        return self._generator.z_dim

    def __len__(self) -> int:
        """
        Повертає кількість елементів у датасеті
        :return: int
           Довжина датасету
        """
        return self._length

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Повертає пару: латентний вектор та відповідне синтезоване зображення (тензор)
        :param item: int
            Індекс вибірки
        :return: tuple[torch.Tensor, torch.Tensor]
            Латентний вектор та трансформоване зображення
        """
        # Генерація латентного вектора з нормального розподілу
        seed = self._seeds[item]
        vector = torch.from_numpy(np.random.RandomState(seed).randn(1, self.z_dim)).float()

        # Генерація зображення генератором у режимі без обрахування градієнтів
        raw_image = self._generate_image(latent_vector=vector.to(self._device))

        # Перетворення на стандартний вхід для класифікатора
        image = self._transform(img=raw_image)
        vector = vector[0]
        return vector, image

    def _generate_image(self, latent_vector: torch.Tensor,
                        truncation_psi: float = 1.0, noise_mode: str = "const") -> np.ndarray:
        """
        Генерує одне зображення з латентним вектором за допомогою генератора
        :param latent_vector: torch.Tensor
            Латентний вектор для генерації зображення
        :param truncation_psi: float
            Параметр тримування генератора
        :param noise_mode: str
            Параметр конкретизації режиму шуму
        :return: np.ndarray
            Масив зображення у форматі HWC, uint8
        """
        # Заздалегідь ініціалізується вектор класу для генератора (усі нулі)
        label = torch.zeros([1, self._generator.c_dim], device=self._device)
        with torch.no_grad():
            # Генерація зображення, зміна порядку осей на (H, W, C), денормалізація та квантизація
            img = self._generator(latent_vector, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
            img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            return img[0].cpu().numpy()


class Factory:
    """
    Клас для створення наборів синтезованих латентних векторів та міток (класів)
    із поділом на частини для подальшого використання
    """
    def __init__(self, generator: torch.nn.Module, classificator: torch.nn.Module, num_classes: int,
                 dataset_part_number: int, dataset_part_size: int, classificator_batch_size: int,
                 device: torch.device | str):
        self._generator = generator
        self._classificator = classificator
        self._num_classes = num_classes

        self._dataset_part_number = dataset_part_number
        self._dataset_part_size = dataset_part_size
        self._classificator_batch_size = classificator_batch_size

        self._device = device
        self._dir_name = "/"

    def create_parts(self, shuffle: bool = True, drop_last: bool = False):
        print("Start")

        save_path = os.path.join(self._dir_name, DATASET_PARTS_FOLDER)
        os.makedirs(save_path, exist_ok=True)

        # Прохід по всіх частинах датасету
        for part in range(self._dataset_part_number):
            start = time.time()

            # Вибірка для поточної частини із новими seed
            dataset = GenerativeDataset(
                generator=self._generator, length=self._dataset_part_size, device=self._device
            )
            loader = torch.utils.data.DataLoader(
                dataset=dataset, shuffle=shuffle, drop_last=drop_last, batch_size=self._classificator_batch_size
            )

            # Заздалегідь ініціалізуються масиви для акумуляції латентних векторів та вірогідностей класів
            output_vectors = np.zeros((len(dataset), dataset.z_dim), dtype="float32")
            output_labels = np.zeros((len(dataset), self._num_classes), dtype="float32")

            # Обробка даних в режимі без обрахунку градієнтів
            counter = 0
            with torch.no_grad():
                for data_vector, data_image in loader:
                    batch_size = data_vector.size(0)
                    data_image = data_image.to(self._device)

                    # Прогін синтетичних зображень через класифікатор для отримання ймовірностей класів
                    probs = self._classificator(data_image)

                    # Збереження відповідних батчів у загальний масив
                    output_vectors[counter:counter + batch_size, :] = data_vector.cpu().numpy()
                    output_labels[counter:counter + batch_size, :] = probs.cpu().numpy()

                    counter += batch_size

            # Створення директорії для поточної частини і збереження масивів у файл формату HDF5
            save_path__part = os.path.join(save_path, f"{part}")
            os.makedirs(save_path__part, exist_ok=True)

            # Збереження у файл HDF5
            h5_path = os.path.join(save_path__part, "data.h5")
            with h5py.File(h5_path, "w") as f:
                f.create_dataset("vectors", data=output_vectors, compression="gzip", compression_opts=3)
                f.create_dataset("labels", data=output_labels, compression="gzip", compression_opts=3)

            print(f"{part + 1} / {self._dataset_part_number}: {time.time() - start} s")
