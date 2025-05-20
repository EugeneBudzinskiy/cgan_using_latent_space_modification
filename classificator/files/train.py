# V7

import os
import json
import zipfile

import numpy as np
import torch
import torch.utils.data
import torchvision
import tqdm
from PIL import Image


LOG_NAME = "log.txt"  # Ім'я файлу журналу
CHECkPOINT_FOLDER = "checkpoint"  # Ім'я папки для збереження контрольних точок (чекпоінтів)


class CustomDateset(torch.utils.data.Dataset):
    """
    Реалізація користувацького датасету для завдань машинного навчання на основі фреймворку PyTorch.
    Підтримується завантаження даних як із файлової системи, так і із ZIP-архівів,
    що забезпечує гнучкість у зберіганні та доступі до вихідних зображень.
    Передбачена можливість попередньої обробки зображень за допомогою типових перетворень.
    """
    def __init__(self, path: str, num_classes: int, labels_path: str):
        # Збереження вхідних параметрів для подальшого використання
        self._path = path
        self._num_classes = num_classes
        self._labels_path = labels_path

        # Ініціалізація конвеєра попередньої обробки зображень із використанням типових для ImageNet нормалізуючих параметрів
        self._transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.RandomCrop(size=(224, 224)),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self._type = None
        self._zipfile = None
        self._supported_types = ["dir", "zip"]  # Підтримувані режими зберігання даних

        # Визначення типу структури даних за вхідним шляхом
        if os.path.isdir(self._path):
            self._type = "dir"
        elif self._file_ext(filename=self._path) == ".zip":
            self._type = "zip"
            self._zipfile = self._load_zipfile(path=self._path)
        else:
            raise NotImplementedError(f"Path should be: {self._supported_types}")

        # Складання переліку файлів-кандидатів на основі структури даних
        self._element_filepaths = self._load_element_filepaths()

        # Ініціалізація словника для кодування назв класів у числові індекси
        self._label_encode = self._load_labels_encode()

        # Формування мапи відповідностей шлях до файлу - числову мітку
        self._element_labels = self._load_element_labels(
            filepaths=self._element_filepaths,
            label_ecode=self._label_encode
        )

    def __del__(self):
        """
        Деструктор. Викликається автоматично для звільнення ресурсів об'єкта,
        зокрема для закриття ZIP-архіву при необхідності
        """
        if self._zipfile is not None:
            self._zipfile.close()
        self._zipfile = None

    def __len__(self) -> int:
        """
        Описує загальну потужність вибірки, тобто кількість елементів у датасеті.
        Дозволяє інтеграцію з модулями PyTorch DataLoader
        :return: int
            Довжина відповідного датасету
        """
        return len(self._element_filepaths)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Забезпечує можливість індексації датасету.
        Повертає пару: зображення, піддане попередній обробці, та його one-hot представлення мітки класу

        :param item: int
            Індекс об'єкта вибірки
        :return: tuple(torch.Tensor, torch.Tensor)
            Попередньо оброблене зображення та відповідна one-hot мітка
        """
        image = self._transform(img=self._get_raw_image(idx=item))
        label = self._label_to_one_hot(label=self._get_raw_label(idx=item))
        return image, label

    @staticmethod
    def _file_ext(filename: str) -> str:
        """
        Відокремлює розширення заданого файлу
        :param filename: str
            Ім'я файлу
        :return: str
            Розширення файлу (у нижньому регістрі)
        """
        return os.path.splitext(filename)[1].lower()

    @staticmethod
    def _load_zipfile(path: str) -> zipfile.ZipFile:
        """
        Відкриває ZIP-архів для подальшого читання зображень без попереднього розпакування
        :param path: str
            Шлях до ZIP-архіву.
        :return: zipfile.ZipFile
            Відкритий об'єкт ZIP-архіву
        """
        return zipfile.ZipFile(file=path)

    def _load_element_filepaths(self) -> list[str]:
        """
        Генерує повний перелік відносних шляхів файлів усіх зображень у вибірці згідно структури зберігання
        :return: list[str]
            Список відносних шляхів до зображень
        """
        result = list()

        if self._type == "dir":
            for style_name in os.listdir(self._path):
                path__style_name = os.path.join(self._path, style_name)
                if os.path.isdir(path__style_name):
                    for element in os.listdir(path__style_name):
                        path__style_name__element = os.path.join(path__style_name, element)
                        if os.path.isfile(path__style_name__element):
                            result.append(os.path.join(style_name, element))

        elif self._type == "zip":
            for el in self._zipfile.namelist():
                if not el.endswith("/"):
                    result.append(el)

        else:
            raise NotImplementedError(f"Path should be: {self._supported_types}")

        return result

    @staticmethod
    def _load_element_labels(filepaths: list[str], label_ecode: dict[str, int]) -> dict[str, int]:
        """
        Формує мапу, що проектує шлях до файлу у відповідний індекс класу відповідно до структури каталогів
        :param filepaths: list[str]
            Список шляхів до прикладів вибірки
        :param label_ecode: dict[str, int]
            Відповідності імен класів їх індексам
        :return: dict[str, int]
            Відповідність шлях -> індекс класу
        """
        result = dict()
        for path in filepaths:
            result[path] = label_ecode[os.path.dirname(path)]
        return result

    def _load_labels_encode(self) -> dict[str, int]:
        """
        Завантажує словник, який ідентифікує зіставлення назв класів та їх індексів із JSON-файлу
        :return: dict[str, int]
            Відповідність назва класу -> індекс
        """
        with open(self._labels_path, mode="r", encoding="utf-8") as f:
            result = json.load(fp=f)
        return result

    def label_decode(self, class_idx: int) -> str:
        """
        Повертає назву класу згідно з його індексом
        :param class_idx: int
            Індекс класу
        :return: str
            Назва класу
        """
        for key, val in self._label_encode.items():
            if val == class_idx:
                return key
        raise IndexError()

    def _open_file(self, filename):
        """
        Відкриває файл для читання відповідно до типу зберігання датасету
        :param filename: str
            Відносний шлях до файла в датасеті
        :return: file-like object
            Об'єкт для читання з файла
        """
        if self._type == 'dir':
            return open(os.path.join(self._path, filename), 'rb')
        if self._type == 'zip':
            return self._zipfile.open(filename, 'r')
        return None

    def _get_raw_image(self, idx: int) -> np.ndarray:
        """
        Повертає сире зображення як масив numpy за індексом
        :param idx: int
            Індекс вибірки
        :return: np.ndarray
            Масив зображення (HWC)
        """
        filename = self._element_filepaths[idx]
        with self._open_file(filename=filename) as image:
            # noinspection PyTypeChecker
            image = np.asarray(Image.open(image))
        return image.copy()

    def _get_raw_label(self, idx: int) -> int:
        """
        Повертає числовий лейбл для зображення за індексом
        :param idx: int
            Індекс вибірки
        :return: int
            Індекс класу (label)
        """
        filename = self._element_filepaths[idx]
        return self._element_labels[filename]

    def _label_to_one_hot(self, label: int) -> torch.Tensor:
        """
        Конвертує індекс класу у one-hot тензор
        :param label: int
            Індекс класу
        :return: torch.Tensor
            One-hot вектор лейблу
        """
        result = torch.zeros(self._num_classes)
        result[label] = 1
        return result


class Classificator(torch.nn.Module):
    def __init__(self, num_classes: int):
        super(Classificator, self).__init__()
        # Ініціалізація моделі ResNet-50 зі зміненим останнім шаром
        self.model = torchvision.models.resnet50(weights='IMAGENET1K_V2')
        self.model.fc = torch.nn.Sequential(
            torch.nn.Linear(in_features=self.model.fc.in_features, out_features=num_classes),
            torch.nn.Softmax(dim=1)
        )

    def forward(self, x):
        """
        Виконує прямий прохід через усю модель
        :param x: torch.Tensor
            Вхідний батч зображень
        :return: torch.Tensor
            Вектор ймовірностей по класах
        """
        output = self.model(x)
        return output


def init_inner_filestructure():
    """
    Створює log-файл та папку для чекпоінтів, якщо вони відсутні
    """
    dirname = os.path.dirname(__file__)

    if not os.path.exists(os.path.join(dirname, LOG_NAME)):
        open(file=os.path.join(dirname, LOG_NAME), mode="w", encoding="utf-8").close()

    if not os.path.exists(os.path.join(dirname, CHECkPOINT_FOLDER)):
        os.mkdir(path=os.path.join(dirname, CHECkPOINT_FOLDER))


def print_and_write_log(log_line: str):
    """
    Виводить рядок на екран та дописує його у журнал логування
    :param log_line: str
        Текст для журналу
    """
    print(log_line)
    with open(file=os.path.join(os.path.dirname(__file__), LOG_NAME), mode="a", encoding="utf-8") as f:
        f.write(log_line + "\n")


def start_train(train_dataset_path: str, test_dataset_path: str, labels_path: str, num_classes: int,
                device: torch.device | str, batch_size: int = 32, epochs: int = 10):
    """
    Запускає повний цикл навчання: підготовка даних, ініціалізація моделі та оптимізатора, тренування,
    валідація, логування та збереження чекпоінтів
    :param train_dataset_path: str
        Шлях до тренувального датасету
    :param test_dataset_path: str
        Шлях до валідаційного датасету
    :param labels_path: str
        Шлях до json-файлу з мапінгом класів
    :param num_classes: int
        Кількість класів задачі
    :param device: torch.device | str
        Обчислювальний пристрій ('cpu' або 'cuda')
    :param batch_size: int
        Розмір batch при тренуванні
    :param epochs: int
        Кількість епох тренування
    """
    init_inner_filestructure()  # Створює усі необіхідні папки

    # Формування тренувального датасету
    dataset_train = CustomDateset(path=train_dataset_path, labels_path=labels_path, num_classes=num_classes)
    loader_train = torch.utils.data.DataLoader(dataset_train, shuffle=True, drop_last=True, batch_size=batch_size)

    # Формування тестового (валідаційного) датасету
    dataset_test = CustomDateset(path=test_dataset_path, labels_path=labels_path, num_classes=num_classes)
    loader_test = torch.utils.data.DataLoader(dataset_test, shuffle=True, drop_last=True, batch_size=batch_size)

    # Ініціалізація списків для накопичення динаміки зміни втрат та точності під час тренування та валідації
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    # Створення архітектури класифікатора
    model = Classificator(num_classes=num_classes)
    model.to(device)

    # Визначення функції втрат
    criterion = torch.nn.CrossEntropyLoss()

    # Ініціалізація оптимізатора для корекції ваг моделі
    optimizer = torch.optim.RMSprop(model.parameters(), lr=0.0001)

    # Основний цикл епох навчання
    for epoch in range(epochs):
        # Переведення моделі у режим тренування
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # Цикл по всіх батчах тренувальної вибірки
        for images, labels in tqdm.tqdm(loader_train, desc='Training loop'):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            # Визначення прогнозу моделі для поточного батчу
            outputs = model(images)

            # Обчислення значення функції втрат на батчі
            loss = criterion(outputs, labels)
            loss.backward()

            # Оновлення ваг моделі згідно з обраним оптимізатором
            optimizer.step()

            # Накопичення втрат за епоху (вага втрати пропорційно розміру батчу)
            running_loss += loss.item() * labels.size(0)

            # Визначення індексу класу з максимальним значенням для кожного прикладу у батчі
            predicted = torch.argmax(outputs.data, dim=1)
            labels = torch.argmax(labels.data, dim=1)

            # Оновлення лічильників всієї кількості об'єктів та правильно передбачених
            total += labels.size(0)
            # noinspection PyUnresolvedReferences
            correct += (predicted == labels).sum().item()

        # Обчислення середнього значення втрат та точності на тренувальній вибірці за епоху
        train_loss = running_loss / len(loader_train)
        train_accuracy = correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        # Переведення моделі у режим оцінки (evaluation)
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        # Вимикається автоматичний обрахунок градієнтів для пришвидшення валідації
        with torch.no_grad():
            # Проходження по всіх батчах тестової (валідаційної) вибірки
            for images, labels in tqdm.tqdm(loader_test, desc='Validation loop'):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_loss += loss.item() * labels.size(0)
                predicted = torch.argmax(outputs.data, dim=1)
                labels = torch.argmax(labels.data, dim=1)
                total += labels.size(0)
                # noinspection PyUnresolvedReferences
                correct += (predicted == labels).sum().item()

        # Обчислення середніх втрат та точності на валідаційній вибірці
        val_loss = running_loss / len(loader_test)
        val_accuracy = correct / total
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        # Формування інформаційного рядка з результатами епохи для журналу
        log_line = (f"Epoch {epoch + 1}/{epochs} - Train loss: {train_loss:.4f}, "
                    f"Validation loss: {val_loss:.4f}, "
                    f"Train Accuracy: {train_accuracy * 100:.2f}%, "
                    f"Validation Accuracy: {val_accuracy * 100:.2f}%")

        # Виведення та запис результатів в лог-файл
        print_and_write_log(log_line=log_line)

        # Збереження контрольної точки моделі після завершення епохи
        save_path = os.path.join(os.path.dirname(__file__), CHECkPOINT_FOLDER, f"model_at_{epoch + 1:04d}.pth")
        torch.save(model.state_dict(), save_path)
