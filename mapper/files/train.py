# V7

import os

import h5py
import numpy as np
import torch
import torch.utils.data
import tqdm


LOG_FILENAME = "log.txt"  # Ім'я файлу журналу
CHECKPOINT_FOLDER = "checkpoint/"  # Ім'я папки для збереження контрольних точок (чекпоінтів)


class CustomDateset(torch.utils.data.Dataset):
    """
    Кастомний датасет, що забезпечує завантаження векторів ознак та міток класів із h5py-файлу.
    """
    def __init__(self, path: str):
        self.vectors, self.labels = self._load_dataset(path=path)

    @classmethod
    def _load_dataset(cls, path: str) -> tuple[np.ndarray, np.ndarray]:
        """
        Завантажує масиви ознак та міток класів із hdf5-файлу
        :param path: str
            Шлях до hdf5-файлу
        :return: tuple[np.ndarray, np.ndarray]
            Масиви векторів ознак та міток класів
        """
        with h5py.File(path, mode="r") as f:
            vectors = f["vectors"][:]
            labels = f["labels"][:]
            return vectors, labels

    def __len__(self) -> int:
        """
        Описує загальну потужність вибірки, тобто кількість елементів у датасеті.
        Дозволяє інтеграцію з модулями PyTorch DataLoader
        :return: int
            Довжина відповідного датасету
        """
        return len(self.labels)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Повертає пару тензорів: вектор ознак та one-hot розмітку класу
        :param item: int
            Індекс елемента датасету
        :return: tuple[torch.Tensor, torch.Tensor]
            Пара (вектор ознак, one-hot розмітка класу)
        """
        vector = torch.from_numpy(self.vectors[item]).float()
        label = torch.tensor(self.labels[item]).float()
        return vector, label


class BasicBlock(torch.nn.Module):
    """
    Базовий блок нейронної мережі, що реалізує функцію двох повнозв'язних шарів з нормалізацією,
    dropout та нелінійністю, а також резидуальний зв'язок (skip-connection)
    """
    def __init__(self, in_channels: int, out_channels: int):
        super(BasicBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.lin_1 = torch.nn.Linear(self.in_channels, self.out_channels, bias=False)
        self.act_1 = torch.nn.ReLU()
        self.norm_1 = torch.nn.LayerNorm(self.out_channels)
        self.drop_1 = torch.nn.Dropout(0.3)

        self.lin_2 = torch.nn.Linear(self.out_channels, self.out_channels, bias=False)
        self.act_2 = torch.nn.ReLU()
        self.norm_2 = torch.nn.LayerNorm(self.out_channels)
        self.drop_2 = torch.nn.Dropout(0.3)

        # Додатковий шлях для вирівнювання розмірностей (downsampling) при зміні ширини шару
        self.lin_down = torch.nn.Linear(self.in_channels, self.out_channels, bias=False)
        self.norm_down = torch.nn.LayerNorm(self.out_channels)

    def forward(self, x):
        """
        Прямий прохід через блок із реалізацією резидуального вкладення для уникнення затухання градієнтів
        :param x: torch.Tensor
            Вхідний тензор ознак
        :return: torch.Tensor
            Вихідний тензор після обробки
        """
        out = self.lin_1(x)
        out = self.norm_1(out)
        out = self.act_1(out)
        out = self.drop_1(out)

        out = self.lin_2(out)
        out = self.norm_2(out)

        # Коригує форму тензора для сумування, якщо кількість каналів змінилась
        if self.in_channels != self.out_channels:
            x = self.lin_down(x)
            x = self.norm_down(x)

        out = out + x

        out = self.act_2(out)
        out = self.drop_2(out)

        return out


class Mapper(torch.nn.Module):
    """
    Архітектура моделі (маппер), що складається з послідовності лінійних шарів, блоків BasicBlock,
    нормалізації, dropout та фінального класифікаційного шару
    """
    def __init__(self, z_dim: int, num_classes: int):
        super(Mapper, self).__init__()

        self.dims = [z_dim, 256, 256, 128, 64, 32]

        self.linear_1 = torch.nn.Linear(self.dims[0], self.dims[1], bias=False)
        self.activation_1 = torch.nn.ReLU()
        self.norm_1 = torch.nn.LayerNorm(self.dims[1])
        self.drop_1 = torch.nn.Dropout(0.3)

        self.layer_1 = BasicBlock(in_channels=self.dims[1], out_channels=self.dims[2])
        self.layer_2 = BasicBlock(in_channels=self.dims[2], out_channels=self.dims[3])
        self.layer_3 = BasicBlock(in_channels=self.dims[3], out_channels=self.dims[4])
        self.layer_4 = BasicBlock(in_channels=self.dims[4], out_channels=self.dims[5])

        self.linear_last = torch.nn.Linear(in_features=self.dims[-1], out_features=num_classes)

    def forward(self, x):
        """
        Прямий прохід через усі шари моделі — вектор ознак трансформується у логіти
        :param x: torch.Tensor
           Вхідний тензор ознак
        :return: torch.Tensor
           Логіти (до softmax)
        """
        x = self.linear_1(x)
        x = self.activation_1(x)
        x = self.norm_1(x)
        x = self.drop_1(x)

        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)

        x = self.linear_last(x)
        return x


def init_inner_filestructure():
    """
    Створює log-файл та папку для чекпоінтів, якщо вони відсутні
    """
    dirname = os.path.dirname(__file__)
    if not os.path.exists(os.path.join(dirname, LOG_FILENAME)):
        open(file=os.path.join(dirname, LOG_FILENAME), mode="w", encoding="utf-8").close()
    os.makedirs(os.path.join(dirname, CHECKPOINT_FOLDER), exist_ok=True)



def print_and_write_log(log_line: str):
    """
    Виводить рядок на екран та дописує його у журнал логування
    :param log_line: str
       Текст для журналу
    """
    print(log_line)
    with open(file=os.path.join(os.path.dirname(__file__), LOG_FILENAME), mode="a", encoding="utf-8") as f:
        f.write(log_line + "\n")


def start_train(train_dataset_path: str, test_dataset_path: str, num_classes: int, z_dim: int,
                batch_size: int, epochs: int, device: torch.device | str):
    """
    Запускає повний цикл навчання: підготовка даних, ініціалізація моделі та оптимізатора, тренування,
    валідація, логування та збереження чекпоінтів
    :param train_dataset_path: str
        Шлях до h5py-файлу тренувального датасету
    :param test_dataset_path: str
        Шлях до h5py-файлу валідаційного датасету
    :param num_classes: int
        Кількість класів гіпотетичної класифікації
    :param z_dim: int
        Розмірність ознак (або латентного простору)
    :param batch_size: int
        Розмір батчу
    :param epochs: int
        Кількість епох навчання нейромережі
    :param device: torch.device | str
        Вибране апаратне забезпечення ('cpu' або 'cuda')
    """
    init_inner_filestructure()  # Створює усі необіхідні папки

    # Формування тренувального датасету
    dataset_train = CustomDateset(path=train_dataset_path)
    loader_train = torch.utils.data.DataLoader(dataset_train, shuffle=True, drop_last=True, batch_size=batch_size)

    # Формування тестового (валідаційного) датасету
    dataset_test = CustomDateset(path=test_dataset_path)
    loader_test = torch.utils.data.DataLoader(dataset_test, shuffle=True, drop_last=True, batch_size=batch_size)

    # Ініціалізація списків для накопичення динаміки зміни втрат та точності під час тренування та валідації
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    # Створення архітектури маппера
    model = Mapper(z_dim=z_dim, num_classes=num_classes)
    model.to(device)

    # Визначення функції втрат
    criterion = torch.nn.KLDivLoss(reduction='batchmean')

    # Ініціалізація оптимізатора для корекції ваг моделі
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)

    # Основний цикл епох навчання
    for epoch in range(epochs):
        # Переведення моделі у режим тренування
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # Цикл по всіх батчах тренувальної вибірки
        for vectors, labels in tqdm.tqdm(loader_train, desc='Training loop'):
            vectors, labels = vectors.to(device), labels.to(device)
            optimizer.zero_grad()

            # Прогнозування виходу моделі
            outputs = model(vectors)

            # Переведення логітів у логймовірності для KLDivLoss
            log_probs = torch.nn.functional.log_softmax(outputs, dim=1)
            loss = criterion(log_probs, labels)
            loss.backward()
            optimizer.step()

            # Накопичення втрат для поточної епохи
            running_loss += loss.item() * labels.size(0)

            # Обчислення метрик
            predicted = torch.argmax(outputs, dim=1)
            true = torch.argmax(labels, dim=1)
            total += labels.size(0)
            correct += (predicted == true).sum().item()

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
            for vectors, labels in tqdm.tqdm(loader_test, desc='Validation loop'):
                vectors, labels = vectors.to(device), labels.to(device)
                outputs = model(vectors)

                log_probs = torch.nn.functional.log_softmax(outputs, dim=1)
                loss = criterion(log_probs, labels)

                running_loss += loss.item() * labels.size(0)
                predicted = torch.argmax(outputs, dim=1)
                true = torch.argmax(labels, dim=1)
                total += labels.size(0)
                correct += (predicted == true).sum().item()

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
        save_path = os.path.join(os.path.dirname(__file__), CHECKPOINT_FOLDER, f"model_at_{epoch + 1:04d}.pth")
        torch.save(model.state_dict(), save_path)