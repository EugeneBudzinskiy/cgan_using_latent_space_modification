import os

import numpy as np
import torch
import torch.utils.data
import tqdm


LOG_FILENAME = "log.txt"  # Ім'я файлу журналу
CHECKPOINT_FOLDER = "checkpoint"  # Ім'я папки для збереження контрольних точок (чекпоінтів)


class CustomDateset(torch.utils.data.Dataset):
    """
    Генерує датасет з фіксованих випадкових векторів шуму
    """
    def __init__(self, noise_size: int, num_classes: int, length: int):
        self.noise_size = noise_size
        self.num_classes = num_classes
        self._length = length

        # Генерується масив випадкових "seed" для відтворюваності кожного шуму
        self._seeds = np.random.randint(low=0, high=2_000_000_000, size=self._length)

    def __len__(self) -> int:
        """
        Описує загальну потужність вибірки, тобто кількість елементів у датасеті.
        Дозволяє інтеграцію з модулями PyTorch DataLoader
        :return: int
           Довжина відповідного датасету
        """
        return self._length

    def __getitem__(self, item: int) -> torch.Tensor:
        """
        За індексом item відтворюється унікальний seed, з якого генерується
        псевдовипадковий вектор нормального розподілу (розміром noise_size)
        :param item: int
            Індекс об'єкта вибірки
        :return: torch.Tensor
            Відповідний вектор шуму
        """
        seed = self._seeds[item]
        noise = torch.from_numpy(np.random.RandomState(seed=seed).randn(1, self.noise_size)[0]).float()
        return noise


class Encoder(torch.nn.Module):
    """
    Архітектура модуля енкодера для формування латентних зсувів для кожного класу
    """
    def __init__(self, num_classes: int, z_dim: int):
        super(Encoder, self).__init__()

        self.num_classes = num_classes
        self.z_dim = z_dim

        # Embedding — словник векторів зсуву для кожного класу (розмірності z_dim)
        self.style_shift = torch.nn.Embedding(num_embeddings=num_classes, embedding_dim=z_dim)

    def forward(self, x):
        return self.style_shift(x)


def init_inner_filestructure():
    """
    Створює log-файл та папку для чекпоінтів, якщо вони відсутні
    """
    dirname = os.path.dirname(__file__)

    if not os.path.exists(os.path.join(dirname, LOG_FILENAME)):
        open(file=os.path.join(dirname, LOG_FILENAME), mode="w", encoding="utf-8").close()

    if not os.path.exists(os.path.join(dirname, CHECKPOINT_FOLDER)):
        os.mkdir(path=os.path.join(dirname, CHECKPOINT_FOLDER))


def print_and_write_log(log_line: str):
    """
    Виводить рядок на екран та дописує його у журнал логування
    :param log_line: str
        Текст для журналу
    """
    print(log_line)
    with open(file=os.path.join(os.path.dirname(__file__), LOG_FILENAME), mode="a", encoding="utf-8") as f:
        f.write(log_line + "\n")


def start_train(mapper: torch.nn.Module, noise_size: int, num_classes: int, z_dim: int, dataset_train_length: int,
                dataset_test_length: int, batch_size: int, epochs: int, device: torch.device | str):
    """
    Запускає повний цикл навчання: підготовка даних, ініціалізація моделі та оптимізатора, тренування,
    валідація, логування та збереження чекпоінтів
    :param mapper: torch.nn.Module
        Навчений або ініціалізований класифікатор, над яким оптимізується енкодер
    :param noise_size: int
        Розмірність латентного вектору шуму
    :param num_classes: int
        Кількість класів
    :param z_dim: int
        Розмірність латентного зсуву (embedding)
    :param dataset_train_length: int
        Кількість елементів у тренувальному датасеті
    :param dataset_test_length: int
        Кількість елементів у тестовому (валідаційному) датасеті
    :param batch_size: int
        Кількість зразків у одному батчі
    :param epochs: int
        Кількість епох навчання
    :param device: torch.device | str
        Обчислювальний пристрій (cpu/gpu)
    """
    init_inner_filestructure()  # Створює усі необіхідні папки

    # Формування тренувального датасету
    dataset_train = CustomDateset(noise_size=noise_size, num_classes=num_classes, length=dataset_train_length)
    loader_train = torch.utils.data.DataLoader(dataset_train, shuffle=True, drop_last=True, batch_size=batch_size)

    # Формування тестового (валідаційного) датасету
    dataset_test = CustomDateset(noise_size=noise_size, num_classes=num_classes, length=dataset_test_length)
    loader_test = torch.utils.data.DataLoader(dataset_test, shuffle=True, drop_last=True, batch_size=batch_size)

    # Ініціалізація списків для накопичення динаміки зміни втрат та точності під час тренування та валідації
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    # Створення архітектури енкодера
    encoder = Encoder(num_classes=num_classes, z_dim=z_dim)
    encoder = encoder.to(device)

    # Визначення функції втрат
    criterion = torch.nn.CrossEntropyLoss()

    # Ініціалізація оптимізатора для корекції ваг моделі
    optimizer = torch.optim.Adam(encoder.parameters(), lr=0.0001)

    # Основний цикл епох навчання
    for epoch in range(epochs):
        # Переведення моделі у режим тренування
        encoder.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # Цикл по всіх батчах тренувальної вибірки
        for noise in tqdm.tqdm(loader_train, desc='Training loop'):
            noise = noise.to(device)

            # Для кожного елементу батчу випадково призначається клас (рівномірний розподіл)
            class_idx = torch.randint(num_classes, size=(1, noise.size(0)))[0].to(device)
            optimizer.zero_grad()

            # Генерується стильовий зсув згідно індексів класу
            delta = encoder(class_idx)

            # Зміщений шум подається на mapper (зовнішній ядерний класифікатор)
            predicted = mapper(noise + delta)

            # Розрахунок функції втрат між прогнозом і справжнім класовим індексом
            loss = criterion(predicted, class_idx)
            loss.backward()
            optimizer.step()

            # Оновлення статистики для оцінки функції втрат
            total +=  noise.size(0)
            running_loss += loss.item() *  noise.size(0)

            # Порівняння argmax по виходу з реальними класами
            predicted_idx = torch.argmax(predicted.data, dim=1)
            correct += (predicted_idx == class_idx).sum().item()

        # Обчислення середнього значення втрат та точності на тренувальній вибірці за епоху
        train_loss = running_loss / len(loader_train)
        train_accuracy = correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        # Переведення моделі у режим оцінки (evaluation)
        encoder.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        # Вимикається автоматичний обрахунок градієнтів для пришвидшення валідації
        with torch.no_grad():
            # Проходження по всіх батчах тестової (валідаційної) вибірки
            for noise in tqdm.tqdm(loader_test, desc='Validation loop'):
                noise = noise.to(device)
                class_idx = torch.randint(num_classes, size=(1, noise.size(0)))[0].to(device)

                delta = encoder(class_idx)
                predicted = mapper(noise + delta)
                loss = criterion(predicted, class_idx)

                total += noise.size(0)
                running_loss += loss.item() * noise.size(0)
                predicted_idx = torch.argmax(predicted.data, dim=1)
                correct += (predicted_idx == class_idx).sum().item()

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
        torch.save(encoder.state_dict(), save_path)
