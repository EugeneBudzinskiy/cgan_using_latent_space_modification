import os
from collections import defaultdict

import numpy as np
import h5py
from matplotlib import pyplot as plt


DATASET_COMBINED_PATH = "data/original/data.h5"
DATASET_CLEAN_FOLDER_PATH = "data/clean/"

CONFIDENCE_THRESHOLD = 0.9  # Поріг довіри до класу для включення прикладу
ROUND_TO = 5000  # Вирівнюємо розмір датасету до кратного цього числа
NOISE_STD = 2000  # Середньоквадратичне відхилення доданої шумової вибірки для кожного класу


def get_confident_by_class(vectors: np.ndarray, labels: np.ndarray) -> dict[int, list[float]]:
    """
    Групує індекси прикладів за найбільш імовірним класом і відкидає приклади з низькою впевненістю класифікатора
    :param vectors: np.ndarray
       Матриця ознак об'єктів
    :param labels: np.ndarray
       Матриця ймовірностей класів для усіх прикладів
    :return: dict[int, list[float]]
       Словник: клас → індекси прикладів, що задовольняють поріг впевненості
    """
    confident_by_class = defaultdict(list)
    # Проходимо усі приклади, визначаємо найбільш ймовірний клас і рівень впевненості
    for i in range(len(vectors)):
        class_id, confidence = labels[i].argmax(), labels[i].max()
        if confidence >= CONFIDENCE_THRESHOLD:
            confident_by_class[class_id].append(i)
    return confident_by_class


def get_balanced_indices(confident_by_class: dict[int, list[float]], min_class_count: int) -> list[int]:
    """
    Для кожного класу випадковим чином добирає близько однакову кількість зразків, використовуючи шум для варіативності
    :param confident_by_class: dict[int, list[float]]
        Словник: клас → набір індексів впевнених прикладів
    :param min_class_count: int
        Мінімальна кількість прикладів серед усіх класів (для балансування)
    :return: list[int]
        Список індексів для формування вирівняного датасету
    """
    target_indices = []
    # Для кожного класу додається флуктуація (шум) для випадкової вибірки навколо min_class_count
    for cls, indices in confident_by_class.items():
        noise = int(np.random.normal(loc=0, scale=NOISE_STD))
        target_count = min(min_class_count + abs(noise), len(indices))  # Не перевищуємо наявну вибірку класу

        sampled = np.random.choice(indices, size=target_count, replace=False)
        target_indices.extend(sampled)
    return target_indices


def trim_to_round_number(target_indices: list[int]) -> list[int]:
    """
    Перемішує та підрізає список індексів так, щоб їх кількість була кратною ROUND_TO
    :param target_indices: list[int]
        Список всіх відібраних індексів
    :return: list[int]
        Обрізаний і перемішаний список індексів
    """
    np.random.shuffle(target_indices)
    rounded_size = (len(target_indices) // ROUND_TO) * ROUND_TO
    return target_indices[:rounded_size]


def main():
    # Завантаження початкового датасету з ознаками та ймовірностями класів
    with h5py.File(DATASET_COMBINED_PATH, mode="r") as f:
        vectors = f["vectors"][:]
        labels = f["labels"][:]

    # Отримання для кожного класу всіх індексів прикладів з високою впевненістю в класифікації
    confident_by_class = get_confident_by_class(vectors=vectors, labels=labels)
    min_class_count = min(len(v) for v in confident_by_class.values())
    print(f"Мінімальний розмір класу: {min_class_count}")

    # Для балансованого датасету вибираються достатні кількості прикладів з кожного класу
    target_indices = get_balanced_indices(confident_by_class=confident_by_class, min_class_count=min_class_count)
    target_indices = trim_to_round_number(target_indices=target_indices)  # Optional
    print(f"Фнальний розмір класу: {len(target_indices)}")

    # Зберігаємо очищений та вирівняний датасет у файл у форматі HDF5 із компресією
    os.makedirs(DATASET_CLEAN_FOLDER_PATH, exist_ok=True)
    with h5py.File(os.path.join(DATASET_CLEAN_FOLDER_PATH, "data.h5"), mode="w") as f:
        f.create_dataset("vectors", data=vectors[target_indices], compression="gzip", compression_opts=3)
        f.create_dataset("labels", data=labels[target_indices], compression="gzip", compression_opts=3)

    # Для візуальної перевірки будуються гістограми розподілу класів у вибірці
    labels = np.argmax(labels[target_indices, :], axis=1)
    label_x, label_y = np.unique(labels, return_counts=True)
    plt.figure(figsize=(10, 6))
    plt.bar(label_x, label_y / len(labels))
    plt.xticks(label_x, label_x)
    plt.xlabel("Клас")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
