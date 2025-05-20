import json

import h5py
import numpy as np
from matplotlib import pyplot as plt


DATASET_ORIGINAL_PATH = "create_latent_dataset/data/original/data.h5"
LABEL_DECODE_PATH = "dataset/artbench256-60k-split/labels.json"


def get_data_dist():
    """
    Візуалізує розподіл класів (стилів) у датасеті, виводить частоту для кожної категорії та їх назви
    """
    # Завантажуємо словник відповідності індекс -> назва стилю
    with open(LABEL_DECODE_PATH, mode="r", encoding="utf-8") as f:
        label_decode = json.load(fp=f)
    label_decode = [x[0] for x in sorted(label_decode.items(), key=lambda x: x[1], reverse=False)]

    # Зчитуємо матрицю ймовірностей класів із файла
    with h5py.File(DATASET_ORIGINAL_PATH, mode="r") as f:
        labels = f["labels"][:]

    # Аргмаксом витягуємо індекси класів
    labels = np.argmax(labels, axis=1)
    label_x, label_y = np.unique(labels, return_counts=True)

    print(label_y)
    print(len(labels))

    # Гістограма частоти зразків за класами
    plt.figure(figsize=(9, 6))
    plt.bar(label_x, label_y / len(labels))
    plt.xticks(label_x, label_decode, rotation=90)
    plt.xlabel("Стиль")
    plt.tight_layout()
    plt.show()

def get_confident_dist():
    """
    Візуалізує розподіл впевненості класифікатора (максимальна ймовірність для кожного зразка)
    """
    # Зчитування ймовірностей для всіх класів
    with h5py.File(DATASET_ORIGINAL_PATH, mode="r") as f:
        labels = f["labels"][:]

    # Вибираємо максимальні значення ймовірностей (впевненість класифікатора)
    labels = np.max(labels, axis=1)

    # Ваги для нормування гістограми (частки)
    weights = np.ones_like(labels) / len(labels)

    # Побудова гістограми впевненості для всієї вибірки
    plt.figure(figsize=(10, 6))
    plt.hist(labels, bins=40, weights=weights)
    plt.xlabel("Значення впевненості класифікатора")
    plt.ylabel("Частка результатів")
    plt.tight_layout()
    plt.show()


def main():
    get_data_dist()
    get_confident_dist()


if __name__ == '__main__':
    main()
