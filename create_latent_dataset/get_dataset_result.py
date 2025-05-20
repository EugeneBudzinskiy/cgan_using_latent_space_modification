import os
import shutil

import h5py
import numpy as np


DATASET_COMBINED_PATH = "data/clean/data.h5"
DATASET_RESULT_PATH = "data/result/"
LABELS_PATH = "dataset/artbench256-60k-split/labels.json"

TRAIN_FRACTION = 0.8  # Частка даних, що відводиться під train


def main():
    # Переконуємося, що папка для збереження результатів існує
    os.makedirs(DATASET_RESULT_PATH, exist_ok=True)

    # Копіюємо словник класів для цілісності структури датасету
    shutil.copy2(src=LABELS_PATH, dst=DATASET_RESULT_PATH)

    # Відкриття вирівняного датасету та зчитування всієї вибірки у пам'ять
    with h5py.File(DATASET_COMBINED_PATH, mode="r") as f:
        vectors = f["vectors"][:]
        labels = f["labels"][:]

    # Обчислюємо кількість об'єктів для train/test поділу
    train_length = int(TRAIN_FRACTION * len(labels))

    # Генеруємо випадкову перестановку індексів для перемішування вибірки
    idx = np.arange(0, len(labels))
    np.random.shuffle(idx)

    # Зберігаємо тренувальну підмножину у файл з компресією
    train_vectors, train_labels = vectors[idx[:train_length]], labels[idx[:train_length]]
    with h5py.File(os.path.join(DATASET_RESULT_PATH, "train.h5"), mode="w") as f:
        f.create_dataset("vectors", data=train_vectors, compression="gzip", compression_opts=3)
        f.create_dataset("labels", data=train_labels, compression="gzip", compression_opts=3)

    # Зберігаємо тестову підмножину у файл з компресією
    test_vectors, test_labels = vectors[idx[train_length:]], labels[idx[train_length:]]
    with h5py.File(os.path.join(DATASET_RESULT_PATH, "test.h5"), mode="w") as f:
        f.create_dataset("vectors", data=test_vectors, compression="gzip", compression_opts=3)
        f.create_dataset("labels", data=test_labels, compression="gzip", compression_opts=3)


if __name__ == '__main__':
    main()
