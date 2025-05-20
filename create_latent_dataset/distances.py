import json
import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import TSNE


DATASET_CLEAN_PATH = "data/clean/data.h5"
LABEL_DECODE_PATH = "dataset/artbench256-60k-split/labels.json"


def find_conflicting_neighbors(vectors: np.ndarray, labels: np.ndarray, eps: float = 1e-4):
    """
    Визначає кількість регіонів у латентному просторі, в яких знаходяться зразки різних класів в межах заданого радіусу
    :param vectors: np.ndarray
        Матриця латентних векторів розмірності (N, d)
    :param labels: np.ndarray
        Вектор або матриця міток класів (N,)
    :param eps: float
        Радіус пошуку сусідів
    :return: int
        Кількість конфліктних регіонів (із різними класами)
    """
    conflicts = 0  # Лічильник конфліктних сусідств

    # Пошук сусідів у радіусі eps для кожної точки
    nbrs = NearestNeighbors(radius=eps, algorithm="ball_tree").fit(vectors)
    neighbors = nbrs.radius_neighbors(vectors, return_distance=False)

    # Перевірка для кожної точки: якщо у сусідстві є декілька різних класів - рахується конфлікт
    for i, inds in enumerate(neighbors):
        if len(inds) > 1:
            local_labels = set(labels[inds])
            if len(local_labels) > 1:
                conflicts += 1

    print(f"Зайдено {conflicts} конфліктуючих регіонів з радіусом {eps}")
    return conflicts


def find_conflicting_neighbors_all(vectors, labels, eps_list):
    """
    Аналізує частку конфліктних регіонів для різних значень радіуса eps та будує відповідний графік
    :param vectors: np.ndarray
        Матриця латентних векторів
    :param labels: np.ndarray
        Вектор міток класів
    :param eps_list: list[float]
        Перелік радіусів для аналізу локальної неоднорідності простору
    :return: tuple
        Кортеж із переліком радіусів та часток конфліктних регіонів
    """
    conflicts_per_eps = []
    for eps in eps_list:
        # Для кожного eps визначаємо частку зразків із багатокласовим локальним сусідством
        conflicts = find_conflicting_neighbors(vectors, labels, eps)
        conflicts /= len(labels)
        conflicts_per_eps.append(conflicts)
        print(f"Епсілон: {eps: .4g}, Конфіліктів: {conflicts}")

    # Візуалізуємо залежність кількості конфліктів від параметру eps
    plt.figure(figsize=(10, 4))
    plt.plot(eps_list, conflicts_per_eps, marker='o')
    plt.xlabel("Епсілон радіус")
    plt.ylabel("Частка конфліктів")
    plt.title("Конфлікти регіонів латентних векторів")
    plt.tight_layout()
    plt.show()

    return eps_list, conflicts_per_eps


def visualize_latents(vectors, labels):
    """
    Візуалізує латентний простір із використанням проекції t-SNE та кольорового кодування класів
    :param vectors: np.ndarray
        Матриця латентних векторів
    :param labels: np.ndarray
        Вектор дискретних міток класів
    :return: None
    """
    # Завантажується словник для декодування індексів класу у символічні назви стилів
    with open(LABEL_DECODE_PATH, mode="r", encoding="utf-8") as f:
        label_decode = json.load(fp=f)
    label_decode = [x[0] for x in sorted(label_decode.items(), key=lambda x: x[1], reverse=False)]

    tsne = TSNE(n_components=2, perplexity=30, max_iter=1000)
    z_2d = tsne.fit_transform(vectors)

    # Точкова діаграма із кодуванням класів кольором
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(z_2d[:, 0], z_2d[:, 1], c=labels, cmap='tab10', s=10, alpha=0.8)
    handles, _ = scatter.legend_elements()
    plt.legend(handles, label_decode, title="Стиль")
    plt.title("t-SNE проекція латентного простору")
    plt.tight_layout()
    plt.show()


def main():
    # Завантаження латентних векторів та розмітки класів із HDF5-файлу
    with h5py.File(DATASET_CLEAN_PATH, mode="r") as f:
        vectors = f["vectors"][:]
        labels = f["labels"][:]

    # Перетворення міток у цілі числа (argmax для one-hot)
    labels = np.argmax(labels, axis=1)

    # Випадкове перемішування для подальшого аналізу підмножини
    indices = np.arange(len(labels))
    np.random.shuffle(indices)

    n = 3_000  # Кількість точок для аналізу
    vectors = vectors[indices[:n], :]
    labels = labels[indices[:n]]

    # Візуалізація t-SNE для латентного простору
    visualize_latents(vectors=vectors, labels=labels)

    # Аналіз частки конфліктних регіонів у латентному просторі для різних eps
    eps_list = [12, 14, 16, 18, 20, 22, 24, 26]
    find_conflicting_neighbors_all(vectors=vectors, labels=labels, eps_list=eps_list)

    # Побудова гістограми відстаней для випадкової підмножини точок
    indices = np.arange(len(vectors))
    np.random.shuffle(indices)
    data = vectors[indices[:n], :]
    distances = pdist(data, metric="euclidean")

    plt.figure(figsize=(10, 6))
    plt.hist(distances, bins=50, edgecolor="black", density=True)
    plt.title("Гістограма попарної відстані")
    plt.xlabel("Відстань")
    plt.ylabel("Частота")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
