import sys

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris


def plot_iris(dataset):
    fig, ax = plt.subplots(3, 3, figsize=(12, 10))
    for i in range(4):
        for j in range(i + 1, 4):
            # точки
            scatter = ax[i][j - 1].scatter(dataset.data[:, i], dataset.data[:, j], c=dataset.target)
            # метки по осям
            ax[i][j - 1].set(xlabel=dataset.feature_names[i], ylabel=dataset.feature_names[j])
            # вывод классов
            ax[i][j - 1].legend(scatter.legend_elements()[0], dataset.target_names,
                                loc="lower right", title="classes")

    plt.show()


def normalize_iris(dataset):
    normalized_dataset = dataset

    # массив массиво в котором каждый массив это данные об 1 элементе(150)
    data = normalized_dataset.data
    # минимум и максимум для каждого столбца
    for j in range(4):
        min_param = min_dataset_param(data, j)
        max_param = max_dataset_param(data, j)

        # нормализируем столбец
        for i in range(len(data)):
            data[i][j] = (data[i][j] - min_param) / (max_param - min_param)

    return normalized_dataset


def normalize_item(item, dataset):
    for i in range(4):
        min_param = min_dataset_param(dataset.data, i)
        max_param = max_dataset_param(dataset.data, i)
        item[i] = min((item[i] - min_param) / (max_param - min_param), 1)
        print(min_param, max_param)

    return item


def min_dataset_param(data, index):
    min_param = sys.maxsize
    for i in range(len(data)):
        if data[i][index] < min_param:
            min_param = data[i][index]
    return min_param


def max_dataset_param(data, index):
    max_param = -float('inf')
    for i in range(len(data)):
        if data[i][index] > max_param:
            max_param = data[i][index]
    return max_param


def generate_test_dataset(dataset):
    test_dataset = []
    train_dataset = []

    items = []
    for i in range(len(dataset.data)):
        items.append([dataset.data[i], dataset.target[i]])

    for i in range(len(dataset.data)):
        if i % 10 == 0:
            test_dataset.append(items[i])
        else:
            train_dataset.append(items[i])

    return train_dataset, test_dataset


def optimal_k(train_dataset, test_dataset):
    n = len(train_dataset) + len(test_dataset)
    optimal = 1
    best_accuracy = 0
    for k in range(1, int(np.sqrt(n))):
        counter = 0
        for item in test_dataset:
            item_class = knn_class(item[0], train_dataset, k)
            if item_class == item[1]:
                counter += 1
        accuracy = counter / len(test_dataset)

        if accuracy > best_accuracy:
            optimal = k
            best_accuracy = accuracy
    return optimal, best_accuracy


def knn_class(item, train_dataset, k):
    # отсортированный массив массивов. Первый элемент дистанция, а второй до какого класса
    distances = sorted_of_distance(item, train_dataset)
    # массив для дальнейшего высчитывания частоты
    array_of_class = []
    # словарь для частоты классов
    class_frequencies = {0: 0, 1: 0, 2: 0}

    for j in range(k):
        array_of_class.append(distances[j][1])
        # Подсчитайте частоту каждого числа
    for num in array_of_class:
        class_frequencies[num] += 1
        # Найдите число с максимальной частотой
    most_common_class = max(class_frequencies, key=class_frequencies.get)
    return most_common_class


def sorted_of_distance(item, train_dataset):
    dist_array = []

    for i in range(len(train_dataset)):
        empty_array = []
        empty_array.append(dist(train_dataset[i][0], item))
        empty_array.append(train_dataset[i][1])

        dist_array.append(empty_array)
    sorted_array = sorted(dist_array, key=lambda x: x[0])

    return sorted_array


def dist(a, b):
    return np.sqrt(
        (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2 + (a[3] - b[3]) ** 2)


if __name__ == "__main__":
    iris_dataset = load_iris()

    plot_iris(iris_dataset)

    normalize_dataset = normalize_iris(iris_dataset)
    plot_iris(iris_dataset)

    train_dataset, test_dataset = generate_test_dataset(normalize_dataset)

    optimal, best_accuracy = optimal_k(train_dataset, test_dataset)

    print(f'Optimal neighbours: {optimal}, Accuracy: {best_accuracy}')

    iris_dataset = load_iris()
    # sep l, sep w, pet l, pet w
    # 4.3-7.9  2.0-4.4 1.0-6.9 0.1-2.5
    item = [7.9, 4, 4, 2]
    item = normalize_item(item, iris_dataset)
    print(item)
    cl = knn_class(item, train_dataset, optimal)
    print(f'New item is {iris_dataset.target_names[cl]}')

