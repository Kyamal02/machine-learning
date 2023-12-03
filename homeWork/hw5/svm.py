import random

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report
from sklearn.svm import SVC


class Point:
    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        self.color = color


colors = ["red", "green"]


def generate_points(n):
    train_points = []
    test_points = []
    for i in range(int(n / 2)):
        if i % 10 == 0:
            test_points.append(Point(random.randint(40, 60), random.randint(40, 60), 0))
        else:
            train_points.append(Point(random.randint(40, 60), random.randint(40, 60), 0))

    for i in range(int(n / 4)):
        if i % 10 == 0:
            test_points.append(Point(random.randint(0, 20), random.randint(0, 20), 1))
        else:
            train_points.append(Point(random.randint(0, 20), random.randint(0, 20), 1))

    for i in range(int(n / 4)):
        if i % 10 == 0:
            test_points.append(Point(random.randint(80, 100), random.randint(80, 100), 1))
        else:
            train_points.append(Point(random.randint(80, 100), random.randint(80, 100), 1))

    # for i in range(int(n / 2)):
    #     if i % 10 == 0:
    #         test_points.append(Point(random.randint(50, 100), random.randint(50, 100), 0))
    #     else:
    #         train_points.append(Point(random.randint(50, 100), random.randint(50, 100), 0))
    #
    # for i in range(int(n / 2)):
    #     if i % 10 == 0:
    #         test_points.append(Point(random.randint(0, 50), random.randint(0, 50), 1))
    #     else:
    #         train_points.append(Point(random.randint(0, 50), random.randint(0, 50), 1))

    for point in train_points:
        plt.scatter(point.x, point.y, color=colors[point.color])
    return train_points, test_points


def svm_alg(train_points, test_points):
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    for i in range(len(train_points)):
        x_train.append([train_points[i].x, train_points[i].y])
        y_train.append((train_points[i].color))
    for i in range(len(test_points)):
        x_test.append([test_points[i].x, test_points[i].y])
        y_test.append((test_points[i].color))

    model = SVC(kernel='rbf', C=1E6)
    # model = SVC(kernel='linear', C=1E6)

    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    print(classification_report(y_test, y_pred))

    X = x_train
    y = y_train
    X = np.array(X)
    y = np.array(y)
    return model


def plot_svc_decision_function(model, ax=None, plot_support=True):
    # получаем лимиты для графика
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # одномерные массивы чисел с равномерно распределенными значениями в заданном диапазоне
    x = np.linspace(xlim[0], xlim[1], 10)
    y = np.linspace(ylim[0], ylim[1], 10)

    # создание сетки точек в двумерном пространстве
    Y, X = np.meshgrid(y, x)
    # plt.plot(Y, X, marker='.', color='k', linestyle='none')
    # plt.show()

    # print(Y)
    # print("  ")
    # print(Y.ravel())
    # для решающей функции
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    # print("  ")
    # print(xy)

    # for point in xy:
    #     plt.scatter(point[0], point[1], color="red")
    # plt.show()

    # Решающая функция
    P = model.decision_function(xy).reshape(X.shape)

    #  levels=[-1, 0, 1] расстояния от гиперплоскости
    ax.contour(X, Y, P, colors='black',
               levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])


if __name__ == "__main__":
    train_points, test_points = generate_points(100)

    plt.savefig('figure1')

    model = svm_alg(train_points, test_points)
    plot_svc_decision_function(model)
    plt.savefig('figure2')

    point = [90, 88]

    new_point_class = model.predict([point])[0]

    plt.scatter(point[0], point[1], color=colors[new_point_class])
    plt.savefig('figure3')
