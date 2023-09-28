import math

import numpy as np
import matplotlib.pyplot as plt
import random
import color


class Point:
    def __init__(self, x, y, clusterID=None):
        self.x = x
        self.y = y
        self.clusterID = clusterID


def random_points(n):
    points = []
    for i in range(n):
        points.append(Point(random.randint(0, 100), random.randint(0, 100)))
    for elem in points:
        plt.scatter(elem.x, elem.y, color="g")
    return points


def dist(pointA, pointB):
    return np.sqrt((pointA.x - pointB.x) ** 2 + (pointA.y - pointB.y) ** 2)


def first_centroids(points, n):
    for point in points:
        plt.scatter(point.x, point.y, color="g")
    pointCenter = Point(0, 0)
    for elem in points:
        pointCenter.x += elem.x
        pointCenter.y += elem.y
    pointCenter.x /= len(points)
    pointCenter.y /= len(points)

    R = 0
    for elem in points:
        distCurr = dist(elem, pointCenter)
        if distCurr > R:
            R = distCurr
    centroids = []

    for i in range(n):
        centroids.append(Point(R * np.cos(2 * np.pi * i / n) + pointCenter.x,
                               R * np.sin(2 * np.pi * i / n) + pointCenter.y, i))

        test = centroids

    for i in range(n):
        plt.scatter(centroids[i].x, centroids[i].y, color="r")

    return centroids


def show_points(points):
    plt.show()


def k_means(points, centroids, display):
    should_continue = True
    count = 0
    prev_centroids = None

    plt.clf()
    while should_continue:
        count += 1
        for point in points:
            min_dist = 2 ** 16
            for i in range(len(centroids)):
                if prev_centroids is None: prev_centroids = list(centroids)
                if min_dist > dist(point, prev_centroids[i]):
                    min_dist = dist(point, prev_centroids[i])
                    point.clusterID = i

        if display:
            draw(points, prev_centroids, count)
            # for i in range(len(centroids)):
            #     print(prev_centroids[i].x, prev_centroids[i].y, prev_centroids[i].clusterID)

        new_centroids = update_centroid(points, prev_centroids)

        are_equal = True
        for i in range(len(centroids)):
            if prev_centroids[i].x != new_centroids[i].x or prev_centroids[i].y != new_centroids[i].y:
                prev_centroids = new_centroids
                are_equal = False
                break
        if are_equal:
            should_continue = False
    return points, prev_centroids


def draw(points, centroids, count):
    for i in range(len(points)):
        plt.scatter(points[i].x, points[i].y, color=color.colors[points[i].clusterID])
    for centroid in centroids:
        plt.scatter(centroid.x, centroid.y, color="r")
    plt.savefig("res/step" + str(count))
    plt.clf()


# Обновляет центроиды
def update_centroid(points, centroids):
    new_centroid = list(centroids)
    for i in range(len(centroids)):
        sum_x = 0
        sum_y = 0
        count = 0
        for point in points:
            if point.clusterID == i:
                sum_x += point.x
                sum_y += point.y
                count += 1
        if (count != 0):
            new_centroid[i] = Point(sum_x / count, sum_y / count, i)
    return new_centroid


# Метод для вывода оптимального количества кластеров
def optimal_clusters(points):
    criteria = [] * 15
    for i in range(10):
        temp_points = points
        centroids = first_centroids(temp_points, i + 1)
        temp_points, centroids = k_means(temp_points, centroids, False)
        criteria.append(sum(dist(point, centroids[point.clusterID]) ** 2 for point in temp_points))
    plt.plot(criteria)
    plt.xticks(np.arange(0, 9), np.arange(1, 10))
    plt.savefig("charts/optimal_cluster_chart")
    plt.clf()
    print(criteria)
    # criteria[k+1] - criteria[k]
    diff = np.diff(criteria)
    print(diff)
    # масштаб изменения (изменение между k+1 и k)
    diff_r = diff[1:] / diff[:-1]
    print(diff_r)
    plt.plot(diff_r)
    plt.xticks(np.arange(0, 8), np.arange(2, 10))
    plt.savefig("charts/scale_of_changes")
    min = 2 ** 16
    cluster = 0
    for i in range(len(diff_r)):
        if (min > diff_r[i]):
            min = diff_r[i]
            cluster = i + 2
    print(cluster)
    return cluster

if __name__ == "__main__":
    points = random_points(150)
    plt.savefig("res/step_random_points")
    plt.clf()
    optimal_clusters(points)
    plt.clf()
    centroids = first_centroids(points, optimal_clusters(points))
    plt.savefig("res/step_first_centroids")
    k_means(points, centroids, True)
