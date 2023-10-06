import pygame
import numpy as np
import color


class Point:

    def __init__(self, x, y, flag, name, claster_id, point_id):
        self.x = x
        self.y = y
        self.flag = flag
        self.name = name
        self.claster_id = claster_id
        self.point_id = point_id


def init_screen():
    pygame.init()
    screen = pygame.display.set_mode((600, 400))
    screen.fill(color="#ffffff")
    pygame.display.update()
    return screen


def draw(screen):
    drawing = False
    clock = pygame.time.Clock()
    points = []
    count_point_id = 0
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Проверка, что нажата левая кнопка мыши
                    drawing = True
            if event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    drawing = False
            if event.type == pygame.MOUSEMOTION:
                if drawing:
                    coords = event.pos
                    pygame.draw.circle(screen, color="#000000", center=coords, radius=3)
                    x1, y1 = coords
                    count_point_id += 1
                    points.append(Point(x1, y1, False, "", None, count_point_id))

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_1:
                    points = dbScan(points, 30, 4, screen)

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_2:
                    paint_clusters(points)

        pygame.display.update()
        clock.tick(10)


def dbScan(points, eps, m, screen):
    clusters_point = 0
    for point in points:
        if point.flag is True:
            continue
        point.flag = True
        NeighborPts = regionQuery(points, point, eps)
        if len(NeighborPts) == 0:
            point.name = "TRASH"
        if len(NeighborPts) < m and len(NeighborPts) != 0:  # Если количество соседей меньше MinPts
            point.name = "NOSE"  # шум
        if len(NeighborPts) >= m and len(NeighborPts) != 0:
            clusters_point += 1
            point.name = "ROOT"
            point.claster_id = clusters_point
            expandCluster(points, NeighborPts, clusters_point, eps, m)
    points = paint(points, m, eps)
    return points


def expandCluster(points, NeighborPts, cluster, eps, m):
    for point in NeighborPts:

        if point.flag is False:
            point.flag = True
            QNeighborPts = regionQuery(points, point, eps)
            if len(QNeighborPts) >= m:  # Если у Q достаточно соседей
                point.name = "ROOT"
                NeighborPts += QNeighborPts  # Объединить списки соседей
        if point.claster_id is None:
            point.claster_id = cluster
    for point in points:
        for neighbor in NeighborPts:
            if point.point_id == neighbor.point_id:
                point = neighbor


def paint(points, m, eps):
    screen.fill(color="#ffffff")
    pygame.display.update()
    for point in points:
        if point.name == "ROOT":
            pygame.draw.circle(screen, color="green", center=(point.x, point.y), radius=3)
            pygame.draw.circle(screen, color="black", center=(point.x, point.y), radius=30, width=1)
        if point.name == "NOSE" and point.claster_id != 0:
            pygame.draw.circle(screen, color="yellow", center=(point.x, point.y), radius=3)
        if (point.name == "TRASH") or (point.name == "NOSE" and point.claster_id is None):
            pygame.draw.circle(screen, color="red", center=(point.x, point.y), radius=3)
        if point.name == "":
            neighborhood = regionQuery(points, point, eps)
            if len(neighborhood) >= m:
                pygame.draw.circle(screen, color="green", center=(point.x, point.y), radius=3)
            for n in neighborhood:
                if n.name == "ROOT":
                    point.claster_id = n.claster_id
                    pygame.draw.circle(screen, color="yellow", center=(point.x, point.y), radius=3)
    for point in points:
        print(point.claster_id, point.name, point.flag, point.point_id)
    return points


def paint_clusters(points):
    for point in points:
        if point.claster_id is not None:
            pygame.draw.circle(screen, color=color.colors[point.claster_id], center=(point.x, point.y), radius=3)


def regionQuery(points, pointA, eps):
    neighborhood = []  # Создаем пустой список для хранения соседей
    for pointB in points:
        if (pointA != pointB):
            # Вычисляем расстояние между точкой P и текущей точкой point
            distance_to_point = dist(pointA, pointB)
            if distance_to_point <= eps:
                neighborhood.append(pointB)  # Если расстояние меньше или равно eps, добавляем точку в окрестность
    return neighborhood


def dist(pointA, pointB):
    x1, y1 = pointA.x, pointA.y
    x2, y2 = pointB.x, pointB.y
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


if __name__ == "__main__":
    screen = init_screen()
    draw(screen)
