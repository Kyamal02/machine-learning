import sys

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


def draw(screen, eps, m):
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
                if event.button == 1:
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
                    points = dbScan(points, eps, m)
                if event.key == pygame.K_2:
                    paint_clusters(points)

        pygame.display.update()
        clock.tick(15)


def dbScan(points, eps, m):
    clusters_point = 0
    for point in points:
        if point.flag is True:
            continue
        point.flag = True
        NeighborPts = regionQuery(points, point, eps)
        if len(NeighborPts) == 0:
            point.name = "NOSE"
        elif len(NeighborPts) >= m:
            clusters_point += 1
            point.name = "ROOT"
            point.claster_id = clusters_point
            expandCluster(points, NeighborPts, clusters_point, eps, m)
    points = paint(points, eps)
    return points


def expandCluster(points, NeighborPts, cluster, eps, m):
    for point in NeighborPts:
        if point.flag is False:
            point.flag = True
            QNeighborPts = regionQuery(points, point, eps)
            if len(QNeighborPts) >= m:
                point.name = "ROOT"
                NeighborPts += QNeighborPts  # Объединить списки соседей
            else:
                point.name = "BORDER"
        if point.claster_id is None:
            point.claster_id = cluster
    for point in points:
        for neighbor in NeighborPts:
            if point.point_id == neighbor.point_id:
                point = neighbor


def regionQuery(points, pointA, eps):
    neighborhood = []
    for pointB in points:
        if (pointA != pointB):
            distance_to_point = dist(pointA, pointB)
            if distance_to_point <= eps:
                neighborhood.append(pointB)
    return neighborhood


def paint(points, eps):
    screen.fill(color="#ffffff")
    pygame.display.update()
    noise_checking(points, eps)
    for point in points:
        if point.name == "ROOT":
            pygame.draw.circle(screen, color="green", center=(point.x, point.y), radius=3)
            # pygame.draw.circle(screen, color="black", center=(point.x, point.y), radius=30, width=1)
        if point.name == "BORDER":
            pygame.draw.circle(screen, color="yellow", center=(point.x, point.y), radius=3)
            pygame.draw.circle(screen, color="black", center=(point.x, point.y), radius=30, width=1)
        if point.name == "NOSE":
            pygame.draw.circle(screen, color="red", center=(point.x, point.y), radius=3)
            # pygame.draw.circle(screen, color="black", center=(point.x, point.y), radius=30, width=1)
    return points


def noise_checking(points, eps):
    for p in points:
        if p.name == "":
            min_dist = sys.maxsize
            neighborhood = regionQuery(points, p, eps)
            for n in neighborhood:
                distance = dist(p, n)
                if n.name == "ROOT" and min_dist > distance:
                    p.name = "BORDER"
                    min_dist = distance
            if p.name == "":
                p.name = "NOSE"


def paint_clusters(points):
    screen.fill(color="#ffffff")
    pygame.display.update()
    for point in points:
        if point.claster_id is not None:
            pygame.draw.circle(screen, color=color.colors[point.claster_id], center=(point.x, point.y), radius=3)
        else:
            pygame.draw.circle(screen, color="red", center=(point.x, point.y), radius=3)


def dist(pointA, pointB):
    x1, y1 = pointA.x, pointA.y
    x2, y2 = pointB.x, pointB.y
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


if __name__ == "__main__":
    m = 4
    eps = 30

    screen = init_screen()
    draw(screen, eps, m)
