import random

import numpy as np
import pandas as pd


# Скрещивание двух маршрутов с фиксированным начальным городом
def crossover_fixed_start(parent1, parent2):
    # Выбираем две случайные точки для обмена участками маршрутов
    start, end = sorted(random.sample(range(1, len(parent1)), 2))
    child = [None] * len(parent1)
    child[0] = 0  # Фиксированный начальный город
    child[start:end] = parent1[start:end]
    # Заполняем оставшиеся города из второго родителя
    child_index = end
    for city in parent2:
        if city not in child:
            if child_index >= len(parent1):
                child_index = 1
            child[child_index] = city
            child_index += 1
    return child


# Выбор выживших в популяции
def select_survivors(population, offspring, distance_matrix, population_size):
    # Объединяем текущую популяцию и потомство
    combined = population + offspring
    # Сортируем по приспособленности (от высокой к низкой)
    combined.sort(key=lambda route: calculate_fitness(route, distance_matrix), reverse=True)
    # Возвращаем лучшие решения
    return combined[:population_size]


# Основная функция генетического алгоритма
def genetic_algorithm_fixed_start(distance_matrix, population_size, num_parents, num_generations):
    num_cities = distance_matrix.shape[0]
    # Инициализация популяции
    population = initialize_population_fixed_start(population_size, num_cities)
    for _ in range(num_generations):
        # Выбор родителей
        parents = select_parents(population, distance_matrix, num_parents)
        # Генерация потомства
        offspring = [crossover_fixed_start(parents[i], parents[(i + 1) % len(parents)]) for i in range(len(parents))]
        # Выбор выживших
        population = select_survivors(population, offspring, distance_matrix, population_size)

    # Выбираем лучший маршрут
    best_route = population[0]
    best_fitness = calculate_fitness(best_route, distance_matrix)
    return best_route, best_fitness


# Выбор родителей для генерации потомства
def select_parents(population, distance_matrix, num_parents):
    # Вычисляем приспособленность для каждого маршрута
    fitness_scores = np.array([calculate_fitness(route, distance_matrix) for route in population])

    # Выбираем индексы родителей на основе их приспособленности(отбор по вероятностям)
    parent_indices = np.random.choice(len(population), size=num_parents, replace=False,
                                      p=fitness_scores / fitness_scores.sum())

    # Возвращаем выбранных родителей
    parents = [population[idx] for idx in parent_indices]
    return parents


# Функция для вычисления приспособленности маршрута
def calculate_fitness(route, distance_matrix):
    # Суммируем расстояния между последовательными городами в маршруте
    total_distance = sum(distance_matrix[route[i], route[i + 1]] for i in range(len(route) - 1))
    # Приспособленность - обратная величина к общей длине маршрута
    return 1 / total_distance


# Инициализация популяции с фиксированным начальным городом
def initialize_population_fixed_start(population_size, num_cities):
    population = []
    routes_set = set()  # Набор для хранения уникальных маршрутов

    while len(population) < population_size:
        # Генерируем уникальный маршрут
        route = np.random.permutation(num_cities - 1) + 1
        route = np.insert(route, 0, 0)

        # Преобразуем маршрут в неизменяемый тип (кортеж), чтобы можно было добавить в набор
        route_tuple = tuple(route)
        # Добавляем маршрут в популяцию, если он уникален
        if route_tuple not in routes_set:
            routes_set.add(route_tuple)
            population.append(route)

    return population


# Функция для фильтрации городов и получения матрицы расстояний
def filter_cities(file_path, cities):
    data = pd.read_excel(file_path, index_col=0, skiprows=1)
    data = data.replace(' ', None)
    selected_data = data.loc[cities, cities]
    distance_array = selected_data.values
    return distance_array


# Вычисление общей длины маршрута
def calculate_total_distance(route, distance_matrix):
    total_distance = sum(distance_matrix[route[i], route[i + 1]] for i in range(len(route) - 1))
    # Добавляем расстояние от последнего города обратно в начальный город
    total_distance += distance_matrix[route[-1], route[0]]
    return total_distance


if __name__ == "__main__":
    file_path = 'table.russia.xls'
    selected_cities = ["Казань", "Москва", "Архангельск", "Краснодар", "Уфа", "Пермь"]
    # Фильтрация городов и получение матрицы расстояний
    distance_matrix = filter_cities(file_path, selected_cities)

    # Замена None на очень большие значения
    distance_matrix = np.where(distance_matrix == None, 1e6, distance_matrix)

    # Параметры генетического алгоритма
    population_size = 120
    num_parents = 4
    num_generations = 5

    # Запуск генетического алгоритма
    best_route, best_fitness = genetic_algorithm_fixed_start(distance_matrix, population_size, num_parents,
                                                             num_generations)

    # Вычисление общей длины лучшего маршрута
    best_route_distance = calculate_total_distance(best_route, distance_matrix)

    print("Лучший маршрут:", best_route)
    print("Приспособленность маршрута:", best_fitness)
    print("Общая длина лучшего маршрута:", best_route_distance)
