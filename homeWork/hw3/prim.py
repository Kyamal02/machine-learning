import random
import sys
import warnings

import matplotlib.pyplot as plt
import networkx as nx

warnings.filterwarnings("ignore", category=DeprecationWarning)


def init(n):
    G = nx.Graph()
    nodes = range(n)
    G.add_nodes_from(nodes)
    edges = []
    edges_weight = {}

    # задаем веса для ребер
    for i in range(n):
        for j in range(i + 1, n):
            weight = random.randint(1, 10)
            edges.append((i, j, weight))
            edges_weight[(i, j)] = weight
    G.add_weighted_edges_from(edges)
    nodes_index = {}

    for i in range(n):
        nodes_index[i] = i

    # Сохраняем координаты вершин
    pos = nx.spring_layout(G)

    # Вывод графа с весами ребер
    nx.draw(G, pos, with_labels=True, node_color="yellow", edge_color="green", labels=nodes_index)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edges_weight)
    plt.show()

    plt.clf()

    return G, pos, edges_weight, nodes_index


def min_graph(graph, edges_weight, pos, nodes_index):
    min_nodes_tree = set()
    edges_tree = set()

    min_element1 = 0
    min_element2 = 0
    min_tuple = ()
    disconnected_wight = edges_weight
    # пока множество для вершин не будет включать в себя все вершины
    while len(min_nodes_tree) != len(graph.nodes.items()):
        min_v = sys.maxsize
        for key, value in edges_weight.items():

            element1 = key[0]
            element2 = key[1]
            # только одна вершина из key должна быть в множестве и она должна быть min
            # если множество пустое, то просто ищем минимальное ребро(для первого шага)
            if (((element1 in min_nodes_tree) != (element2 in min_nodes_tree)) and value < min_v) \
                    or (len(min_nodes_tree) == 0 and value < min_v):
                min_element1 = element1
                min_element2 = element2
                min_tuple = key
                min_v = value

        min_nodes_tree.add(min_element1)
        min_nodes_tree.add(min_element2)
        edges_tree.add(min_tuple)

    # рисуем красным минимальное дерево во всем графе
    nx.draw(graph, pos, with_labels=True, node_color="yellow", edge_color="green", labels=nodes_index)
    nx.draw_networkx_edges(graph, pos, edgelist=edges_tree, edge_color='red')
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edges_weight)
    plt.show()
    plt.clf()

    # рисуем минимальоне дерево
    nx.draw_networkx_nodes(graph, pos, node_color="yellow")
    nx.draw_networkx_labels(graph, pos, labels=nodes_index)
    nx.draw_networkx_edges(graph, pos, edgelist=edges_tree, edge_color='red')

    plt.show()
    plt.clf()

    return edges_tree, min_nodes_tree


def prim_algorithm(graph, pos, edges_tree, min_nodes_tree, edges_weight, number_of_clusters):
    counter_number_of_clusters = number_of_clusters - 1
    disconnected_graph = edges_tree

    # убираем лишние ребра
    while counter_number_of_clusters > 0:
        counter_number_of_clusters -= 1
        max_v = -sys.maxsize
        max_edge = ()
        for key, value in edges_weight.items():
            if value > max_v and key in edges_tree:
                max_v = value
                max_edge = key
        disconnected_graph.remove(max_edge)

    draw_clusters(disconnected_graph, min_nodes_tree, graph)


def draw_clusters(disconnected_graph, min_nodes_tree, graph):
    # найдем кластеры, которые содержат одну вершину и далее закрасим их
    single_nodes = set()
    for element in disconnected_graph:
        for node in min_nodes_tree:
            if node not in element:
                single_nodes.add(node)
    for i, node in enumerate(single_nodes):
        single_node = [node]
        nx.draw_networkx_nodes(graph, pos, nodelist=single_node, node_color=colors[i])

    graph = nx.Graph(disconnected_graph)

    components = list(nx.connected_components(graph))

    for i, component in enumerate(components):
        nodes_to_draw = list(component)  # Получаем список вершин в текущей компоненте
        nx.draw_networkx_nodes(graph, pos, nodelist=nodes_to_draw, node_color=colors[i + 100])
    nx.draw_networkx_labels(graph, pos, labels=nodes_index)
    nx.draw_networkx_edges(graph, pos, edgelist=disconnected_graph, edge_color='red')
    keys_to_remove = []

    for key, value in edges_weight.items():
        if key not in disconnected_graph:
            keys_to_remove.append(key)

    for key in keys_to_remove:
        edges_weight.pop(key)
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edges_weight)

    plt.show()
    plt.clf()


colors = random_colors = [
    '#{:02x}{:02x}{:02x}'.format(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in
    range(255)]

if __name__ == "__main__":
    NUMBER_OF_NODES = 20
    NUMBER_OF_CLUSTER = 5

    graph, pos, edges_weight, nodes_index = init(NUMBER_OF_NODES)

    # получили вершины и ребра для минимального дерева без циклов
    edges_tree, min_nodes_tree = min_graph(graph, edges_weight, pos, nodes_index)

    prim_algorithm(graph, pos, edges_tree, min_nodes_tree, edges_weight, NUMBER_OF_CLUSTER)
