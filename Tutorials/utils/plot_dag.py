import itertools
from graphviz import Digraph


def get_dag_plot(dag):
    dot = Digraph()
    nodes = set(itertools.chain.from_iterable(dag))
    for node in nodes:
        dot.node(node, node)
    dot.edges(dag)

    return dot
