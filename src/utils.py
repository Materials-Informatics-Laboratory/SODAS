from scipy.spatial.distance import squareform, pdist
from scipy import stats
import networkx as nx

from graphite.data import AngularGraphPairData
from graphite.utils.alignnd import *
from graphite.utils.graph import *

__all__ = [
    'nearest_path_node',
    'generate_latent_space_path',

]

def generate_latent_space_path(x,k=7):
    D = squareform(pdist(x, 'minkowski', p=2.))

    # select the kNN for each datapoint
    neighbors = np.sort(np.argsort(D, axis=1)[:, 0:k])

    G = nx.Graph()
    for i, x in enumerate(x):
        G.add_node(i)
    for i, row in enumerate(neighbors):
        for column in row:
            if i != column:
                G.add_edge(i, column)
                G[i][column]['weight'] = D[i][column]

    sink = len(x) - 1
    current_node = 0
    weighted_path = [current_node]
    checked_nodes = [current_node]
    while current_node != sink:
        nn = G.neighbors(current_node)

        chosen_nn = -1
        e_weight = 1E30
        for neigh in nn:
            if neigh not in checked_nodes:
                if nx.has_path(G,neigh,sink):
                    if G[current_node][neigh]['weight'] < e_weight:
                        e_weight = G[current_node][neigh]['weight']
                        chosen_nn = neigh
        if chosen_nn < 0:
            weighted_path.pop()
            current_node = weighted_path[-1]
        else:
            current_node = chosen_nn
            checked_nodes.append(current_node)
            weighted_path.append(current_node)

    path_edges = list(zip(weighted_path, weighted_path[1:]))

    path_distances = []
    for edge in path_edges:
        path_distances.append(G[edge[0]][edge[1]]['weight'])
    total_distance = sum(path_distances)
    travelled_distances = [0.0]
    d = 0.0
    for dist in path_distances:
        d += dist
        travelled_distances.append((d / total_distance))
    x_to_path, x_to_path_dist = nearest_path_node(x, weighted_path, travelled_distances)

    data = {
        "path": x_to_path,
        "path_dist": x_to_path_dist,
        "graph": G,
        "weighted_path": weighted_path,
        "d": travelled_distances,
        "path_edges": path_edges
    }

    return data

def nearest_path_node(x,nodes,distances):
    nn = []
    nd = []
    for val in x:
        nn.append(-1)
        nd.append(-1)
        d = 1E30
        for node in nodes:
            A = tuple([val,x[node]])
            D = squareform(pdist(A, 'minkowski', p=2.))
            r = D[0][1]
            if r < d:
                d = r
                nn[-1] = node
                nd[-1] = distances[nodes.index(node)]
    return nn, nd

def generate_graph(atoms,graph_type='',cutoff=3.0):
    # Construct normal graph G
    if graph_type == 'ALIGNN':
        edge_index_G, x_atm, x_bnd = atoms2graph(atoms, cutoff=cutoff, edge_dist=True)
        # Construct angular graph A
        edge_index_L_bnd_ang = line_graph(edge_index_G)
        x_bnd_ang = get_bnd_angs(atoms, edge_index_G, edge_index_L_bnd_ang)
        x_ang = np.concatenate([x_bnd_ang])
        edge_index_L = np.hstack([edge_index_L_bnd_ang])

        # Store everything into the custom `AngularGraphPairData` data class
        data = AngularGraphPairData(
            edge_index_G=torch.tensor(edge_index_G, dtype=torch.long),
            x_atm=torch.tensor(x_atm, dtype=torch.long),
            x_bnd=torch.tensor(x_bnd, dtype=torch.float),
            edge_index_A=torch.tensor(edge_index_L, dtype=torch.long),
            x_ang=torch.tensor(x_ang, dtype=torch.float),
        )

    return data













