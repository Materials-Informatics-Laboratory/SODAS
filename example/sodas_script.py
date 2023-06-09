from graphite.nn.models import ALIGNN
from umap import umap_

from sodas.utils.utils import generate_latent_space_path
from sodas.utils.utils import generate_graph
from sodas.model.sodas import SODAS

from torch_geometric.data import DataLoader
from ase.io import read
import networkx as nx
import torch as torch
import numpy as np
import os

import matplotlib.pyplot as plt
import matplotlib as mpl

read_path = r'PATH\TO\melt.dump'
write_path = r'PATH\TO\melt.dump'
test_file = r'melt.dump'
sub_dir = 'melt'

save_frequency = 500

test_file = os.path.join(read_path,test_file)

traj = read(test_file,index=':')

sodas_model = SODAS(mod=ALIGNN(dim=100, num_interactions=6, num_species=3, cutoff=3.0),
                 ls_mod=umap_.UMAP(n_neighbors=10, min_dist=0.25,n_components=10))

dataset = []
for i, snapshot in enumerate(traj):
    print('Generating graph for snapshot ', i)
    data = generate_graph(snapshot, graph_type='ALIGNN',cutoff=3.0)
    dataset.append(data)
    if i % save_frequency == 0:
        torch.save(dataset, os.path.join(write_path, sub_dir + '.pt'))
torch.save(dataset, os.path.join(write_path, sub_dir + '.pt'))

dataset = torch.load(os.path.join(write_path, sub_dir + '.pt'))
follow_batch = ['x_atm', 'x_bnd', 'x_ang'] if hasattr(dataset[0], 'x_ang') else ['x_atm']
loader = DataLoader(dataset, batch_size=1, shuffle=False, follow_batch=follow_batch)

encoded_data = sodas_model.generate_gnn_latent_space(device='cuda',loader=loader)
np.save(os.path.join(write_path,'total_gnn_encoding.npy'), encoded_data,allow_pickle=True)
sodas_model.fit_umap(data=encoded_data)
projected_data = sodas_model.project_data(data=encoded_data)

c = np.linspace(0,len(projected_data),len(projected_data))
plt.scatter(projected_data[:,0],projected_data[:,1],c=c)
plt.show()

path_data = generate_latent_space_path(X=projected_data,k=10)

pos = nx.kamada_kawai_layout(path_data["graph"])
color_lookup = {k: v for v, k in enumerate(sorted(set(path_data["graph"].nodes())))}
low, *_, high = sorted(color_lookup.values())
norm = mpl.colors.Normalize(vmin=low, vmax=high, clip=True)
mapper = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.viridis)

nx.draw(path_data["graph"], pos,node_size=500, node_color=[mapper.to_rgba(i) for i in color_lookup.values()], with_labels=False,
        edge_color='black', edgecolors='k',alpha=0.75)
plt.show()

t = np.linspace(0, len(projected_data), len(projected_data))
fig, (ax0,ax1) = plt.subplots(1, 2, figsize=(12, 4))
import matplotlib.patheffects as pe
colors = plt.cm.viridis(path_data["d"])
for k in range(len(path_data["weighted_path"])):
    if k == len(path_data["weighted_path"]) - 1:
        break

    ax1.plot([projected_data[path_data["weighted_path"][k], 0],projected_data[path_data["weighted_path"][k + 1], 0]],
                             [projected_data[path_data["weighted_path"][k], 1], projected_data[path_data["weighted_path"][k + 1], 1]], color=colors[k], lw=4,
                             path_effects=[pe.Stroke(linewidth=6, foreground='k'), pe.Normal()])

im1 = ax0.scatter(projected_data[:, 0], projected_data[:, 1], c=t, cmap='viridis', edgecolor='k',s=100)
im2 = ax1.scatter(projected_data[:, 0], projected_data[:, 1], c=path_data["path_dist"], cmap='viridis', edgecolor='k',s=100)

fig.colorbar(im1,ax=ax0, extend="both")
fig.colorbar(im2,ax=ax1, extend="both")
plt.show()
