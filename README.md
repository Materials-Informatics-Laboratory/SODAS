# SODAS
Toolkit for the characterization of atomistic phase trajectories

- `sodas` allows for the conversion of atomic graphs (in the form of the graph + line graph method known as ALIGNN) to a spatio-temperally resolved latent space. Useful for understanding structural trnaisitons during atomistic simulations. The projection scheme allows for the spatial and temporal characterization of structure during a transition (otherwise known as a reaction coordinate). `sodas` can tell you how similar structures are to one another, as well as quantify their evolution through time by labelling each structure during a transition based on how far it is located in the latent space from know end points. Note, each latent space projection scheme you choose will vary, ex: PCA may give different results than UMAP. 

![Screenshot](https://github.com/Materials-Informatics-Laboratory/SODAS/blob/main/method.png?raw=true)

## Installation

The following dependencies need to be installed before installing `sodas`. The installation time is typically within 10 minutes on a normal local machine.
- PyTorch (`pytorch>=1.8.1`)
- PyTorch-Geometric (`pyg>=2.0.1`): for implementing graph representations
- Networkx (`networkx>=2.8.6`)
- Scipy (`scipy>=1.9.0`)
- Numpy (`numpy>=1.21.1`)
- Atomic Simulation Environment (`ase>= 3.22.1`): for reading/writing atomic structures

To install `sodas`, clone this repo and run:
```bash
pip install -e /path/to/the/repo
```

The `-e` option signifies an [editable install](https://pip.pypa.io/en/stable/topics/local-project-installs/), which is well suited for development; this allows you to edit the source code without having to re-install.

To uninstall:
```bash
pip uninstall sodas
```

## How to use

`sodas` is intended to be a plug-and-play framework where you provide data in the form as an `ase` atoms object and sodas++ does the rest. You have full control over the ALIGNN and the data projections through the sodas class.

- The `src` folder contains the source code.
- The `example` folder contains an example for how to use SODAS++ to characterize an Al melt molecular dynamics simulation.

## Citing SODAS

Please use the following citiation to cite the SODAS toolkit:
Bamidele Aroboto, Shaohua Chen, Tim Hsu, Brandon C. Wood, Yang Jiao, James Chapman; Universal and interpretable classification of atomistic structural transitions via unsupervised graph learning. Appl. Phys. Lett. 28 August 2023; 123 (9): 094103.

Or cite directly from the manuscript at: https://pubs.aip.org/aip/apl/article/123/9/094103/2909293/Universal-and-interpretable-classification-of
