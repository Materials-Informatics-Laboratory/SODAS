# SODAS
Toolkit for the characterization of atomistic phase trajectories

![Screenshot](https://github.com/Materials-Informatics-Laboratory/SODAS/blob/main/method.png?raw=true)

image/sodas++_workflow.png
## Installation

The following dependencies need to be installed before installing `sodas`. The installation time is typically within 10 minutes on a normal local machine.
- PyTorch (`pytorch>=1.8.1`)
- PyTorch-Geometric (`pyg>=2.0.1`): for implementing graph representations
- LLNL/Graphite: for ALIGNN GNN representation (https://github.com/LLNL/graphite)
- UMAP-learn (`umap-learn>=0.5.3`)
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

`sodas` is intended to be a plug-and-play framework where you provide data in the form as an `ase` atoms object and sodas++ does the rest. You have full control over the ALIGNN and UMAP projections through the sodas class.

- The `src` folder contains the source code.
- The `example` folder contains an example for how to use SODAS++ to characterize an Al melt molecular dynamics simulation.
