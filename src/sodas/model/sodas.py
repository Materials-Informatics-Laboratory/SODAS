from torch_geometric.data import DataLoader
from graphite.nn.models import ALIGNN
from umap import umap_
import torch.nn as nn
import numpy as np

class SODAS():
    def __init__(self, mod=ALIGNN(dim=100, num_interactions=6, num_species=3, cutoff=3.0),
                 ls_mod=umap_.UMAP(n_neighbors=10, min_dist=0.5,n_components=2)):
        super().__init__()

        self.model      = mod
        self.umap       = ls_mod
        self.preprocess = None

    def generate_gnn_latent_space(self,loader,device=''):
        if device == '':
            print(f"Is CUDA supported by this system?{torch.cuda.is_available()}")
            print(f"CUDA version: {torch.version.cuda}")

            # Storing ID of current CUDA device
            cuda_id = torch.cuda.current_device()
            print(f"ID of current CUDA device:{torch.cuda.current_device()}")

            print(f"Name of current CUDA device:{torch.cuda.get_device_name(cuda_id)}")

            # Prepare model
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = nn.DataParallel(self.model)
        self.model.eval()
        self.model.to(device)

        print('Generating GNN latent space...')
        # Run a forward pass
        counter = 0
        total_data = []
        for data in loader:
            data = data.to(device)
            pred = self.model(data)
            data.detach()
            for p in pred[0]:
                x = p.cpu().detach().numpy()
                total_data.append(x)
            print('Finished generating data ',str(counter))
            counter += 1
        return np.array(total_data)

    def fit_umap(self,data,preprocess_data=1):
        if preprocess_data:
            from sklearn import preprocessing
            self.preprocess = preprocessing.StandardScaler().fit(data)
            data = self.preprocess.transform(data)
        self.umap.fit(data)

    def project_data(self,data,preprocess_data=1):
        if preprocess_data:
            data = self.preprocess.transform(data)
        data = self.umap.transform(data)

        return data















