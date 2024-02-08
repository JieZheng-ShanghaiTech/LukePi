import torch

import numpy as np
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
from  torch_geometric.data import DataLoader
from torch_geometric.data import NeighborSampler
from tqdm import tqdm
from time import time
import numpy as np
import pandas as pd
import pickle




class HeteroDataset(InMemoryDataset):
    def __init__(self, save_root, transform=None, pre_transform=None):
        """
        loading heterogenous dataset
        Args:
        :save_root: the root of saving data
        :param pre_transform: data pre_transform operation before loading data
        
        """
        super(HeteroDataset, self).__init__(save_root, transform, pre_transform)
        print(self.processed_file_names[0])
        # self.data, self.slices = torch.load(self.processed_file_names[0])
        with open(self.processed_file_names[0],'rb') as f:
            self.data=pickle.load(f)

    @property
    def raw_file_names(self): 
        return ['origin_dataset']
    

  

    @property
    def processed_file_names(self):
       
        return ["./BKG/kgdata.pkl"]
    
      

    def download(self):  
        pass

    def process(self):   
        
        pass








if __name__ == "__main__":

   
    toy_data = HeteroDataset(save_root="/home/siyutao/SL/pretrain-gnns/Rebuttal")  
 
  
    

    