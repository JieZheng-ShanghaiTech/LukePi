import random
import torch
import numpy as np
import pandas as pd
from torch_geometric.loader import HGTLoader



class EdgeMaskNegative:
    def __init__(self, mask_rate,device):
        """
        Randomly sample negative edges
        Assume edge_attr is of the form:
        [30 attr, self_loop, mask]
        :param mask_rate: % of edges to be masked
        """
        self.mask_rate = mask_rate
        self.device=device
    
    def __call__(self,data,masked_edge_indices=None):
        edge_types=data.edge_types
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)
    
        if masked_edge_indices == None:
            # sample for each edge type
            for i in range(len(edge_types)):
                num_edges = int(data[edge_types[i]].edge_index.size()[1] / 2)  # num unique edges
                
                sample_size = int(num_edges * self.mask_rate)
                masked_edge_indices = [k for k in random.sample(range(num_edges), sample_size)]
                positive_pair=data[edge_types[i]].edge_index[:,masked_edge_indices].to(self.device)
                negative_pair=Negativesub(data,num_edges,i)
                data[edge_types[i]].negative_pair=negative_pair
                data[edge_types[i]].prediction_edge = torch.cat((positive_pair,negative_pair.to(self.device)),dim=1)
                positive_label=torch.ones(len(masked_edge_indices),1)*i
                negative_label=torch.ones(negative_pair.shape[1],1)*(len(data.edge_types))
                data[edge_types[i]].prediction_label=torch.cat((positive_label,negative_label),dim=0)
                all_edge=set(list(range(data[edge_types[i]].edge_index.shape[1])))
                mask_edge=set(masked_edge_indices)
                train_edge=list(all_edge.difference(mask_edge))
                
                data[edge_types[i]].edge_index=data[edge_types[i]].edge_index[:,train_edge]

               
        return data


def Negativesub(data,num_edges,edge_type):
    random_seed=0
    edge_types=data.edge_types
    torch.manual_seed(random_seed)
    node_a_set=list(set(data[edge_types[edge_type]].edge_index[0].detach().cpu().numpy()))
    node_b_set=list(data[edge_types[edge_type]].edge_index[1].detach().cpu().numpy())
    edge_set = set([str(data[edge_types[edge_type]].edge_index[0,i].cpu().item()) + "," + str(data[edge_types[edge_type]].edge_index[1,i].cpu().item()) for i in range(data[edge_types[edge_type]].edge_index.shape[1])])
    negative_pairs=[]
    for i in range(num_edges*2):
        pair_a=random.sample(node_a_set,1)
        pair_b=random.sample(node_b_set,1)
        negative_pairs.append([pair_a[0],pair_b[0]])
    
    sampled_ind = []
    sampled_edge_set = set([])
    for i in range(len(negative_pairs)):
        node1 = negative_pairs[i][0]
        node2 = negative_pairs[i][1]
        edge_str = str(node1) + "," + str(node2)
        if not edge_str in edge_set :
            sampled_edge_set.add(edge_str)
            sampled_ind.append(i)
        if len(sampled_ind) == round(int(num_edges)):
            break
   
    negative_pairs=torch.tensor(negative_pairs)[sampled_ind,:]
    p=torch.tensor([negative_pairs[:,0].tolist()])[0]
    m=torch.tensor([negative_pairs[:,1].tolist()])[0]
 
    
    p=p.unsqueeze(0)
    m=m.unsqueeze(0)
    negative_pairs=torch.cat((p,m),dim=0)
    return negative_pairs







def Downstream_data_preprocess(args,cv,n_fold,node_type_dict):
    """
    load SL data and preprocess before training 
    """
    task_data_path=args.Task_data_path
    train_data=pd.read_csv(f"{task_data_path}/{cv}/cv_{n_fold}/train.txt",header=None,sep=' ')
    test_data=pd.read_csv(f"{task_data_path}/{cv}/cv_{n_fold}/test.txt",header=None,sep=' ')
   
    test_data.columns=[0,1,2]
    train_data[0]=train_data[0].astype(str).map(node_type_dict)
    train_data[1]=train_data[1].astype(str).map(node_type_dict)
    test_data[0]=test_data[0].astype(str).map(node_type_dict)
    test_data[1]=test_data[1].astype(str).map(node_type_dict)
    train_data=train_data.dropna()
    test_data=test_data.dropna()
    train_data[0]=train_data[0].astype(int)
    train_data[1]=train_data[1].astype(int)
    test_data[0]=test_data[0].astype(int)
    test_data[1]=test_data[1].astype(int)
    # low data scenario settings
    if args.do_low_data:
        num_sample=int(train_data.shape[0]*args.train_data_ratio)
        print(num_sample)
        train_data=train_data.sample(num_sample,replace=False,random_state=0)
        train_data.reset_index(inplace=True)
        print(f'train_data.size:{train_data.shape[0]}')

    train_node=list(set(train_data[0])|set(train_data[1]))
    train_mask=torch.zeros((27671))
    test_mask=torch.zeros((27671))
    test_node=list(set(test_data[0])|set(test_data[1]))
    train_mask[train_node]=1
    test_mask[test_node]=1
    train_mask=train_mask.bool()
    test_mask=test_mask.bool()
    num_train_node=len(train_node)
    num_test_node=len(test_node)
    return train_data,test_data,train_mask,test_mask,num_train_node,num_test_node


def Construct_loader(args,kgdata,train_mask,test_mask,node_type,num_train_node,num_test_node):
    """
    construct loader for train/test data
    """

    train_loader = HGTLoader(kgdata,
    num_samples={key: [args.sample_nodes] * args.sample_layers for key in kgdata.node_types},shuffle=False,
    batch_size=num_train_node,
    input_nodes=(node_type,train_mask),num_workers=args.num_workers)

    test_loader=HGTLoader(kgdata,
    num_samples={key: [args.sample_nodes] * args.sample_layers for key in kgdata.node_types},
    batch_size=num_test_node,
    input_nodes=(node_type,test_mask),num_workers=args.num_workers,shuffle=False)

    return train_loader,test_loader













    
