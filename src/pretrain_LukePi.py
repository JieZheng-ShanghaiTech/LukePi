import argparse
import json
import logging
from time import time
import os
import torch_geometric.transforms as T
from loader_norm import HeteroDataset
from torch_geometric.loader import HGTLoader, NeighborLoader
# from dataloader import DataLoaderMasking 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from model import HGT
import pandas as pd
from util import EdgeMaskNegative
import pickle
import math
from torch_geometric.datasets import OGB_MAG
import torch.nn.init as init
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score,roc_auc_score,precision_recall_curve,auc



def compute_accuracy(target,pred, pred_edge):
    
    target=target.clone().detach().cpu().numpy()
    pred=pred.clone().detach().cpu().numpy()
    pred_edge=pred_edge.clone().detach().cpu()
    scores = torch.softmax(pred_edge, 1).numpy()
   
    target=target.astype(int)
    aucu=roc_auc_score(np.eye(51)[target],scores,multi_class='ovo')
    # precision_tmp, recall_tmp, _thresholds = precision_recall_curve(target, pred)
    # aupr = auc(recall_tmp, precision_tmp)
    f1 = f1_score(target,pred, average='micro', zero_division=0)
    
    return aucu,f1

def compute_accuracy_node(target,pred, pred_edge):
    
    target=target.clone().detach().cpu().numpy()
    pred=pred.clone().detach().cpu().numpy()
    pred_edge=pred_edge.clone().detach().cpu()
    scores = torch.softmax(pred_edge, 1).numpy()
   
    target=target.astype(int)
    aucu=roc_auc_score(np.eye(9)[target],scores,multi_class='ovo')
    # precision_tmp, recall_tmp, _thresholds = precision_recall_curve(target, pred)
    # aupr = auc(recall_tmp, precision_tmp)
    f1 = f1_score(target,pred, average='micro', zero_division=0)
    
    return aucu,f1


def train(args, Edge_mask_Negative,model_list, loader, optimizer_model,optimizer_linear_pred_edges,optimizer_linear_pred_nodes, device):
    model, linear_pred_edges,linear_pred_nodes= model_list
    criterion = nn.CrossEntropyLoss()
    model.train()
    linear_pred_edges.train()
    linear_pred_nodes.train()
    loss_sum = 0
    aucu_sum=0
    f1_sum=0
    aucu_sum_node=0
    f1_sum_node=0

    for step,batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        batch=Edge_mask_Negative(batch)
        node_rep= model(batch.x_dict, batch.edge_index_dict)
        all_prediction_label=[]
        all_prediction_result=[]
       
        all_prediction_label_node=[]
        all_prediction_result_node=[]

        
        #--------------LukePi ------------
        for i in range(len(batch.edge_types)):
            prediction_edge=batch[batch.edge_types[i]].prediction_edge
            prediction_label=batch[batch.edge_types[i]].prediction_label
            nodea,rel,nodeb=batch.edge_types[i]
            edge_a, edge_b=prediction_edge[0],prediction_edge[1]
            nodea_emb=node_rep[nodea][edge_a]
            nodeb_emb=node_rep[nodeb][edge_b]
            edge_emb=torch.cat((nodea_emb,nodeb_emb),dim=1)
            prediction_result=linear_pred_edges(edge_emb)
            all_prediction_label.append(prediction_label)
            all_prediction_result.append(prediction_result)
          
        #---------------node ------------
        for i in range(len(batch.node_types)):
            prediction_label_node=batch[batch.node_types[i]].degree_label
            type_node_emb=node_rep[batch.node_types[i]]
            prediction_result_node=linear_pred_nodes(type_node_emb)
            all_prediction_label_node.append(prediction_label_node)
            all_prediction_result_node.append(prediction_result_node)

        optimizer_model.zero_grad()
        optimizer_linear_pred_edges.zero_grad()
        optimizer_linear_pred_nodes.zero_grad()
        
        all_prediction_label=torch.cat(all_prediction_label,dim=0).to(device)
        all_prediction_result=torch.cat(all_prediction_result,dim=0)
        #node 
        all_prediction_label_node=torch.cat(all_prediction_label_node,dim=0).to(device)
        all_prediction_result_node=torch.cat(all_prediction_result_node,dim=0)
        loss_edge= criterion(all_prediction_result,all_prediction_label[:,0].long())
        loss_node=criterion(all_prediction_result_node,all_prediction_label_node[:,0].long())
      
        
        loss=loss_edge+loss_node
        #Node degree prediction
        all_prediction=torch.max(all_prediction_result.detach(),dim=1)[1]
        all_prediction_node=torch.max(all_prediction_result_node.detach(),dim=1)[1]

        aucu,f1=compute_accuracy(all_prediction_label[:,0],all_prediction,all_prediction_result)
        aucu_node,f1_node=compute_accuracy_node(all_prediction_label_node[:,0],all_prediction_node,all_prediction_result_node)
        loss.backward()
        optimizer_model.step()
        optimizer_linear_pred_edges.step()
        optimizer_linear_pred_nodes.step()
        loss_sum += float(loss.cpu().item())
        aucu_sum+=float(aucu)
        # aupr_sum+=float(aupr)
        f1_sum+=float(f1)


        aucu_sum_node+=float(aucu_node)
        f1_sum_node+=float(f1_node)
        
        log = {
            'loss': loss_sum/(step+1),
            'auc':aucu_sum/(step+1),
            'f1':f1_sum/(step+1),
            'auc_node':aucu_sum_node/(step+1),
            'f1_node':f1_sum_node/(step+1)
        }
    



    return log


def override_config(args):
    '''
    Override model and data configuration 
    '''
    with open(os.path.join(args.init_checkpoint, 'config.json'), 'r') as fjson:
        argparse_dict = json.load(fjson)
    
    args.method=argparse_dict['method']
    #training config
    args.epochs = argparse_dict['epochs']
    args.batch_size=argparse_dict['batch_size']
    args.sample_nodes=argparse_dict['sample_nodes']
    args.sample_layers=argparse_dict['sample_layers']
    args.lr = argparse_dict['lr']
    args.emb_dim = argparse_dict['emb_dim']
    args.mask_rate = argparse_dict['mask_rate']
    #model config
    args.gnn_type=argparse_dict['gnn_type']
    args.num_heads=argparse_dict['num_heads']
    args.num_layer = argparse_dict['num_layer']

    if args.Save_model_path is None:
        args.Save_model_path = argparse_dict['Save_model_path']



def save_model(model, optimizer_model,optimizer_linear_pred_edges,optimizer_linear_pred_nodes,save_variable_list, args):
    '''
    Save the parameters of the model and the optimizer,
    as well as some other variables such as step and learning_rate
    '''
    
    argparse_dict = vars(args)
    print(args.Save_model_path)
    with open(os.path.join(args.Save_model_path, 'config.json'), 'w') as fjson:
        json.dump(argparse_dict, fjson)

    torch.save({
        **save_variable_list,
        'model_state_dict': model.state_dict(),
        'optimizer_model_state_dict': optimizer_model.state_dict(),
        'optimizer_classification_state_dict': optimizer_linear_pred_edges.state_dict(),
        'optimizer_node_classification_state_dict': optimizer_linear_pred_nodes.state_dict()},

        os.path.join(args.Save_model_path, 'checkpoint')
    )

def set_logger(args):
    '''
    Write logs to checkpoint and console 
    '''

    if args.do_train:
        log_file = os.path.join(args.Save_model_path or args.init_checkpoint, 'train.log') 
    else:
        log_file = os.path.join(args.Save_model_path or args.init_checkpoint, 'test.log')  
    
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s', 
        level=logging.INFO,  # 
        datefmt='%Y-%m-%d %H:%M:%S', 
        filename=log_file, 
        filemode='w'  
    )
    console = logging.StreamHandler() # 
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s') 
    console.setFormatter(formatter) 
    logging.getLogger('').addHandler(console) 

def log_metrics(mode, step, metrics):
    '''
    Print the evaluation logs
    '''
    for metric in metrics:
        logging.info('%s %s at step %d: %f' % (mode,metric,step, metrics[metric]))

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=1,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--do_train', default=1,type=int)
    parser.add_argument('--kg', default='PrimeKG',type=str)
    parser.add_argument('--Full_data_path',default='../data/kgdata.pkl',type=str,help='Data filename to input')
    parser.add_argument('--method',default='edge_recovery',type=str,help='pre-training method used')
    parser.add_argument('-init', '--init_checkpoint', default=None, type=str)
    parser.add_argument('--batch_size', type=int, default=512,
                        help='input batch size for training (default: 256)')
    parser.add_argument('--epochs', type=int, default=1,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.005,
                        help='learning rate (default: 0.001)')   
    parser.add_argument('--decay', type=float, default=0,
                        help='weight decay (default: 0)')
    parser.add_argument('--num_layer', type=int, default=4,
                        help='number of GNN message passing layers (default:4 ).')
    parser.add_argument('--num_heads', type=int, default=4,
                        help='number of GNN head.')
    parser.add_argument('--emb_dim', type=int, default=128,
                        help='embedding dimensions (default: )')
    parser.add_argument('--dropout_ratio', type=float, default=0,
                        help='dropout ratio (default: 0)')
    parser.add_argument('--mask_rate', type=float, default=0.2,
                        help='dropout ratio (default: 0.2)')
    parser.add_argument('--sample_nodes', type=int, default=1024,
                        help='the number of sampled nodes for each type ')
    parser.add_argument('--sample_layers', type=int, default=4,
                        help='the number of sampled iterations ')
    parser.add_argument('--gnn_type', type=str, default="HGT")
    parser.add_argument('--save_checkpoint_steps', default=5, type=int)
    parser.add_argument('--log_steps', default=1, type=int, help='train log every xx steps')
    parser.add_argument('--Save_model_path', type=str,default='../pre_trained_model',help='filename to output the model')
    # parser.add_argument('--model_file_classfication', type=str, default = '../pre_trained_model', help='filename to output the model')
    parser.add_argument('--seed', type=int, default=0, help = "Seed for splitting dataset.")
    parser.add_argument('--num_workers', type=int, default =16, help='number of workers for dataset loading')
    args = parser.parse_args()
    
    #save model path
    # args.device='cpu'
    args.Save_model_path=args.Save_model_path+'/'+args.method+'/'+args.kg+'_'+args.gnn_type+'_'+str(args.mask_rate)+'_'+str(args.lr)+'_'+str(args.emb_dim)
    
    

    if (not args.do_train): 
        raise ValueError('one of train/val/test mode must be choosed.')
    if args.init_checkpoint:  
        override_config(args)
    elif args.Full_data_path is None: 
        raise ValueError('one of init_checkpoint/data_path must be choosed.')
    if args.Save_model_path is None:
        raise ValueError('Where do you want to save your trained model?')
    
    if args.Save_model_path and not os.path.exists(args.Save_model_path): 
        os.makedirs(args.Save_model_path)
    
    set_logger(args)
    torch.manual_seed(0)
    np.random.seed(0)
    # device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    # device=torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    logging.info('Method: %s' % args.method) 
    logging.info('Data Path: %s' % args.Full_data_path)  
    logging.info('GNN type: %s' % args.gnn_type)
    logging.info("Save model Path:%s" %args.Save_model_path)
    logging.info('Model Parameter Configuration:')
  

    #set up dataset
    path=args.Full_data_path
    with open(path,'rb') as f:
        dataset=pickle.load(f)
    data=dataset
    #initiliaze the node embeddings for each type
    num_nodes_type=len(data.node_types)
    num_edge_type=len(data.edge_types)
    num_nodes=data.num_nodes
    input_node_embeddings = torch.nn.Embedding(num_nodes_type, 16)
    torch.nn.init.xavier_uniform_(input_node_embeddings.weight.data)
    for i in range(len(data.node_types)):
        num_repeat=data[data.node_types[i]].x.shape[0]
        data[data.node_types[i]].x =input_node_embeddings(torch.tensor(i)).repeat([num_repeat,1]).detach()
    # input_node_embeddings = torch.nn.Embedding(num_nodes, 16)
    # for i in range(len(data.node_types)):
    # # num_repeat=kgdata[kgdata.node_types[0]].x.shape[0]
    #     l=data[data.node_types[i]].x
    #     data[data.node_types[i]].x =torch.squeeze(input_node_embeddings(torch.tensor(l))).detach()
    #initiliaze the node embeddings for each node
    # input_node_embeddings = torch.nn.Embedding(num_nodes, 16)
    # for i in range(len(data.node_types)):
    # # num_repeat=kgdata[kgdata.node_types[0]].x.shape[0]
    #     l=data[data.node_types[i]].x
    #     data[data.node_types[i]].x =torch.squeeze(input_node_embeddings(torch.tensor(l))).detach()

    # Pre-training method
    Edge_mask_Negative=EdgeMaskNegative(args.mask_rate,args.device)
  
    # loader = HGTLoader(data,
    # # Sample args.samplez_nodes nodes per type and per iteration for args.sample_layers iterations
    # num_samples={key: [args.sample_nodes] * args.sample_layers for key in data.node_types},
    # # Use a batch size of 128 for sampling training nodes of type paper
    # batch_size=args.batch_size,
    # input_nodes=('gene/protein',torch.ones(data['gene/protein'].x.shape[0],dtype=torch.bool)),num_workers=args.num_workers)
    loader = HGTLoader(data,
    # Sample args.samplez_nodes nodes per type and per iteration for args.sample_layers iterations
    num_samples={key: [args.sample_nodes] * args.sample_layers for key in data.node_types},
    # Use a batch size of 128 for sampling training nodes of type paper
    batch_size=args.batch_size,
    input_nodes=('gene/protein',torch.ones(data['gene/protein'].x.shape[0],dtype=torch.bool)),num_workers=args.num_workers)




    #set up models, one for pre-training and one for context embeddings
    model = HGT(data,2*args.emb_dim,args.emb_dim,args.num_heads,args.num_layer).to(args.device)
    
    linear_pred_edges=torch.nn.Sequential(torch.nn.Linear(2*args.emb_dim,args.emb_dim),torch.nn.ReLU(), torch.nn.Linear(args.emb_dim,num_edge_type+1)).to(args.device)

    linear_pred_nodes=torch.nn.Sequential(torch.nn.Linear(args.emb_dim,args.emb_dim),torch.nn.ReLU(), torch.nn.Linear(args.emb_dim,9)).to(args.device)

    model_list = [model, linear_pred_edges,linear_pred_nodes]

    #set up optimizers
    optimizer_model = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
    optimizer_linear_pred_edges= optim.Adam(linear_pred_edges.parameters(), lr=args.lr, weight_decay=args.decay)
    optimizer_linear_pred_nodes= optim.Adam(linear_pred_nodes.parameters(), lr=args.lr, weight_decay=args.decay)

    # optimizer_list = [optimizer_model, optimizer_linear_pred_edges]
   
    if args.init_checkpoint: 
        # Restore model from checkpoint directory  
        logging.info('Loading checkpoint %s...' % args.init_checkpoint)
        checkpoint = torch.load(os.path.join(args.init_checkpoint, 'checkpoint'))
        init_step = checkpoint['step']
        model.load_state_dict(checkpoint['model_state_dict'])
        if args.do_train:
            optimizer_model.load_state_dict(checkpoint['optimizer_model_state_dict'])
            optimizer_linear_pred_edges.load_state_dict(checkpoint['optimizer_classification_state_dict'])
            optimizer_linear_pred_nodes.load_state_dict(checkpoint['optimizer_node_classification_state_dict'])

    else:
        logging.info('Ramdomly Initializing %s Model...' % args.gnn_type)  
        init_step = 0
    
    step = init_step 
    
    logging.info('Start Training...')
    logging.info('init_step = %d' % init_step)
    logging.info('batch_size = %d' % args.batch_size)
    logging.info('sample_nodes = %d' % args.sample_nodes)
    logging.info('sample_layers = %d' % args.sample_layers)

    # epoch_loss,epoch_auc,epoch_f1=train(args,Edge_mask,model_list, loader, optimizer_model,optimizer_linear_pred_edges, args.device)
    # print(f'epoch_loss:{epoch_loss}')
    # print(f'epoch_auc:{epoch_auc}')
    # # print(f'epoch_aupr:{epoch_aupr}')
    # print(f'epoch_f1:{epoch_f1}')

    logging.info('learning_rate = %f' %round(args.lr,4))

    training_logs = []
    for step in range(1, args.epochs+1):
        log= train(args, Edge_mask_Negative,model_list, loader, optimizer_model,optimizer_linear_pred_edges,optimizer_linear_pred_nodes, device=args.device)
        training_logs.append(log)

        if step % args.save_checkpoint_steps == 0: # save model
            save_variable_list = {
                'step': step, 
                'current_learning_rate': args.lr,
                'model':args.gnn_type,
            }
            save_model(model, optimizer_model,optimizer_linear_pred_edges, optimizer_linear_pred_nodes,save_variable_list, args)
        
    
        #store log information
        if step % args.log_steps == 0:
            metrics = {}
            for metric in training_logs[0].keys():
                metrics[metric] = sum([log[metric] for log in training_logs])/len(training_logs)
            logging.info('Metric on training Dataset...')
            log_metrics('Training average', step, metrics)
            training_logs = []
         
    save_variable_list = {
            'step': step, 
            'current_learning_rate':  args.lr,
           
        }
    #Save model 
    save_model(model, optimizer_model,optimizer_linear_pred_edges,optimizer_linear_pred_nodes, save_variable_list, args)
            

    
       


if __name__ == "__main__":
    s=time()
    main()
    e=time()
    print(f"Total running time: {round(e - s, 2)}s")

