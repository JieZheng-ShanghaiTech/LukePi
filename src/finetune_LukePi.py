import argparse
import json
import logging
from time import time
import os
import torch_geometric.transforms as T
from loader_norm import HeteroDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from model import HGT
import pandas as pd
from util import Downstream_data_preprocess,Construct_loader
import pickle
import math
from torch_geometric.datasets import OGB_MAG
import torch.nn.init as init
from sklearn.metrics import f1_score, roc_auc_score,auc,balanced_accuracy_score,cohen_kappa_score,precision_recall_curve

import pickle
# import os 
# os.environ['CUDA_LAUNCH_BLOCKING']='1'




def save_model(model, optimizer_list,save_variable_list, args):
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
        'optimizer_model_state_dict': optimizer_list[0].state_dict(),
        'optimizer_classification_state_dict': optimizer_list[1].state_dict()},
        os.path.join(args.Save_model_path, 'checkpoint')
    )
                 
def set_logger(args):
    '''
    Write logs to checkpoint and console 
    '''

    if args.do_train:
        # train_log=str(linear_layer_count)+'_'+args.lr+'_'+'train.log'
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



def compute_accuracy(target,pred, pred_edge):
    
    target=target.clone().detach().cpu().numpy()
    pred=pred.clone().detach().cpu().numpy()
    pred_edge=pred_edge.clone().detach().cpu()
    scores = torch.softmax(pred_edge, 1).numpy()
    target=target.astype(int)
   
    aucu=roc_auc_score(target,scores[:,1])
    precision_tmp, recall_tmp, _thresholds = precision_recall_curve(target, pred)
    aupr = auc(recall_tmp, precision_tmp)
    f1 = f1_score(target,pred)
    kappa=cohen_kappa_score(target,pred)
    bacc=balanced_accuracy_score(target,pred)
    
    return aucu,aupr,f1,kappa,bacc


def train(args,batch_size,model_list, loader, optimizer_model,optimizer_linear_pred_edges, sldata,node_type,device):
    '''
    Train LukePi
    Args:
        batch_size: [int] the size of batch
        model_list: [str] the list of models, including a KG encoder and a set of MLP layers
        sldata: the test data
        node_type: [str] the type of nodes in the SLdata (defalut: gene/protein)
        
    '''


    model, linear_pred_edges= model_list
    
    criterion = nn.CrossEntropyLoss()
    model.train()
    linear_pred_edges.train()
    loss_sum = 0
    aucu_sum=0
    f1_sum=0
    bacc_sum=0
    kappa_sum=0
    aupr_sum=0
    edge_used=[]
    node_type=node_type
    for step,batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        node_rep= model(batch.x_dict, batch.edge_index_dict)
        node_rep=node_rep[node_type]
        node_set=pd.DataFrame(list(batch[node_type].n_id[:batch_size].squeeze().detach().cpu().numpy()))
        node_set.drop_duplicates(inplace=True,keep='first')
        node_set[1]=range(node_set.shape[0])
        node_map=dict(zip(node_set[0],node_set[1]))
        prediction_edge=sldata[[0,1]]
        prediction_label=sldata[2]
        edge_used.append(prediction_edge.shape[0])
        edge_a,edge_b=prediction_edge[0],prediction_edge[1]
        edge_a=edge_a.map(node_map)
        edge_b=edge_b.map(node_map)
        nodea_emb=node_rep[edge_a.values]
        nodeb_emb=node_rep[edge_b.values]
        edge_emb=torch.cat((nodea_emb,nodeb_emb),dim=1)
        prediction_result=linear_pred_edges(edge_emb)
        all_prediction_label=prediction_label
        all_prediction_result=prediction_result
        optimizer_model.zero_grad()
        optimizer_linear_pred_edges.zero_grad()
        all_prediction_label=torch.tensor(all_prediction_label.values).to(device)
        loss = criterion(all_prediction_result,all_prediction_label)
        all_prediction=torch.max(all_prediction_result.detach(),dim=1)[1]
        aucu,aupr,f1,kappa,bacc=compute_accuracy(all_prediction_label,all_prediction,all_prediction_result)
        loss.backward()
        optimizer_model.step()
        optimizer_linear_pred_edges.step() 
        loss_sum += float(loss.cpu().item())
        aucu_sum+=float(aucu)
        aupr_sum+=float(aupr)
        f1_sum+=float(f1)
        bacc_sum+=float(bacc)
        kappa_sum+=float(kappa)
        
  
        log = {
            'loss': loss_sum/(step+1),
            'auc':aucu_sum/(step+1),
            'aupr':aupr_sum/(step+1),
            'f1':f1_sum/(step+1),
            'bacc':bacc_sum/(step+1),
            'kappa':kappa_sum/(step+1)
        }
    
    return log




def eval(args,batch_size,model_list, loader, sldata,node_type,device):
    '''
    Test LukePi
    Args:
        batch_size: [int] the size of batch
        model_list: [str] the list of models, including a KG encoder and a set of MLP layers
        sldata: the test data
        node_type: [str] the type of nodes in the SLdata (defalut: gene/protein)

    '''
    model, linear_pred_edges= model_list
    model.eval()
    linear_pred_edges.eval()
    aucu_sum=0
    f1_sum=0
    bacc_sum=0
    kappa_sum=0
    aupr_sum=0
    edge_used=[]
    with torch.no_grad():
        for step,batch in enumerate(tqdm(loader, desc="Iteration")):
            batch = batch.to(device)
            node_rep= model(batch.x_dict, batch.edge_index_dict)
            node_rep=node_rep[node_type]
            node_set=pd.DataFrame(list(batch[args.node_type].n_id[:batch_size].squeeze().detach().cpu().numpy()))
            node_set.drop_duplicates(inplace=True,keep='first')
            node_set[1]=range(node_set.shape[0])
            node_map=dict(zip(node_set[0],node_set[1]))
            prediction_edge=sldata[[0,1]]
            prediction_label=sldata[2]
            edge_used.append(prediction_edge.shape[0])
            edge_a,edge_b=prediction_edge[0],prediction_edge[1]
            edge_a=edge_a.map(node_map)
            edge_b=edge_b.map(node_map)
            nodea_emb=node_rep[edge_a.values]
            nodeb_emb=node_rep[edge_b.values]
            edge_emb=torch.cat((nodea_emb,nodeb_emb),dim=1)
            prediction_result=linear_pred_edges(edge_emb)
            all_prediction_label=prediction_label
            all_prediction_result=prediction_result
            all_prediction=torch.max(all_prediction_result.detach(),dim=1)[1]
            all_prediction_label=torch.tensor(all_prediction_label.values).to(device)
            aucu,aupr,f1,kappa,bacc=compute_accuracy(all_prediction_label,all_prediction,all_prediction_result)
            aucu_sum+=float(aucu)
            aupr_sum+=float(aupr)
            f1_sum+=float(f1)
            bacc_sum+=float(bacc)
            kappa_sum+=float(kappa)

           
            log = {
                'auc':aucu_sum/(step+1),
                'aupr':aupr_sum/(step+1),
                 'f1':f1_sum/(step+1),
                'bacc':bacc_sum/(step+1),
                 'kappa':kappa_sum/(step+1)

        }
        return aucu_sum/(step+1),aupr_sum/(step+1),f1_sum/(step+1),bacc_sum/(step+1),kappa_sum/(step+1),log
        
  
        

        

def override_config(args):
    '''
    Override model and data configuration 
    '''
    with open(os.path.join(args.init_checkpoint, 'config.json'), 'r') as fjson:
        argparse_dict = json.load(fjson)
    
    args.method=argparse_dict['method']
    # args.epochs = argparse_dict['epochs']
    args.lr = argparse_dict['lr']
    args.num_layer = argparse_dict['num_layer']
    args.emb_dim = argparse_dict['emb_dim']
    args.mask_rate = argparse_dict['mask_rate']
    args.gnn_type=argparse_dict['gnn_type']

    if args.Save_model_path is None:
        args.Save_model_path = argparse_dict['Save_model_path']





def log_metrics(mode, step, metrics):
    '''
    Print the evaluation logs
    '''
    for metric in metrics:
        logging.info('%s %s at step %d: %f' % (mode,metric,step, metrics[metric]))

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=2,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--do_train', default=1,type=int)
    parser.add_argument('--do_test', default=1,type=int)
    parser.add_argument('--do_low_data',default=0,type=int)
    parser.add_argument('--train_data_ratio', default=1,type=float)
    parser.add_argument('--kg', default='Primekg',type=str)
    parser.add_argument('--task', default='SL',type=str)
    parser.add_argument('--cv', default='C3',type=str)
    parser.add_argument('--pretrain', default='edge_recovery_degree',type=str)
    parser.add_argument('--method', default='edge_recovery_degree',type=str)
    parser.add_argument('--node_type', default='gene/protein',type=str)
    parser.add_argument('--Task_data_path',default='./data/SL',type=str,help='Data filename to input')
    parser.add_argument('-init', '--init_checkpoint', default='pre_trained_model/Primekg_HGT_0.2_0.001', type=str)
    parser.add_argument('--Full_data_path',default='./data/BKG/kgdata.pkl', type=str)
    parser.add_argument('--Node_index_path',default=None, type=str)
    parser.add_argument('--batch_size', type=int, default=512,
                        help='input batch size for training (default: 512)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--freeze', type=int, default=1,
                        help='freeze the pre-trianed model or not')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--lr1', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--decay', type=float, default=0,
                        help='weight decay (default: 0)')
    parser.add_argument('--mask_rate', type=float, default=0.2,
                        help='mask rate (default: 0.2)')
    parser.add_argument('--num_layer', type=int, default=3,
                        help='number of GNN message passing layers (default: 3).')
    parser.add_argument('--emb_dim', type=int, default=128,
                        help='embedding dimensions (default: 128)')
    parser.add_argument('--num_heads', type=int, default=4,
                        help='number of GNN head.')
    parser.add_argument('--dropout_ratio', type=float, default=0.2,
                        help='dropout ratio (default: 0.2)')
    parser.add_argument('--sample_nodes', type=int, default=1024,
                        help='the number of sampled nodes for each type ')
    parser.add_argument('--sample_layers', type=int, default=4,
                        help='the number of sampled iterations ')
    parser.add_argument('--gnn_type', type=str, default="HGT")
    parser.add_argument('--save_checkpoint_steps', default=10, type=int)
    parser.add_argument('--log_steps', default=1, type=int, help='train log every xx steps')
    parser.add_argument('--Save_model_path', default='./result',type=str, help='filename to output the model')
    parser.add_argument('--seed', type=int, default=0, help = "Seed for splitting dataset.")
    parser.add_argument('--num_workers', type=int, default =16, help='number of workers for dataset loading')
    args = parser.parse_args()
    if args.freeze:
        if args.do_low_data:
            args.Save_model_path=args.Save_model_path+'/'+args.task+'/'+args.method+'_'+args.kg+'_'+args.gnn_type+'_'+str(args.mask_rate)+'_'+str(args.lr)+'freeze'+'/'+'low_data'+'/'+args.cv
        else:
            args.Save_model_path=args.Save_model_path+'/'+args.task+'/'+args.method+'_'+args.kg+'_'+args.gnn_type+'_'+str(args.mask_rate)+'_'+str(args.lr)+'freeze'+'/'+args.cv

    else:
        if args.do_low_data:
            args.Save_model_path=args.Save_model_path+'/'+args.task+'/'+args.method+'_'+args.kg+'_'+args.gnn_type+'_'+str(args.mask_rate)+'_'+str(args.lr)+'_'+str(args.epochs)+'/'+'low_data'+'/'+args.cv
        else:
            args.Save_model_path=args.Save_model_path+'/'+args.task+'/'+args.method+'_'+args.kg+'_'+args.gnn_type+'_'+str(args.mask_rate)+'_'+str(args.lr)+'_'+str(args.epochs)+'/'+args.cv
       
    
    
    if (not args.do_train): 
        raise ValueError('one of train/val/test mode must be choosed.')
    
    if args.init_checkpoint:  
        override_config(args)
        
    elif args.Task_data_path is None: 
        raise ValueError('one of init_checkpoint/data_path must be choosed.')
    if args.do_train and args.Save_model_path is None:
        raise ValueError('Where do you want to save your trained model?')
    
    if args.Save_model_path and not os.path.exists(args.Save_model_path): 
            os.makedirs(args.Save_model_path)
    
    set_logger(args)
    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
  
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    logging.info('Task: %s' % args.task) 
    logging.info('Data Path: %s' % args.Task_data_path)  
    logging.info('GNN type: %s' % args.gnn_type)

    logging.info('Model Parameter Configuration:')
    

  

    #set up dataset
    with open (args.Full_data_path,'rb') as f:
        kgdata=pickle.load(f)
   
    with open("./data/BKG/node_index_dic.json",'rb') as f:
        node_index=json.load(f)


    
    gene_protein=node_index[args.node_type] 
    eval_metric_folds={'fold':[],'auc':[],'aupr':[],'f1':[],'bacc':[],'kappa':[]}
    node_type=args.node_type
    num_nodes_type=len(kgdata.node_types)
    num_edge_type=len(kgdata.edge_types)
    num_nodes=kgdata.num_nodes
    input_node_embeddings = torch.nn.Embedding(num_nodes_type, 16)
    torch.nn.init.xavier_uniform_(input_node_embeddings.weight.data)
    for i in range(len(kgdata.node_types)):
        num_repeat=kgdata[kgdata.node_types[i]].x.shape[0]
        kgdata[kgdata.node_types[i]].x =input_node_embeddings(torch.tensor(i)).repeat([num_repeat,1]).detach()
    

    # initiliaze the model
    for i in range(1,6):
        logging.info(f'Fold_{i} training...')
        n_fold=i
        #initialize models
        model = HGT(kgdata,2*args.emb_dim,args.emb_dim,args.num_heads,args.num_layer).to(args.device)
        if args.freeze:
            for param in model.parameters():
                param.requires_grad = False
        else:
            print('training')
       
        linear_pred_edges=torch.nn.Sequential(
                                                torch.nn.Linear(2*args.emb_dim,args.emb_dim),
                                                torch.nn.ReLU(), 
                                                # torch.nn.Dropout(0.2),
                                                torch.nn.Linear(args.emb_dim,64),
                                                torch.nn.ReLU(),
                                                # torch.nn.Dropout(0.2),
                                                torch.nn.Linear(64,32),
                                                torch.nn.ReLU(),
                                                # torch.nn.Dropout(0.2),
                                                torch.nn.Linear(32, 2)).to(device)

        model_list = [model, linear_pred_edges]
        
        
        linear_layer_count = 0
        
        for layer in linear_pred_edges:
            if isinstance(layer, nn.Linear):
                linear_layer_count += 1
        


        #set up optimizers
        optimizer_model = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
        optimizer_linear_pred_edges= optim.Adam(linear_pred_edges.parameters(), lr=args.lr1, weight_decay=args.decay)
        optimizer_list=[optimizer_model,optimizer_linear_pred_edges]
    
        if args.init_checkpoint: 
            # Restore model from checkpoint directory  
            logging.info('Loading checkpoint %s...' % args.init_checkpoint)
            checkpoint = torch.load(os.path.join(args.init_checkpoint, 'checkpoint'))
            init_step = checkpoint['step']
            model.load_state_dict(checkpoint['model_state_dict'])
            if args.do_train:
                optimizer_model.load_state_dict(checkpoint['optimizer_model_state_dict'])
                # optimizer_linear_pred_edges.load_state_dict(checkpoint['optimizer_classification_state_dict'])

        else:
            logging.info('Ramdomly Initializing %s Model...' % args.gnn_type)  
            init_step = 0

        step = init_step 
        train_data,test_data,train_mask,test_mask,num_train_node,num_test_node=Downstream_data_preprocess(args,args.cv,n_fold,gene_protein)
        train_loader,test_loader=Construct_loader(args,kgdata,train_mask,test_mask,node_type,num_train_node,num_test_node)
        logging.info('Start Training...')
        logging.info('init_step = %d' % init_step)
        logging.info('num_train_node = %d' % num_train_node)
        logging.info('num_test_node = %d' % num_test_node)
        if args.do_train:
            logging.info('learning_rate = %d' %args.lr)
            training_logs = []
            testing_logs=[]
           
            auc_sum_fold=[]
            aupr_sum_fold=[]
            f1_sum_fold=[]
            bacc_sum_fold=[]
            kappa_sum_fold=[]
            for step in range(1, args.epochs+1):
                log=train(args,num_train_node,model_list,train_loader,optimizer_model,optimizer_linear_pred_edges,train_data,args.node_type,args.device)
                training_logs.append(log)
                eval_auc,eval_aupr,eval_f1,eval_bacc,eval_kappa,testing_log=eval(args,num_test_node,model_list,test_loader,test_data,args.node_type,args.device)
                auc_sum_fold.append(eval_auc)
                aupr_sum_fold.append(eval_aupr)
                f1_sum_fold.append(eval_f1)
                bacc_sum_fold.append(eval_bacc)
                kappa_sum_fold.append(eval_kappa)
                testing_logs.append(testing_log)

                
            
                #log training and test information
                if step % args.log_steps == 0:
                    training_metrics = {}
                    testing_metrics={}
                    for metric in training_logs[0].keys():
                        training_metrics[metric] = sum([log[metric] for log in training_logs])/len(training_logs)
                    logging.info('loss on training Dataset...')
                    logging.info('eval on test  Dataset...')
                    log_metrics('Training average', step, training_metrics)
                    for metric in testing_logs[0].keys():
                        testing_metrics[metric] = sum([log[metric] for log in testing_logs])/len(testing_logs)
                    log_metrics('Test average', step, testing_metrics)
                    training_logs = []
                    testing_logs=[]

                if step % args.save_checkpoint_steps == 0: # save model
                    save_variable_list = {
                        'step': step, 
                        'current_learning_rate': args.lr,
                        'model':args.gnn_type,
                    }
                    save_model(model, optimizer_list,save_variable_list, args)
                    print(f'fold_{n_fold}_auc:{round(auc_sum_fold[-1],4)}')
                    print(f'fold_{n_fold}_aupr:{round(aupr_sum_fold[-1],4)}')
                    print(f'fold_{n_fold}_f1:{round(f1_sum_fold[-1],4)}')
                    print(f'fold_{n_fold}_bacc:{round(bacc_sum_fold[-1],4)}')
                    print(f'fold_{n_fold}_kappa:{round(kappa_sum_fold[-1],4)}')
                # store the result
            eval_metric_folds['fold'].append(n_fold)
            eval_metric_folds['auc'].append(round(auc_sum_fold[-1],4))   
            eval_metric_folds['f1'].append(round(f1_sum_fold[-1],4)) 
            eval_metric_folds['aupr'].append(round(aupr_sum_fold[-1],4)) 
            eval_metric_folds['bacc'].append(round(bacc_sum_fold[-1],4)) 
            eval_metric_folds['kappa'].append(round(kappa_sum_fold[-1],4)) 
        
    eval_metric_folds=pd.DataFrame(eval_metric_folds)
    eval_metric_folds.loc[5,'fold']='average'
    eval_metric_folds.loc[5,'auc']=round(eval_metric_folds['auc'].mean(),4)
    eval_metric_folds.loc[5,'aupr']=round(eval_metric_folds['aupr'].mean(),4)
    eval_metric_folds.loc[5,'f1']=round(eval_metric_folds['f1'].mean(),4)
    eval_metric_folds.loc[5,'bacc']=round(eval_metric_folds['bacc'].mean(),4)
    eval_metric_folds.loc[5,'kappa']=round(eval_metric_folds['kappa'].mean(),4)
    eval_metric_folds.loc[6,'fold']='std'
    eval_metric_folds.loc[6,'auc']=round(eval_metric_folds.loc[:4,'auc'].std(),4)
    eval_metric_folds.loc[6,'aupr']=round(eval_metric_folds.loc[:4,'aupr'].std(),4)
    eval_metric_folds.loc[6,'f1']=round(eval_metric_folds.loc[:4,'f1'].std(),4)
    eval_metric_folds.loc[6,'bacc']=round(eval_metric_folds.loc[:4,'bacc'].std(),4)
    eval_metric_folds.loc[6,'kappa']=round(eval_metric_folds.loc[:4,'kappa'].std(),4)

    if args.do_low_data:
        eval_metric_folds.to_csv(args.Save_model_path+'/'+args.cv+'_'+str(args.train_data_ratio)+'_'+str(args.lr1)+'_'+str(linear_layer_count)+'_''result_eval.csv',index=False)
    else:
        eval_metric_folds.to_csv(args.Save_model_path+'/'+args.cv+'_'+str(args.lr1)+'_'+str(linear_layer_count)+'_''result_eval.csv',index=False)
    

            
if __name__ == "__main__":
    s=time()
    main()
    e=time()
    print(f"Total running time: {round(e - s, 2)}s")
