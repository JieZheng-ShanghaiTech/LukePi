#!/bin/sh
Code_path=Code

Data_path=dataprimekg
Save_path=model


#Parameters config
method=edge_recovery_degree
kg=Primekg

gnn_type=HGT
Full_data_path='./data/BKG/kgdata.pkl'
Node_index_path='./data/BKG/node_index_dic.json'

cv='C3'
node_type='gene/protein'
sample_nodes=1024
sample_layers=4
num_heads=4
num_layer=4
epoch=50
lr=0.001
lr1=0.003
finetune='true'
freeze=1
mask_rate=0.2
do_low_data=0


if [ "$finetune" = 'true' ]; then
    echo "Start finetune...."
  
    init_checkpoint='./pre_trained_model/Primekg_HGT_0.2_0.001'
    Save_model_path='./result'

    python finetune_LukePi.py  --gnn_type $gnn_type --device 0 --do_low_data $do_low_data \
    --epochs $epoch  --sample_nodes $sample_nodes --kg $kg --Full_data_path $Full_data_path\
    --num_heads $num_heads --num_layer $num_layer --method $method --Node_index_path $Full_data_path\
    --sample_layers $sample_layers  --epochs $epoch --lr $lr --lr1 $lr1 --cv $cv --node_type $node_type\
    --Save_model_path $Save_model_path --init_checkpoint $init_checkpoint --mask_rate $mask_rate --freeze $freeze
else
    echo "Start training...."
    Save_model_path='./result'
    python finetune_original.py  --gnn_type $gnn_type --device 1 --do_low_data $do_low_data --train_data_ratio $train_data_ratio \
    --epochs $epoch  --sample_nodes $sample_nodes --kg $kg --Full_data_path $Full_data_path\
    --num_heads $num_heads --num_layer $num_layer --Node_index_path $Node_index_path\
    --sample_layers $sample_layers  --epochs $epoch --lr $lr --cv $cv --node_type $node_type\
    --Save_model_path $Save_model_path  --freeze $freeze 
fi