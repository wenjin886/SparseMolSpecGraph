#!/bin/bash
echo "Training..."
python train_val.py \
    --exp_name h_nmr_graph_nf512_edgelinear \
    --exp_save_path ../exp/exp_hnmr \
    --dataset_path ../Dataset/h_nmr/train_val_test_set \
    --dataset_info_path ../Dataset/h_nmr/h_nmr.json \
    --wandb_project NMR-Graph \
    --spec_type h_nmr \
    --max_epochs 50 \
    --batch_size 256 \
    --lr 1 \
    # --code_test