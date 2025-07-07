#!/bin/bash
echo "Training..."
python train_val.py \
    --exp_name test_h_nmr_graph \
    --exp_save_path ../exp/exp_hnmr \
    --dataset_path ../Dataset/h_nmr/h_nmr_masked.pt \
    --dataset_info_path ../Dataset/h_nmr/h_nmr.json \
    --wandb_project NMR-Graph \
    --spec_type h_nmr \
    --code_test