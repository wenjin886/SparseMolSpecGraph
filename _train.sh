#!/bin/bash
echo "Training..."
python train_val.py \
    --exp_name test_nmr_graph_nHcls \
    --exp_save_path ../exp/exp_example \
    --dataset_path ../example_data/example_hnmr_masked.pt \
    --dataset_info_path ../example_data/example_hnmr.json \
    --wandb_project NMR-Graph \
    --spec_type h_nmr \
    --code_test