#!/bin/bash
echo "Training..."
# python train_val.py \
#     --exp_name h_nmr_graph_multweight \
#     --exp_save_path ../exp/exp_hnmr \
#     --dataset_path ../Dataset/h_nmr/train_val_test_set \
#     --dataset_info_path ../Dataset/h_nmr/h_nmr.json \
#     --wandb_project NMR-Graph \
#     --spec_type h_nmr \
#     --max_epochs 50 \
#     --batch_size 256 \
#     --lr 1 \
#     # --code_test

python train_val.py \
    --exp_name labelmapped_lrinnodedim_innodedim320_3predhedlayer_1n2n \
    --num_heads 4 \
    --mult_embed_dim 128 \
    --nH_embed_dim 64 \
    --c_w_embed_dim 64 \
    --exp_save_path ../exp/exp_hnmr \
    --dataset_info_path ../Dataset/h_nmr/h_nmr.json \
    --wandb_project NMR-Graph \
    --spec_type h_nmr \
    --label_type mapped \
    --max_epochs 50 \
    --batch_size 256 \
    --lr 1 \
    --dataset_path ../Dataset/h_nmr/mapped_train_val_test_set \
    # --code_test \
    # --dataset_path /rds/projects/c/chenlv-ai-and-chemistry/wuwj/NMR_MS/sparsespec2graph/Dataset/h_nmr/masked_h_nmr_label_mapped.pt \
    # --splitted_set_save_dir_name mapped_train_val_test_set\