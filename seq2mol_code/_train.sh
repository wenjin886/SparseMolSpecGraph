#!/bin/bash

python train.py \
    --exp_name "seq2mol_updateTokenizer_woformula" \
    --use_formula False \
    --exp_save_path "../../exp/exp_hnmr_seq2mol" \
    --data_dir_path "/rds/projects/c/chenlv-ai-and-chemistry/wuwj/NMR_MS/sparsespec2graph/multimodal-spectroscopic-dataset/runs/runs_new_onmt_w_formula/h_nmr/data" \
    --src_tokenizer_path "./tokenizer/src_tokenizer/nmr_formula_tokenizer_fast/tokenizer.json" \
    --tgt_tokenizer_path "./tokenizer/tgt_tokenizer/smiles_tokenizer_fast/tokenizer.json" \
    --warm_up_step 8000 \
    --lr 2 \
    --batch_size 512 \
    --max_epochs 500 \
    --monitor val_acc \
    --monitor_mode max \
    # --code_test

# python train.py \
#     --exp_name "seq2mol_updateTokenizer" \
#     --exp_save_path "../../exp/exp_hnmr_seq2mol" \
#     --data_dir_path "/rds/projects/c/chenlv-ai-and-chemistry/wuwj/NMR_MS/sparsespec2graph/multimodal-spectroscopic-dataset/runs/runs_new_onmt_w_formula/h_nmr/data" \
#     --src_tokenizer_path "./tokenizer/src_tokenizer/nmr_formula_tokenizer_fast/tokenizer.json" \
#     --tgt_tokenizer_path "./tokenizer/tgt_tokenizer/smiles_tokenizer_fast/tokenizer.json" \
#     --warm_up_step 8000 \
#     --lr 2 \
#     --batch_size 512 \
#     --max_epochs 500 \
#     --monitor val_acc \
#     --monitor_mode max \
    # --resume \
    # --checkpoint_path /rds/projects/c/chenlv-ai-and-chemistry/wuwj/NMR_MS/sparsespec2graph/exp/exp_hnmr_seq2mol/seq2mol_15-58_17-07-2025/last.ckpt \
    # --code_test \
