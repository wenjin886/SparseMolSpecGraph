#!/bin/bash
echo "Training..."

python train.py \
    --exp_name graphtmr_nmr6L_fs0L_cls2mol_bins0_cls_initpara \
    --rel_pos_bins 0 \
    --spec_node_type "cls" \
    --spec_encoder_layer 6 \
    --spec_formula_encoder_layer 0 \
    --data_dir_path "/rds/projects/c/chenlv-ai-and-chemistry/wuwj/NMR_MS/sparsespec2graph/multimodal-spectroscopic-dataset/runs/runs_new_onmt_w_formula/h_nmr/data" \
    --formula_tokenizer_path "./tokenizers/formula_tokenizer_fast/tokenizer.json" \
    --nmr_tokenizer_path "./tokenizers/nmr_tokenizer_fast/tokenizer.json" \
    --tgt_tokenizer_path "./tokenizers/smiles_tokenizer_fast/tokenizer.json" \
    --warm_up_step 8000 \
    --lr 2 \
    --max_epochs 500 \
    --monitor val_acc \
    --monitor_mode max \
    --batch_size 512 \

# python train.py \
#     --exp_name code_test_graphtmr_nmr4L_fs4L \
#     --rel_pos_bins 0 \
#     --spec_node_type "concat" \
#     --spec_encoder_layer 4 \
#     --spec_formula_encoder_layer 4 \
#     --data_dir_path "/rds/projects/c/chenlv-ai-and-chemistry/wuwj/NMR_MS/sparsespec2graph/multimodal-spectroscopic-dataset/runs/runs_new_onmt_w_formula/h_nmr/data" \
#     --formula_tokenizer_path "./tokenizers/formula_tokenizer_fast/tokenizer.json" \
#     --nmr_tokenizer_path "./tokenizers/nmr_tokenizer_fast/tokenizer.json" \
#     --tgt_tokenizer_path "./tokenizers/smiles_tokenizer_fast/tokenizer.json" \
#     --warm_up_step 8000 \
#     --lr 2 \
#     --max_epochs 500 \
#     --monitor val_acc \
#     --monitor_mode max \
#     --code_test \
#     --batch_size 4 \