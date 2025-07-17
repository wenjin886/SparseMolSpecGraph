#!/bin/bash

python train.py \
    --exp_name "test" \
    --exp_save_path "../exp" \
    --data_dir_path "../example_data/h_nmr/data" \
    --src_tokenizer_path "./tokenizer/src_tokenizer/smiles_tokenizer_fast/tokenizer.json" \
    --tgt_tokenizer_path "./tokenizer/tgt_tokenizer/smiles_tokenizer_fast/tokenizer.json" \
    --smiles_tokenizer_path "./tokenizer/smiles_tokenizer_fast/tokenizer.json" \
    --code_test \
    --warm_up_step 800 \
    --lr 2 \