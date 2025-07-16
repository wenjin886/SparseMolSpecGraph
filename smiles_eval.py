import torch
from torch_geometric.loader import DataLoader
from transformers import PreTrainedTokenizerFast

import json

from model_gen_mol import NMR2MolGenerator, pad_str_ids


def predict_step(batch, model, device):
    if model.use_formula:
        padding_formula_ids, padding_formula_masks = pad_str_ids(batch.formula_ids, batch.formula_len)
        padding_formula_ids, padding_formula_masks = padding_formula_ids.to(device), padding_formula_masks.to(device)
    else:
        padding_formula_ids, padding_formula_masks = None, None
    src, src_mask = model.encode(batch, padding_formula_ids, padding_formula_masks)

    bos_token_id = model.smiles_tokenizer.convert_tokens_to_ids('[BOS]')
    eos_token_id = model.smiles_tokenizer.convert_tokens_to_ids('[EOS]')

    input_ids = torch.tensor([[bos_token_id]]).repeat(src.shape[0], 1).to(model.device)
   

    output_ids = model.smiles_decoder.generate(
        input_ids=input_ids,
        encoder_hidden_states=src,
        encoder_attention_mask=src_mask,
        num_beams=10,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        pad_token_id=model.smiles_tokenizer.pad_token_id,
        num_return_sequences=10  # 返回前5个最佳序列
    )
    predictions = model.smiles_tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    return model._postprocess_smiles(predictions)

def generate_smiles(checkpoint_path, dataset):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps")
    dataset_info_path = "../Dataset/h_nmr.json"
    with open(dataset_info_path, "r") as f:
        dataset_info = json.load(f)
        MULTIPLETS = dataset_info['MULTIPLETS']
        NUM_H = dataset_info['NUM_H']


    smiles_tokenizer_path = "/Users/wuwj/Desktop/NMR-IR/multi-spectra/NMR-Graph/example_data/smiles_tokenizer_fast/tokenizer.json"
    smiles_tokenizer = PreTrainedTokenizerFast(tokenizer_file=smiles_tokenizer_path,
                                                            bos_token="[BOS]",
                                                            eos_token="[EOS]",
                                                            pad_token="[PAD]",
                                                            unk_token="[UNK]",
                                                            padding_side="right" )
    # len(smiles_tokenizer.get_vocab())

    model = NMR2MolGenerator.load_from_checkpoint(checkpoint_path)
    model.eval()
    model.to(device)

    smiles_dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    for batch in smiles_dataloader:
        pred = predict_step(batch, model)
        print(pred)
        break

if __name__ == "__main__":
    checkpoint_path = "/Users/wuwj/Desktop/NMR-IR/multi-spectra/NMR-Graph/exp/exp_hnmr/hnmr_graph2smi_d512_nomap_formula_noproj_16-31-14-07-2025/last.ckpt"
    dataset = torch.load("../Dataset/h_nmr/train_val_test_set_nomap_f_s_ids/test_set.pt")
    generate_smiles(checkpoint_path, dataset)