import torch
from torch_geometric.loader import DataLoader
from transformers import PreTrainedTokenizerFast

import pandas as pd
import json

from model_gen_mol import NMR2MolGenerator, pad_str_ids


def predict_step(batch, model, device):

    padding_smiles_ids, padding_smiles_masks = pad_str_ids(batch.smiles_ids, batch.smiles_len)

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
        num_return_sequences=10,  # 返回前n个最佳序列
        max_length=120 
    )
    predictions = model.smiles_tokenizer.batch_decode(output_ids)
    print("predictions", len(predictions), predictions)
    tgt = model.smiles_tokenizer.batch_decode(padding_smiles_ids)
    
    predictions = [model._postprocess_smiles(pred_i) for pred_i in predictions]
    grouped_pred = [predictions[i*10:(i+1)*10] for i in range(len(tgt))]
    print("predictions", len(predictions), predictions)
    print("grouped_pred", len(grouped_pred), grouped_pred)

    tgt = [model._postprocess_smiles(tgt_i) for tgt_i in tgt]
    return grouped_pred, tgt

def generate_smiles(checkpoint_path, dataset):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps")
    dataset_info_path = "../Dataset/h_nmr/h_nmr.json"
    with open(dataset_info_path, "r") as f:
        dataset_info = json.load(f)
        MULTIPLETS = dataset_info['MULTIPLETS']
        NUM_H = dataset_info['NUM_H']


    model = NMR2MolGenerator.load_from_checkpoint(checkpoint_path)
    model.eval()
    model.to(device)

    smiles_dataloader = DataLoader(dataset, batch_size=10, )
    
    for batch in smiles_dataloader:
        batch = batch.to(device)
        pred, tgt = predict_step(batch, model, device)
        print("tgt", len(tgt), tgt)
        print("pred", len(pred), pred)
        assert len(pred) == len(tgt)

        pred_data = pd.DataFrame({'pred': pred, 'target': tgt})
        pred_data['rank'] = pred_data.apply(lambda row : row['pred'].index(row['target']) if row['target'] in row['pred'] else 10, axis=1)

        for i in range(1, 11):
            print(f"Top {i}: {(pred_data['rank'] < i).sum() / len(pred_data):.5f}")

        
        break

if __name__ == "__main__":
    checkpoint_path = "/rds/projects/c/chenlv-ai-and-chemistry/wuwj/NMR_MS/sparsespec2graph/exp/exp_hnmr/hnmr_graph2smi_d512_lr2_nomap_graphconvdropout_14-39-15-07-2025/last.ckpt"
    # checkpoint_path = "/rds/projects/c/chenlv-ai-and-chemistry/wuwj/NMR_MS/sparsespec2graph/exp/exp_hnmr/hnmr_graph2smi_d512_lr2_nomap_convdropout_edgesilu16_nodenorm_13-18-16-07-2025/last.ckpt"
    print("Loading dataset...")
    # dataset = torch.load("/rds/projects/c/chenlv-ai-and-chemistry/wuwj/NMR_MS/sparsespec2graph/Dataset/h_nmr/train_val_test_set_nomap_f_s_ids/val_set.pt")
    # print(type(dataset), len(dataset))
    # example = dataset[:100]
    # torch.save(example, "/rds/projects/c/chenlv-ai-and-chemistry/wuwj/NMR_MS/sparsespec2graph/Dataset/example_dataset/hnmr_f_s_ids")
    dataset = torch.load("/rds/projects/c/chenlv-ai-and-chemistry/wuwj/NMR_MS/sparsespec2graph/Dataset/example_dataset/hnmr_f_s_ids.pt")
    print("Generating...")
    generate_smiles(checkpoint_path, dataset)
