import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedTokenizerFast

class NMR2MolDataset(torch.utils.data.Dataset):
    def __init__(self, src_data, tgt_data, formula_tokenizer, nmr_tokenizer, tgt_tokenizer):
        
        with open(src_data, 'r') as f:
            self.src_data = f.readlines()
        self.formula_data, self.nmr_data = [], []
        for line in self.src_data:
            line = line.split("1HNMR")
            self.formula_data.append(line[0].strip())
            self.nmr_data.append(line[1].strip())
            
        
        with open(tgt_data, 'r') as f:
            self.tgt_data = f.readlines()
        self.tgt_data = [line.strip() for line in self.tgt_data]
        
        self.tgt_tokenizer = tgt_tokenizer
        self.formula_tokenizer = formula_tokenizer
        self.nmr_tokenizer = nmr_tokenizer
        
        if "|" not in self.nmr_tokenizer.vocab:
            raise ValueError("Token '|' not found in nmr_tokenizer")
        self.peak_split_token = self.nmr_tokenizer.vocab["|"]

    def __len__(self):
        return len(self.tgt_data)
    
    def __getitem__(self, idx):
        tgt_ids = torch.tensor(self.tgt_tokenizer.encode(self.tgt_data[idx]))

        formula_ids = torch.tensor(self.formula_tokenizer.encode(self.formula_data[idx]))

        peaks_ids = torch.tensor(self.nmr_tokenizer.encode(self.nmr_data[idx]))
        peak_split_token_idx = torch.where(peaks_ids == self.peak_split_token)[0]
        peaks_list = []
        for i, idx in enumerate(peak_split_token_idx):
            if i == 0:
                peaks_list.append(peaks_ids[:idx])
            else:
                peaks_list.append(peaks_ids[peak_split_token_idx[i-1]+1:idx]) 
        peaks_ids = pad_sequence(peaks_list, batch_first=True, padding_value=0)
        
        return {"tgt_ids": tgt_ids, "formula_ids": formula_ids,  "peaks_ids": peaks_ids, "peak_num":peaks_ids.shape[0]}

def collate_fn(batch):
    
    tgt_ids = [item["tgt_ids"] for item in batch]
    tgt_ids = pad_sequence(tgt_ids, batch_first=True, padding_value=0)
    tgt_mask = (tgt_ids != 0).float()
    
    formula_ids = [item["formula_ids"] for item in batch]
    formula_ids = pad_sequence(formula_ids, batch_first=True, padding_value=0)
    formula_mask = (formula_ids != 0).float()


    peaks_ids = [item["peaks_ids"] for item in batch]
    max_feat_dim = max([p.shape[-1] for p in peaks_ids])
    # 对每个 tensor 补齐到相同特征维
    peaks_ids = [F.pad(p, (0, max_feat_dim - p.shape[1])) for p in peaks_ids]
    

    peaks_ids = pad_sequence(peaks_ids, batch_first=True, padding_value=0)
    peaks_mask = (peaks_ids != 0).float()
    
    return {
        "tgt_ids": tgt_ids,
        "tgt_mask": tgt_mask,

        "formula_ids": formula_ids,
        "formula_mask": formula_mask,

        "peaks_ids": peaks_ids,
        "peaks_mask": peaks_mask,
        
    }

def create_nmr2mol_dataloader(dataset, batch_size, shuffle=True, num_workers=4):
    
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn
    )



if __name__ == "__main__":
    src_data = "/rds/projects/c/chenlv-ai-and-chemistry/wuwj/NMR_MS/sparsespec2graph/multimodal-spectroscopic-dataset/runs/runs_new_onmt_w_formula/h_nmr/data/src-val.txt"
    tgt_data = "/rds/projects/c/chenlv-ai-and-chemistry/wuwj/NMR_MS/sparsespec2graph/multimodal-spectroscopic-dataset/runs/runs_new_onmt_w_formula/h_nmr/data/tgt-val.txt"
    nmr_tokenizer = PreTrainedTokenizerFast(tokenizer_file="./tokenizers/nmr_tokenizer_fast/tokenizer.json")
    formula_tokenizer = PreTrainedTokenizerFast(tokenizer_file="./tokenizers/formula_tokenizer_fast/tokenizer.json")
    tgt_tokenizer = PreTrainedTokenizerFast(tokenizer_file="./tokenizers/smiles_tokenizer_fast/tokenizer.json")
    dataset = NMR2MolDataset(src_data, tgt_data, formula_tokenizer, nmr_tokenizer, tgt_tokenizer)
    print(dataset[0])
    print('--------------------------------')
    dataloader = create_nmr2mol_dataloader(dataset, batch_size=2, shuffle=False)
    for batch in dataloader:
        print(batch)
        break