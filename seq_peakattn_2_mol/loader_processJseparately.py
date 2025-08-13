import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedTokenizerFast
import os

# 设置tokenizers并行化环境变量
os.environ["TOKENIZERS_PARALLELISM"] = "false"

H_NMR_INFO = {"max_num_J": 4}

def get_max_num_J(src_data):
    with open(src_data, 'r') as f:
        src_data = f.readlines()
    
    src_data = [line.strip() for line in src_data]
    
    max_num_J = 0
    for src in src_data:
        NMR_seq = src.split("1HNMR")[1]
        peaks_list = NMR_seq.split("|")
        for peak in peaks_list:
            if "J" not in peak: continue
            J_list = peak.split("J")[1].strip().split(" ")
            print(J_list)
            if len(J_list) > max_num_J:
                max_num_J = len(J_list)
            
    return max_num_J

class NMR2MolDataset(torch.utils.data.Dataset):
    def __init__(self, src_data, tgt_data, src_tokenizer, tgt_tokenizer):
        
        with open(src_data, 'r') as f:
            self.src_data = f.readlines()
            self.src_data = [line.strip() for line in self.src_data]
        
        with open(tgt_data, 'r') as f:
            self.tgt_data = f.readlines()
            self.tgt_data = [line.strip() for line in self.tgt_data]
        
        self.tgt_tokenizer = tgt_tokenizer
        self.src_tokenizer = src_tokenizer

        
        
        
        # 检查特殊token是否存在，如果不存在则抛出有意义的错误
        if "1HNMR" not in self.src_tokenizer.vocab:
            raise ValueError("Token '1HNMR' not found in source tokenizer vocabulary")
        if "|" not in self.src_tokenizer.vocab:
            raise ValueError("Token '|' not found in source tokenizer vocabulary")
        self.nmr_split_token = self.src_tokenizer.vocab["1HNMR"]
        self.peak_split_token = self.src_tokenizer.vocab["|"]
        self.J_split_token = self.src_tokenizer.vocab["J"]
        self.max_num_J = H_NMR_INFO["max_num_J"]
        print("max_num_J: ", self.max_num_J, "J_split_token: ", self.J_split_token)
        
    def __len__(self):
        return len(self.tgt_data)
    
    def __getitem__(self, idx):
        tgt_ids = torch.tensor(self.tgt_tokenizer.encode(self.tgt_data[idx]))

        src_ids = torch.tensor(self.src_tokenizer.encode(self.src_data[idx]))
        # process src by splitting peaks
        nmr_split_token_idx = torch.where(src_ids == self.nmr_split_token)[0]
        formula_ids = src_ids[1:nmr_split_token_idx] # remove <bos>

        peaks_ids = src_ids[nmr_split_token_idx+1:]
        peak_split_token_idx = torch.where(peaks_ids == self.peak_split_token)[0]
        peaks_list = []
        for i, idx in enumerate(peak_split_token_idx):
            if i == 0:
                peak = peaks_ids[:idx]
            else:
                peak = peaks_ids[peak_split_token_idx[i-1]+1:idx] # already remove <eos>
            
            if self.J_split_token in peak:
                # 计算J的个数
                num_J = len(peak[torch.where(peak == self.J_split_token)[0]+1:])
                # print("num_J: ", num_J, peak[torch.where(peak == self.J_split_token)[0]+1:])
                # 如果J的个数少于max_num_J，用0填充到max_num_J
                if num_J < self.max_num_J:
                    padding_length = self.max_num_J - num_J
                    peak = F.pad(peak, (0, padding_length), value=0)
            else:
                # 如果没有J，直接填充到max_num_J
                peak = F.pad(peak, (0, self.max_num_J), value=0)
            peaks_list.append(peak)
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
    src_data = "/Users/wuwj/Desktop/NMR-IR/multi-spectra/NMR-Graph/example_data/h_nmr/data/src-val.txt"
    # max_num_J = get_max_num_J(src_data)
    # print(max_num_J)
    tgt_data = "/Users/wuwj/Desktop/NMR-IR/multi-spectra/NMR-Graph/example_data/h_nmr/data/tgt-val.txt"
    src_tokenizer = PreTrainedTokenizerFast(tokenizer_file="../seq2mol_code/tokenizer/src_tokenizer/nmr_formula_tokenizer_fast/tokenizer.json")
    tgt_tokenizer = PreTrainedTokenizerFast(tokenizer_file="../seq2mol_code/tokenizer/tgt_tokenizer/smiles_tokenizer_fast/tokenizer.json")
    dataset = NMR2MolDataset(src_data, tgt_data, src_tokenizer, tgt_tokenizer)
    print(dataset[0])
    print('--------------------------------')
    dataloader = create_nmr2mol_dataloader(dataset, batch_size=2, shuffle=False)
    for batch in dataloader:
        print(batch)
        break