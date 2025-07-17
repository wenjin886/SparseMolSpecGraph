import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedTokenizerFast

class NMR2MolDataset(torch.utils.data.Dataset):
    def __init__(self, src_data, tgt_data, src_tokenizer, tgt_tokenizer):
        
        with open(src_data, 'r') as f:
            self.src_data = f.readlines()
            self.src_data = [line.strip() for line in self.src_data]
        
        with open(tgt_data, 'r') as f:
            self.tgt_data = f.readlines()
            self.tgt_data = [line.strip() for line in self.tgt_data]
        
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer

    def __len__(self):
        return len(self.tgt_data)
    
    def __getitem__(self, idx):
        src_ids = self.src_tokenizer.encode(self.src_data[idx])
        tgt_ids = self.tgt_tokenizer.encode(self.tgt_data[idx])
        return {"src_ids": src_ids, "tgt_ids": tgt_ids}

class NMR2MolDataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size, shuffle=True):
        dataset = NMR2MolDataset(src_data, tgt_data, src_tokenizer, tgt_tokenizer)
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=self.collate_fn)
    
    def collate_fn(self, batch):
        src_ids = [torch.tensor(item["src_ids"],dtype=torch.long) for item in batch]
        tgt_ids = [torch.tensor(item["tgt_ids"],dtype=torch.long) for item in batch]
        print(src_ids)
        print(tgt_ids)
        src_ids = pad_sequence(src_ids, batch_first=True, padding_value=0)
        src_mask = (src_ids != 0).float()
        tgt_ids = pad_sequence(tgt_ids, batch_first=True, padding_value=0)
        tgt_mask = (tgt_ids != 0).float()
        return {"src_ids": src_ids, "tgt_ids": tgt_ids, "src_mask": src_mask, "tgt_mask": tgt_mask}

if __name__ == "__main__":
    src_data = "/Users/wuwj/Desktop/NMR-IR/multi-spectra/NMR-Graph/example_data/h_nmr/data/src-val.txt"
    tgt_data = "/Users/wuwj/Desktop/NMR-IR/multi-spectra/NMR-Graph/example_data/h_nmr/data/tgt-val.txt"
    src_tokenizer = PreTrainedTokenizerFast(tokenizer_file="./tokenizer/src_tokenizer/smiles_tokenizer_fast/tokenizer.json")
    tgt_tokenizer = PreTrainedTokenizerFast(tokenizer_file="./tokenizer/tgt_tokenizer/smiles_tokenizer_fast/tokenizer.json")
    dataset = NMR2MolDataset(src_data, tgt_data, src_tokenizer, tgt_tokenizer)
    print(dataset[0])
    print('--------------------------------')
    dataloader = NMR2MolDataLoader(dataset, batch_size=2, shuffle=False)
    for batch in dataloader:
        print(batch)
        break