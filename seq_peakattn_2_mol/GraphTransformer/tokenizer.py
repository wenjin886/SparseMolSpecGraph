from tokenizers import Tokenizer, models, pre_tokenizers, processors
from transformers import PreTrainedTokenizerFast
import os



def build_vocab_with_special_tokens(vocab_path):
    # 指定特殊 token 顺序
    special_tokens = [ "[UNK]", "[BOS]", "[EOS]", "[PAD]"]
    token2id = {tok: i for i, tok in enumerate(special_tokens)}
    next_id = len(special_tokens)
    with open(vocab_path, 'r', encoding='utf-8') as f:
        for line in f:
            token = line.strip().split()[0]
            if token not in token2id:
                token2id[token] = next_id
                next_id += 1
    return token2id

def build_tokenizer_from_vocab(vocab_path, save_dir):
    
    token2id = build_vocab_with_special_tokens(vocab_path)
    tokenizer = Tokenizer(models.WordLevel(token2id, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()

    # 你可以根据需要添加 special tokens
    bos_token_id = token2id.get("[BOS]", None)
    eos_token_id = token2id.get("[EOS]", None)
    if bos_token_id is not None and eos_token_id is not None:
        tokenizer.post_processor = processors.TemplateProcessing(
            single="[BOS]:0 $A:0 [EOS]:0",
            special_tokens=[("[BOS]", bos_token_id), ("[EOS]", eos_token_id)]
        )

    tokenizer.save(f"{save_dir}/smiles_tokenizer.json")

    wrapped_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        pad_token="[PAD]",
        bos_token="[BOS]",
        eos_token="[EOS]"
    )
    wrapped_tokenizer.save_pretrained(f"{save_dir}/smiles_tokenizer_fast")
    return wrapped_tokenizer

def load_data(txt_data_path):
    with open(txt_data_path, 'r') as f:
        data = f.readlines()
    return data

def get_src_vocab(src_data_path, save_dir, mode="formula"):
    """
    mode: formula, nmr
    """
    if os.path.isdir(src_data_path):
        
        files = os.listdir(src_data_path)
        src_files = []
        for file in files:
            if "src" in file:
                print(file)
                src_files.append(os.path.join(src_data_path, file))
        src_data = []
        for file in src_files:
            print(len(src_data))
            src_data.extend(load_data(file))
        
    else:
        src_data = load_data(src_data_path)

    if mode == "formula":
        src_data = [line.strip().split("1HNMR")[0].strip().split() for line in src_data]
        save_name = "formula_vocab.txt"
    elif mode == "nmr":
        src_data = [line.strip().split("1HNMR")[1].strip().split() for line in src_data]
        save_name = "nmr_vocab.txt"
    else:
        raise ValueError(f"Invalid mode: {mode}")
    
    vocab = {}
    for data in src_data:
        for item in data:
            if item not in vocab:
                vocab[item] = 1
            else:
                vocab[item] += 1
    vocab_keys = list(vocab.keys()) 
    vocab_keys = sorted(vocab_keys, key=lambda x: vocab[x], reverse=True)

    if "vocab" not in save_dir:
        save_dir = os.path.join(save_dir, "vocab")
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, save_name), 'w') as f:
        for item in vocab_keys:
            f.write(f"{item} {vocab[item]}\n")

if __name__ == "__main__":
    src_data_path = "/Users/wuwj/Desktop/NMR-IR/multi-spectra/NMR-Graph/example_data/h_nmr/data/"
    save_dir = os.path.dirname(src_data_path)
    get_src_vocab(src_data_path, save_dir, mode="formula")
    get_src_vocab(src_data_path, save_dir, mode="nmr")
    # vocab_path = "/rds/projects/c/chenlv-ai-and-chemistry/wuwj/NMR_MS/sparsespec2graph/code/seq2mol_code/seqGraph2mol_code/tokenizer/vocab.src"
    # save_dir = os.path.join(os.path.dirname(vocab_path),"src_tokenizer")
    # print(save_dir)
    # # save_dir = "./tokenizer/tgt_tokenizer"

    # # vocab_path = "/Users/wuwj/Desktop/NMR-IR/multi-spectra/NMR-Graph/example_data/h_nmr/vocab/vocab.src"
    # # save_dir = "./tokenizer/src_tokenizer"

    # os.makedirs(save_dir, exist_ok=True)
    # build_tokenizer_from_vocab(vocab_path, save_dir)