from tokenizers import Tokenizer, models, pre_tokenizers, processors, normalizers, Regex
from transformers import PreTrainedTokenizerFast
import os



def build_vocab_with_special_tokens(vocab_path, mode):
    # 指定特殊 token 顺序
    if mode == "nmr":
        special_tokens = [ "[PAD]", "[CLS]", "[SEP]", "[UNK]"]
    else:    
        special_tokens = [ "[PAD]", "[BOS]", "[EOS]", "[UNK]"]
    token2id = {tok: i for i, tok in enumerate(special_tokens)}
    next_id = len(special_tokens)
    with open(vocab_path, 'r', encoding='utf-8') as f:
        for line in f:
            token = line.strip().split()[0]
            if token not in token2id:
                token2id[token] = next_id
                next_id += 1
    return token2id

def build_tokenizer_from_vocab(vocab_path, mode, save_dir):
    
    token2id = build_vocab_with_special_tokens(vocab_path, mode)
    tokenizer = Tokenizer(models.WordLevel(token2id, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()

    

    # tokenizer.save(f"{save_dir}/{mode}_tokenizer.json")

    if mode == "nmr":
        tokenizer.normalizer = normalizers.Sequence([
            normalizers.Replace(Regex(r"\|"), " [SEP] [CLS] ")
        ])

        tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()

        cls_token_id = token2id.get("[CLS]", None)
        sep_token_id = token2id.get("[SEP]", None)
        if cls_token_id is not None and sep_token_id is not None:
            tokenizer.post_processor = processors.TemplateProcessing(
                single="[CLS]:0 $A:0",
                special_tokens=[("[CLS]", cls_token_id), ("[SEP]", sep_token_id)]
            )

        wrapped_tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer,
            unk_token="[UNK]",
            pad_token="[PAD]",
            cls_token="[CLS]",
            sep_token="[SEP]"
        )
    else:
        # 可以根据需要添加 special tokens
        bos_token_id = token2id.get("[BOS]", None)
        eos_token_id = token2id.get("[EOS]", None)
        if bos_token_id is not None and eos_token_id is not None:
            tokenizer.post_processor = processors.TemplateProcessing(
                single="[BOS]:0 $A:0 [EOS]:0",
                special_tokens=[("[BOS]", bos_token_id), ("[EOS]", eos_token_id)]
            )

        wrapped_tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer,
            unk_token="[UNK]",
            pad_token="[PAD]",
            bos_token="[BOS]",
            eos_token="[EOS]"
        )
    wrapped_tokenizer.save_pretrained(f"{save_dir}/{mode}_tokenizer_fast")
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
    # src_data_path = "/rds/projects/c/chenlv-ai-and-chemistry/wuwj/NMR_MS/sparsespec2graph/multimodal-spectroscopic-dataset/runs/runs_new_onmt_w_formula/h_nmr/data"
    # src_data_path = "/rds/projects/c/chenlv-ai-and-chemistry/wuwj/NMR_MS/sparsespec2graph/multimodal-spectroscopic-dataset/runs/example/h_nmr/data"
    # save_dir = os.path.dirname(src_data_path)
    # get_src_vocab(src_data_path, save_dir, mode="formula")
    # save_dir = "./"
    # get_src_vocab(src_data_path, save_dir, mode="nmr")

    
    save_dir = "./tokenizers"
    os.makedirs(save_dir, exist_ok=True)
    src_vocab_dir = "/rds/projects/c/chenlv-ai-and-chemistry/wuwj/NMR_MS/sparsespec2graph/multimodal-spectroscopic-dataset/runs/runs_new_onmt_w_formula/h_nmr/vocab"
    build_tokenizer_from_vocab(os.path.join(src_vocab_dir, "nmr_vocab.txt"), "nmr", save_dir)
    # build_tokenizer_from_vocab(os.path.join(src_vocab_dir, "formula_vocab.txt"), "formula", save_dir)
    
    # tgt_vocab = "/rds/projects/c/chenlv-ai-and-chemistry/wuwj/NMR_MS/sparsespec2graph/multimodal-spectroscopic-dataset/runs/runs_new_onmt_w_formula/h_nmr/data/vocab/vocab.tgt"
    # build_tokenizer_from_vocab(tgt_vocab, "smiles", save_dir)
