import torch
import os
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset, DataLoader

class IWSLTTranslationDataset(Dataset):
    def __init__(self, split="train", src_lang="en", tgt_lang="de", seq_len=64,
                 tokenizer=None, data_dir=None, src_stoi=None, tgt_stoi=None):
        super().__init__()
        # 设置数据集路径
        if data_dir is None:
            current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.data_dir = os.path.join(current_dir, "en-de")
        else:
            self.data_dir = data_dir

        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_len = seq_len
        self.split = split

        # 加载数据
        self.dataset = self.load_local_dataset(split)

        # 默认分词
        if tokenizer is None:
            self.tokenizer = lambda s: s.lower().split()
        else:
            self.tokenizer = tokenizer

        # 如果外部传入词表（来自训练集），则直接使用，不再重新构建
        if src_stoi is not None and tgt_stoi is not None:
            self.src_stoi = src_stoi
            self.tgt_stoi = tgt_stoi
            self.src_itos = {i: s for s, i in src_stoi.items()}
            self.tgt_itos = {i: s for s, i in tgt_stoi.items()}
        else:
            self.src_stoi, self.src_itos = self.build_vocab(self.dataset, lang=src_lang)
            self.tgt_stoi, self.tgt_itos = self.build_vocab(self.dataset, lang=tgt_lang)

    def load_local_dataset(self, split):
        dataset = []
        if split == "train":
            src_file = os.path.join(self.data_dir, f"train.tags.{self.src_lang}-{self.tgt_lang}.{self.src_lang}")
            tgt_file = os.path.join(self.data_dir, f"train.tags.{self.src_lang}-{self.tgt_lang}.{self.tgt_lang}")
            
            with open(src_file, 'r', encoding='utf-8') as sf, open(tgt_file, 'r', encoding='utf-8') as tf:
                for src_line, tgt_line in zip(sf, tf):
                    src_line = src_line.strip()
                    tgt_line = tgt_line.strip()
                    
                    # 跳过XML标签行
                    if src_line.startswith('<') and not src_line.startswith('<seg'):
                        continue
                    
                    # 添加到数据集
                    dataset.append({
                        self.src_lang: src_line,
                        self.tgt_lang: tgt_line
                    })
        else:
            # 处理验证集和测试集（XML格式）
            if split == "validation":
                file_prefix = "IWSLT17.TED.dev2010"
            else:  # test
                file_prefix = "IWSLT17.TED.tst2010"  # 使用2010年的测试集
                
            src_file = os.path.join(self.data_dir, f"{file_prefix}.{self.src_lang}-{self.tgt_lang}.{self.src_lang}.xml")
            tgt_file = os.path.join(self.data_dir, f"{file_prefix}.{self.src_lang}-{self.tgt_lang}.{self.tgt_lang}.xml")
            
            # 解析XML文件
            src_segments = self._parse_xml_file(src_file)
            tgt_segments = self._parse_xml_file(tgt_file)
            
            for src_text, tgt_text in zip(src_segments, tgt_segments):
                dataset.append({
                    self.src_lang: src_text,
                    self.tgt_lang: tgt_text
                })
                
        return dataset[:20000]  # 限制样本数以加快训练
    
    def _parse_xml_file(self, file_path):
        segments = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if '<seg id=' in line:
                    # 提取<seg>标签中的文本
                    text = line.split('>', 1)[1].split('<')[0].strip()
                    segments.append(text)
        return segments
        
    def build_vocab(self, dataset, lang):
        tokens = set()
        for sample in dataset[:20000]:  # 限制2万条，节省时间
            text = sample[lang]
            tokens.update(self.tokenizer(text))
        stoi = {t: i+4 for i,t in enumerate(sorted(tokens))}
        stoi["<pad>"]=0; stoi["<bos>"]=1; stoi["<eos>"]=2; stoi["<unk>"]=3
        itos = {i:s for s,i in stoi.items()}
        return stoi, itos

    def encode(self, text, vocab):
        toks = self.tokenizer(text)
        ids = [vocab.get(tok, vocab["<unk>"]) for tok in toks]
        ids = [vocab["<bos>"]] + ids[:self.seq_len-2] + [vocab["<eos>"]]
        ids += [vocab["<pad>"]] * max(0, self.seq_len - len(ids))
        return torch.tensor(ids, dtype=torch.long)

    def __len__(self):
        return len(self.dataset)  # 已在load_local_dataset中限制了样本数

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        src = self.encode(sample[self.src_lang], self.src_stoi)
        tgt = self.encode(sample[self.tgt_lang], self.tgt_stoi)
        return src, tgt
