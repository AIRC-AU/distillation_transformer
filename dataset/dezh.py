import os
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from torchtext.data import get_tokenizer
import jieba
from torchtext.vocab import build_vocab_from_iterator
import zhconv
#构建数据集

class TranslationDataset(Dataset):
    def __init__(self, filepath, use_cache=True):
        self.row_count = self.get_row_count(filepath)
        self.tokenizer = get_tokenizer('basic_english')  # 使用basic_english分词器处理德语
        self.use_cache = use_cache

        # 加载词典和token
        self.de_vocab = self.get_de_vocab(filepath)
        self.zh_vocab = self.get_zh_vocab(filepath)
        self.de_tokens = self.load_tokens(filepath, self.de_tokenizer, self.de_vocab, "构建德语tokens", 'de')
        self.zh_tokens = self.load_tokens(filepath, self.zh_tokenizer, self.zh_vocab, "构建中文tokens", 'zh')

    def __getitem__(self, index):
        return self.de_tokens[index], self.zh_tokens[index]

    def __len__(self):
        return self.row_count

    def get_row_count(self, filepath):
        count = 0
        for _ in open(filepath, encoding='utf-8'):
            count += 1
        return count

    def de_tokenizer(self, line):
        # 德语分词，保持大小写
        return self.tokenizer(line)

    def zh_tokenizer(self, line):
        return list(jieba.cut(line))

    def get_de_vocab(self, filepath):
        def yield_de_tokens():
            with open(filepath, encoding='utf-8') as f:
                print("---开始构建德语词典---")
                for line in tqdm(f, desc="构建德语词典", total=self.row_count):
                    sentence = line.split('\t')
                    if len(sentence) >= 1:
                        german = sentence[0]
                        yield self.de_tokenizer(german)

        dir_path = os.path.dirname(filepath)
        de_vocab_file = os.path.join(dir_path, "vocab_de.pt")
        if self.use_cache and os.path.exists(de_vocab_file):
            de_vocab = torch.load(de_vocab_file, map_location="cpu")
        else:
            de_vocab = build_vocab_from_iterator(
                yield_de_tokens(),
                min_freq=2,
                specials=["<s>", "</s>", "<pad>", "<unk>"]
            )
            de_vocab.set_default_index(de_vocab["<unk>"])
            if self.use_cache:
                torch.save(de_vocab, de_vocab_file)
        return de_vocab

    def get_zh_vocab(self, filepath):
        def yield_zh_tokens():
            with open(filepath, encoding='utf-8') as f:
                print("---开始构建中文词典---")
                for line in tqdm(f, desc="构建中文词典", total=self.row_count):
                    sentence = line.split('\t')
                    if len(sentence) >= 2:
                        chinese = zhconv.convert(sentence[1], 'zh-cn')
                        yield self.zh_tokenizer(chinese)

        dir_path = os.path.dirname(filepath)
        zh_vocab_file = os.path.join(dir_path, "vocab_zh.pt")
        if self.use_cache and os.path.exists(zh_vocab_file):
            zh_vocab = torch.load(zh_vocab_file, map_location="cpu")
        else:
            zh_vocab = build_vocab_from_iterator(
                yield_zh_tokens(),
                min_freq=1,
                specials=["<s>", "</s>", "<pad>", "<unk>"]
            )
            zh_vocab.set_default_index(zh_vocab["<unk>"])
            if self.use_cache:
                torch.save(zh_vocab, zh_vocab_file)
        return zh_vocab

    def load_tokens(self, filepath, tokenizer, vocab, desc, lang):
        dir_path = os.path.dirname(filepath)
        cache_file = os.path.join(dir_path, f"tokens_list_{lang}.pt")
        if self.use_cache and os.path.exists(cache_file):
            print(f"正在加载缓存文件[{cache_file}]，请稍候...")
            return torch.load(cache_file, map_location="cpu")

        tokens_list = []
        with open(filepath, encoding='utf-8') as f:
            for line in tqdm(f, desc=desc, total=self.row_count):
                sentence = line.strip().split('\t')
                if (lang == 'de' and len(sentence) >= 1) or (lang != 'de' and len(sentence) >= 2):
                    if lang == 'de':
                        text = sentence[0]  # 德语保持原始大小写
                    else:
                        text = zhconv.convert(sentence[1], 'zh-cn')
                    tokens = tokenizer(text)
                    token_indices = [vocab[token] for token in tokens]
                    token_tensor = torch.tensor([vocab["<s>"]] + token_indices + [vocab["</s>"]])
                    tokens_list.append(token_tensor)

        if self.use_cache:
            torch.save(tokens_list, cache_file)
        return tokens_list


if __name__ == '__main__':
    dataset = TranslationDataset("data/de-zh/de-zh.txt")
    print("句子数量为:", dataset.row_count)
    print(dataset.de_tokenizer("Ich bin ein deutscher Tokenizer."))
    print("德语词典大小:", len(dataset.de_vocab))
    print("中文词典大小:", len(dataset.zh_vocab))
    print(dict((i, dataset.de_vocab.lookup_token(i)) for i in range(10)))
    print(dict((i, dataset.zh_vocab.lookup_token(i)) for i in range(10))) 