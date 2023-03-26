import os
import requests
import tiktoken
import numpy as np
import torch


class Datasets:
    def __init__(self, sub_dir, save_path="data", split="train", block_size=1024, batch_size=12, device="cpu"):
        self.sub_dir, self.save_path, self.split = sub_dir, save_path, split
        data_file = "train.bin" if split == "train" else "val.bin"
        data_path = os.path.join(save_path, sub_dir, data_file)
        if not os.path.exists(data_path):
            self.download()
        print(">>>> Load data from {}".format(data_path))
        self.data = np.memmap(data_path, dtype=np.uint16, mode="r")

        self.block_size, self.batch_size, self.device = block_size, batch_size, device
        self.device_type = "cuda" if "cuda" in device else "cpu"  # for later use in torch.autocast

    def download(self):
        raise NotImplemented

    def get_random_batch(self):
        idx = torch.randint(len(self.data) - self.block_size, (self.batch_size,))
        xx = torch.stack([torch.from_numpy((self.data[i : i + self.block_size]).astype(np.int64)) for i in idx])
        yy = torch.stack([torch.from_numpy((self.data[i + 1 : i + 1 + self.block_size]).astype(np.int64)) for i in idx])
        if self.device_type == "cuda":
            xx, yy = xx.pin_memory().to(self.device, non_blocking=True), yy.pin_memory().to(self.device, non_blocking=True)
        else:
            xx, yy = xx.to(self.device), yy.to(self.device)
        return xx, yy


class TinyShakespeare(Datasets):
    def __init__(self, sub_dir="shakespeare", save_path="data", split="train", block_size=1024, batch_size=12, device="cpu"):
        super().__init__(sub_dir, save_path, split, block_size, batch_size, device=device)

    def download(self, val_split=0.1):
        data_dir = os.path.join(self.save_path, self.sub_dir)
        train_bin_file, val_bin_file = os.path.join(data_dir, "train.bin"), os.path.join(data_dir, "val.bin")
        if os.path.exists(train_bin_file) and os.path.exists(val_bin_file):
            return train_bin_file, val_bin_file

        if not os.path.exists(data_dir):
            os.makedirs(data_dir, exist_ok=True)

        input_file_path = os.path.join(data_dir, "input.txt")
        if not os.path.exists(input_file_path):
            data_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
            print(">>>> Downloading {} from {}".format(input_file_path, data_url))
            with open(input_file_path, "w") as ff:
                ff.write(requests.get(data_url).text)

        with open(input_file_path, "r") as ff:
            data = ff.read()
        total = len(data)
        train_split = 1 - val_split
        train_data = data[: int(total * train_split)]
        val_data = data[int(total * train_split) :]

        # encode with tiktoken gpt2 bpe
        enc = tiktoken.get_encoding("gpt2")
        train_ids = enc.encode_ordinary(train_data)
        val_ids = enc.encode_ordinary(val_data)
        print(f"train has {len(train_ids):,} tokens")
        print(f"val has {len(val_ids):,} tokens")

        # export to bin files
        train_ids = np.array(train_ids, dtype=np.uint16)
        val_ids = np.array(val_ids, dtype=np.uint16)
        train_ids.tofile(train_bin_file)
        val_ids.tofile(val_bin_file)
        return train_bin_file, val_bin_file
