import torch


class StreamDataset(torch.utils.data.Dataset):
    def __init__(self, data, tgt_len, batch_size):
        self.total_len = len(data)
        self.tgt_len = tgt_len
        self.batch_size = batch_size

        self.data, self.tgt = self.format_bptt(data)

    def format_bptt(self, data):
        n_total = (self.total_len // self.batch_size // self.tgt_len) * self.tgt_len * self.batch_size
        data_bptt = data[:n_total].reshape(self.batch_size, -1, self.tgt_len)
        tgt_bptt = data[1:n_total+1].reshape(self.batch_size, -1, self.tgt_len)
        return data_bptt, tgt_bptt

    def __len__(self):
        return self.data.shape[0] * self.data.shape[1]

    def __getitem__(self, i):
        col = i // self.batch_size
        row = i - (col * self.batch_size)
        return self.data[row, col], self.tgt[row, col]


class StreamCollator:
    def __call__(self, features):
        data = torch.stack([x[0] for x in features])
        tgt = torch.stack([x[1] for x in features])
        return data, tgt