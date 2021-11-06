class TextStreamBpttIterator:
    def __init__(self, data, batch_size, bptt, device='cpu', ext_len=None):
        """
            data -- LongTensor -- the LongTensor is strictly ordered
        """
        self.batch_size = batch_size
        self.bptt = bptt
        self.ext_len = ext_len if ext_len is not None else 0

        self.device = device

        # Work out how cleanly we can divide the dataset into batch_size parts.
        self.n_step = data.size(0) // batch_size

        # Trim off any extra elements that wouldn't cleanly fit (remainders).
        data = data.narrow(0, 0, self.n_step * batch_size)

        # Evenly divide the data across the batch_size batches.
        self.data = data.view(batch_size, -1).contiguous().to(device)

        # Number of mini-batches
        self.n_batch = (self.n_step + self.bptt - 1) // self.bptt

    def get_batch(self, i, bptt=None):
        if bptt is None:
            bptt = self.bptt

        seq_len = min(bptt, self.data.size(1) - 1 - i)

        end_idx = i + seq_len
        beg_idx = max(0, i - self.ext_len)

        data = self.data[:, beg_idx:end_idx].contiguous()
        target = self.data[:, i+1:i+1+seq_len].contiguous()

        return data, target, seq_len

    def __len__(self):
        return self.data.size(1)

    def get_fixlen_iter(self, start=0):
        for i in range(start, self.data.size(1) - 1, self.bptt):
            yield self.get_batch(i)

    def __iter__(self):
        return self.get_fixlen_iter()