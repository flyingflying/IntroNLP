# Author: lqxu

import torch
from torch import Tensor


if __name__ == '__main__':
    queue_size = 5
    batch_size = 3
    hidden_size = 2
    queue_ptr = 0
    queue = torch.zeros(queue_size, hidden_size)

    def update_queue(keys: Tensor):
        global queue_ptr
        start_idx = queue_ptr
        end_idx = start_idx + batch_size
        if end_idx <= queue_size:
            queue[start_idx:end_idx, :] = keys
        else:
            extra_size = end_idx - queue_size
            queue[start_idx:, :] = keys[:batch_size-extra_size]
            queue[:extra_size, :] = keys[batch_size-extra_size:]
        queue_ptr = end_idx % queue_size

    for idx in range(1, 1000):
        update_queue(torch.full(size=(batch_size, hidden_size), fill_value=idx))
        print(queue)
