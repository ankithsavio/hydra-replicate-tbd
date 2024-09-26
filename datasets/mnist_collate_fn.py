import torch

def collate_batch(batch):
        inputs = []
        targets = []
        sample_indices = []

        for data in batch:
             index = data[0]['index']
             input, target = data[0]['data']
             sample_indices.append(index)
             inputs.append(input)
             targets.append(target)

        batch_size = len(batch)
        res = {
             "batch_size": batch_size,
             "inputs": torch.stack(inputs),
             "targets": torch.stack(targets),
             "sample_indices": sample_indices # not sure why but we will figure it out later
        }
        return res