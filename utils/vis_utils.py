# from datasets.mnist import MNIST
import matplotlib.pyplot as plt
import torch

def visualize(data, *args, **kwargs):
    if isinstance(data, list):
        assert len(data) == 1, 'visualize a single image only'
        _,(img, label) = list(data[0].values())
        plt.title(label.item())
        return plt.imshow(img.permute(1, 2, 0), *args)
    
    if isinstance(data, torch.Tensor):
        assert len(data) == 1, 'visualize a single image only'
        # _,(img, label) = list(data[0].values())
        # plt.title(label.item())
        if kwargs:
            plt.title(kwargs['label'])
        return plt.imshow(data.permute(1, 2, 0), *args)