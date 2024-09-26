# from datasets.mnist import MNIST
import matplotlib.pyplot as plt

def visualize(data, *args):
    assert len(data) == 1, 'visualize a single image only'
    _,(img, label) = list(data[0].values())
    plt.title(label.item())
    return plt.imshow(img.permute(1, 2, 0), *args)