import torch
from torchvision import datasets
from typing import Dict

class MNIST(datasets.MNIST):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
    
    def __getitem__(self, _index):

        if isinstance(_index, slice):
            _img, _target = self.data[_index], self.targets[_index]
            _index = range(*_index.indices(len(self.data)))

            if self.transform is not None:
                _img = self.transform(_img)
            
            return [{'index': index, 'data' : (img.to(torch.float32)[None], int(target))} for index, img, target in zip(_index, _img, _target)]
    
        elif isinstance(_index, int):
            img, target = self.data[_index], self.targets[_index]
            
            if self.transform is not None:
                img = self.transform(img[None]).squeeze()

            return [{'index': _index, 'data' : (img.to(torch.float32)[None], target)}]
        
        else: raise ValueError("Invalid argumnet")
    