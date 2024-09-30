import torch
from torch import nn
from torch.func import grad, vmap, functional_call

class Trainer():
    def __init__(self, model: nn.Module, optim , loss_fn, tracked_indices):
        self.model = model
        self.criterion = loss_fn 
        self.optim = optim
        self.tracked_indices = tracked_indices

    def train(self, train_loader, val_loader, epochs):
        for epoch in range(epochs):
            print(f'Epoch {epoch + 1} :')
            train_loss = self.execute_epoch(train_loader, train = True)
            print(f'Training Loss  is {train_loss}')
            val_loss = self.execute_epoch(val_loader, train = False)
            print(f'Validation Loss  is {val_loss}')
  
    def test(self, test_loader):
        test_loss = self.execute_epoch(test_loader, train = False)
        print(f'Test Loss  is {test_loss}')
        

    def execute_epoch(self, loader,  train: bool):
        if train:
            self.model.train(True)
            running_loss = 0.0
            for i, data in enumerate(loader):
                loss = self.execute_batch(data, train = True)
                running_loss += loss
            avg_loss = running_loss/(i + 1)
            return avg_loss
        else:
            self.model.eval()
            running_loss = 0.0
            for i, data in enumerate(loader):
                loss = self.execute_batch(data, train = False)
                running_loss += loss
            avg_loss = running_loss/(i + 1)
            return avg_loss

    def execute_batch(self, batch, train : bool):
        input, target = batch['inputs'], batch['targets']
        if train:
            self.optim.zero_grad()
            out = self.model(input)
            loss = self.criterion(out, target)
            instance_indices = self.tracked_indices & set(batch['batch_indices'])
            if instance_indices:
                hyper_data = {}
                for idx in instance_indices:
                    rel_idx = batch['batch_indices'].index(idx)
                    hyper_data[idx] = (batch['inputs'][rel_idx], batch['targets'][rel_idx])
                self.get_sample_gradient(hyper_data)
            loss.backward()
            self.optim.step()
            return loss.detach()
        else: 
            with torch.no_grad():
                out = self.model(input)
                loss = self.criterion(out, target)
            return loss

    def get_sample_gradient(self, data):
        def _forward(params, input, target):
            out = functional_call(self.model, params, input)
            loss = self.criterion(out, target)
            return loss
        indices = list(data.keys())
        input = torch.stack([data[idx][0] for idx in indices])
        target = torch.stack([data[idx][1] for idx in indices])
        params = dict(self.model.named_parameters())
        gradient_dict = vmap(grad(_forward), in_dims= (None, 0, 0))(params, input, target)

        sample_gradient_dict = {}
        for i, idx in enumerate(indices):
            sample_gradient_dict[idx] = {}
            for param, grads in gradient_dict.items():
                sample_gradient_dict[idx] |= {param : grads[i]}
        self.sample_gradient_dict = sample_gradient_dict
        self.hydra_sgd_hook()
        self.sample_gradient_dict = None
    
    def hydra_sgd_hook(self, **kwargs):
        pass