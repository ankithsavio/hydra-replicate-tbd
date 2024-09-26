import torch
from torch import nn

class Trainer():
    def __init__(self, model: nn.Module, optim , loss_fn):
        self.model = model
        self.criterion = loss_fn 
        self.optim = optim 

    def train(self, train_loader, val_loader, epochs):
        for epoch in range(epochs):
            self.model.train(True)
            count = 0
            for data in train_loader:
                input, target = data['inputs'], data['targets']
                self.optim.zero_grad()
                out = self.model(input)
                loss = self.criterion(out, target)
                loss.backward()
                self.optim.step()
                count+=1
                if count%10 == 0:
                    print(f'{count} batches trained. Loss = {loss.item()}')
            print(f'{epoch} Epoch finished')

            running_loss = 0.0
            self.model.eval()
            with torch.no_grad():
                for i, data in enumerate(val_loader):
                    input, target = data['inputs'], data['targets']
                    out = self.model(input)
                    vloss = self.criterion(out, target)
                    running_loss += vloss
            avg_loss = running_loss/(i + 1)
            print(f'Average Validation Loss for Epoch {epoch} is {avg_loss}')

    def test(self, test_loader):
        running_loss = 0.0
        self.model.eval()
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                input, target = data['inputs'], data['targets']
                out = self.model(input)
                tloss = self.criterion(out, target)
                running_loss += tloss
        avg_loss = running_loss/(i + 1)
        print(f'Average Test Loss is {avg_loss}')