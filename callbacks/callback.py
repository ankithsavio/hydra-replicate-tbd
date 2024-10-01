from torch.utils.tensorboard import SummaryWriter
from utils.math_utils import addition, multiplication

class Callbacks: 
    def before_execute(self, **kwargs):
        pass
    def after_execute(self, **kwargs):
        pass
    def before_epoch(self, **kwargs):
        pass
    def after_epoch(self, **kwargs):
        pass
    def before_batch(self, **kwargs):
        pass
    def after_batch(self, **kwargs):
        pass

class Logger(Callbacks):
    def __init__(self, root_dir):
        self.logs = SummaryWriter(log_dir = root_dir)

    def after_batch(self, **kwargs):
        if kwargs['train']:
            self.logs.add_scalar('Loss/Train', kwargs['loss'], kwargs['epoch'])
        else: 
            self.logs.add_scalar('Loss/Val', kwargs['loss'], kwargs['epoch'])
    
    def after_execute(self, **kwargs):
        self.logs.close()

class HyDRASGD(Callbacks):
    def __init__(self):
        self._lr = None
        self._momentum = None
        self._weight_decay = None
        self.train_set_size = None
        self.hyper_gradient_dict = {}
        self.momentum_gradient_dict = {}
    
    def before_execute(self, **kwargs):
        self.train_set_size = kwargs['train_set_size']
    
    def before_batch(self, **kwargs):
        self._lr = kwargs['lr']
        self._momentum = kwargs['momentum']
        self._weight_decay = kwargs['weight_decay']
    
    def after_batch(self, **kwargs):
        batch_size = kwargs['batch_size']
        batch_indices = kwargs['batch_indices']
        sample_gradient_dict = kwargs['sample_gradient_dict']
        for idx in batch_indices:
            instance_gradient = sample_gradient_dict.get(idx, None)
            if instance_gradient is not None:
                instance_gradient = (
                    instance_gradient 
                    * self.train_set_size
                    / batch_size
                )
            self._update_hyper_gradient(idx, instance_gradient)

    def _update_hyper_gradient(self, index, instance_gradient):
        hyper_gradient, momentum_gradient = self._get_hyper_gradient_tensors(index)
        gg = addition(multiplication(self._weight_decay, hyper_gradient), instance_gradient)
        momentum_gradient = addition(multiplication(momentum_gradient, self._momentum), gg)
        hyper_gradient = addition(hyper_gradient, multiplication(momentum_gradient, -self._lr))
        self._set_hyper_gradient_tensors(index, hyper_gradient, momentum_gradient)

    def _get_hyper_gradient_tensors(self, index):
        hyper_gradient = self.hyper_gradient_dict.get(index, None)
        momentum_gradient = self.momentum_gradient_dict.get(index, None)
        return (hyper_gradient, momentum_gradient)
    
    def _set_hyper_gradient_tensors(self, index, hyper_gradient, momentum_gradient):
        self.hyper_gradient_dict[index] = hyper_gradient
        self.momentum_gradient_dict[index] = momentum_gradient
