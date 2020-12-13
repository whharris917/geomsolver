import torch, copy
from munch import Munch

class BaseParameter(torch.nn.Module):
    def __init__(self, tensor, parent, name, range, units, locked=False):
        super(BaseParameter, self).__init__()
        self.parent = parent
        self.name = name
        self.range = range
        self.units = units
        self.locked = locked
        self._tensor = None
        self.tensor = tensor
        
    def __call__(self):
        return(self.tensor)
    
    def __repr__(self):
        return('{}({}, range={}, units={}, locked={})'.format(
            self.__class__.__name__, str(self.tensor), self.range, self.units, self.locked))
    
    @property
    def tensor(self):
        return(self._tensor)
    
    @tensor.setter
    def tensor(self, _tensor):
        if type(_tensor) is not list:
            _tensor = [_tensor]
        if self.locked:
            self._tensor = torch.tensor(_tensor, requires_grad=False).to(torch.float) #.view(-1,1)
        else:
            self._tensor = torch.nn.Parameter(torch.tensor(_tensor).to(torch.float)) #.view(-1,1)
        
    @property
    def min(self):
        return(self.range[0])
    
    @property
    def max(self):
        return(self.range[1])
    
    @property
    def full_name(self):
        full_name = '{}.{}.{}'.format(self.parent.type, self.parent.name, self.name)
        return(full_name)
        
    def lock(self):
        self.locked = True
        value = copy.deepcopy(self._tensor.tolist())
        del(self._tensor)
        self.tensor = value
    
    def unlock(self):
        self.locked = False
        value = copy.deepcopy(self._tensor.tolist())
        del(self._tensor)
        self.tensor = value
        
class BaseGeometry(torch.nn.Module):
    def __init__(self, linkage, name):
        super(BaseGeometry, self).__init__()
        self.linkage = linkage
        self.name = name
        self.params = Munch({})
        self._params = Munch({})
    
    def __repr__(self):
        raise Exception('Override this method.')
    
    def E(self):
        raise Exception('Override this method.')
    
    def get_free_params(self):
        free_params = []
        for param in self.params.values():
            if not param.locked:
                free_params.append(param)
        return(free_params)
    
    def set_parameter(self, param_name, value, manual=False, solve=True):
        if manual:
            self._params[param_name].tensor = value
        else:
            self.params[param_name].tensor = value
        self.linkage.config_plot.update()
        if solve:
            try:
                self.linkage.update()
            except:
                pass
        
    def lock(self, param_name=None):
        param_names = self.params.keys() if param_name is None else [param_name]
        for param_name in param_names:
            self.params[param_name].lock()
            self._params[param_name].lock()
            
    def unlock(self, param_name=None):
        param_names = self.params.keys() if param_name is None else [param_name]
        for param_name in param_names:
            self.params[param_name].unlock()
            self._params[param_name].unlock()