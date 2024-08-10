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
        self.is_constrained = False
        self.target = None
        
    def __call__(self):
        if self.parent.linkage.use_manual_params:
            return(self.manual.tensor)
        else:
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
            self._tensor = torch.tensor(_tensor, requires_grad=False).to(torch.float)
        else:
            self._tensor = torch.nn.Parameter(torch.tensor(_tensor).to(torch.float))
        
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
        
    def constraint_E(self):
        if self.is_constrained:
            return((self.tensor-self.target).pow(2))
        else:
            return(0)
           
    def constrain(self, target):
        self.is_constrained = True
        self.target = target
        self.parent.linkage.update()
        
    def unconstrained(self):
        self.is_constrained = False
        self.target = None
        
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
    
    def param_info(self):
        if not self.params.values():
            print('\t\tNo Parameters')
            return()
        print('\t\tLocked Parameters:')
        counter = 0
        for param in self.params.values():
            if param.locked:
                counter += 1
                print('\t\t\t', param.name, '=', param.tensor)
        if counter == 0:
            print('\t\t\tNone')
        print('\t\tFree Parameters:')
        counter = 0
        for param in self.params.values():
            if not param.locked:
                counter += 1
                label = '{}({})'.format(repr(param.tensor)[:9],repr(param.tensor)[22:])
                print('\t\t\t', param.name, '=', label, '########## is_constrained:', param.is_constrained)
        if counter == 0:
            print('\t\t\tNone')
    
    def info(self):
        raise Exception('Override this property.')
    
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
    
    def set_parameter(self, param_name, value):
        if self.linkage.use_manual_params:
            self.params[param_name].manual.tensor = value
        else:
            self.params[param_name].tensor = value
        if not self.linkage.use_manual_params:
            self.linkage.config_plot.update()
        if self.linkage.solve and bool(self.linkage.get_param_dict().values()):
            self.linkage.update()
        
    def lock(self, param_name=None):
        param_names = self.params.keys() if param_name is None else [param_name]
        for param_name in param_names:
            self.params[param_name].lock()
            
    def unlock(self, param_name=None):
        param_names = self.params.keys() if param_name is None else [param_name]
        for param_name in param_names:
            self.params[param_name].unlock()