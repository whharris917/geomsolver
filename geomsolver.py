import numpy as np
from scipy import optimize
import pandas as pd
import torch, IPython, itertools, string
import random, time, warnings, copy
import matplotlib.pyplot as plt
from matplotlib import animation
from munch import Munch
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets

class BaseParameter(torch.nn.Module):
    def __init__(self, tensor, locked=False):
        super(BaseParameter, self).__init__()
        self.locked = locked
        self._tensor = None
        self.tensor = tensor
        
    def __call__(self):
        return(self.tensor)
    
    def __repr__(self):
        return('{}({}, locked={})'.format(self.__class__.__name__, str(self.tensor), self.locked))
    
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
        
class Parameter(BaseParameter):
    def __init__(self, tensor, locked=False):
        super(Parameter, self).__init__(tensor, locked)
        
class ManualParameter(BaseParameter):
    def __init__(self, tensor, locked=False):
        super(ManualParameter, self).__init__(tensor, locked)
    
class Point(torch.nn.Module):
    def __init__(self, linkage, name):
        super(Point, self).__init__()
        self.linkage = linkage
        self.name = name
        self.params = Munch({})
        self._params = Munch({})
        
    def __repr__(self):
        raise Exception('Override this method.')
    
    @property
    def r(self):
        raise Exception('Override this property.')

    def root(self):
        raise Exception('Override this method.')
    
    def E(self):
        raise Exception('Override this method.')
        
    def add_frompointline(self, L, theta, phi=None, ux=None, uz=None, locked=False):
        new_line = self.linkage.add_frompointline(self, L, theta, phi, ux, uz, locked)
        return(new_line)
    
    def add_onpointline(self, L, theta, phi=None, ux=None, uz=None, beta=None):
        new_line = self.linkage.add_onpointline(self, L, theta, phi, ux, uz, beta)
        return(new_line)
    
    def get_free_params(self):
        free_params = []
        for param in self.params.values():
            if not param.locked:
                free_params.append(param)
        return(free_params)
    
    def set_parameter(self, param_name, value):
        self.params[param_name].tensor = value
        self._params[param_name].tensor = value
        self.linkage.plot.update()
        try:
            self.linkage.update()
        except:
            pass
        
    def lock(self):
        for param_name in self.params.keys():
            self.params[param_name].lock()
            self._params[param_name].lock()
            
    def unlock(self):
        for param_name in self.params.keys():
            self.params[param_name].unlock()
            self._params[param_name].unlock()
    
class AtPoint(Point):
    def __init__(self, linkage, name, at):
        super(AtPoint, self).__init__(linkage, name)
        self.params.x = Parameter(at[0], locked=False)
        self.params.y = Parameter(at[1], locked=False)
        self.params.z = Parameter(at[2], locked=False)
        self._params.x = ManualParameter(at[0], locked=False)
        self._params.y = ManualParameter(at[1], locked=False)
        self._params.z = ManualParameter(at[2], locked=False)
    
    def __repr__(self):
        label = self.__class__.__name__[:-5]
        return('[{}]Point_{}(at={})'.format(label, self.name, str(self.r.tolist())))
    
    @property
    def r(self):
        if self.linkage.use_manual_params:
            return(torch.cat([self._params.x(), self._params.y(), self._params.z()]))
        else:
            return(torch.cat([self.params.x(), self.params.y(), self.params.z()]))
    
    def root(self):
        return(self)
    
    def E(self):
        return(0)
    
class AnchorPoint(Point):
    def __init__(self, linkage, name, at):
        super(AnchorPoint, self).__init__(linkage, name)
        self.params.x = Parameter(at[0], locked=True)
        self.params.y = Parameter(at[1], locked=True)
        self.params.z = Parameter(at[2], locked=True)
        self._params.x = ManualParameter(at[0], locked=True)
        self._params.y = ManualParameter(at[1], locked=True)
        self._params.z = ManualParameter(at[2], locked=True)
        
    def __repr__(self):
        label = self.__class__.__name__[:-5]
        return('[{}]Point_{}(at={})'.format(label, self.name, str(self.r.tolist())))
        
    @property
    def r(self):
        if self.linkage.use_manual_params:
            return(torch.cat([self._params.x(), self._params.y(), self._params.z()]))
        else:
            return(torch.cat([self.params.x(), self.params.y(), self.params.z()]))
        
    def root(self):
        return(self)
    
    def E(self):
        return(0)
    
class OnPointPoint(Point):
    def __init__(self, linkage, name, parent):
        super(OnPointPoint, self).__init__(linkage, name)
        self.parent = parent
        
    def __repr__(self):
        label = self.__class__.__name__[:-5]
        return('[{}]Point_{}(on={})'.format(label, self.name, str(self.parent)))
        
    @property
    def r(self):
        return(self.parent.r)
    
    def root(self):
        return(self.parent.root())
    
    def E(self):
        return(0)
    
class ToPointPoint(Point):
    def __init__(self, linkage, name, at, parent):
        super(ToPointPoint, self).__init__(linkage, name)
        self.parent = parent
        self.params.x = Parameter(at[0], locked=False)
        self.params.y = Parameter(at[1], locked=False)
        self.params.z = Parameter(at[2], locked=False)
        self._params.x = ManualParameter(at[0], locked=False)
        self._params.y = ManualParameter(at[1], locked=False)
        self._params.z = ManualParameter(at[2], locked=False)
        
    def __repr__(self):
        label = self.__class__.__name__[:-5]
        return('[{}]Point_{}(to={})'.format(label, self.name, str(self.parent)))
        
    @property
    def r(self):
        if self.linkage.use_manual_params:
            return(torch.cat([self._params.x(), self._params.y(), self._params.z()]))
        else:
            return(torch.cat([self.params.x(), self.params.y(), self.params.z()]))
    
    def root(self):
        return(self)
    
    def E(self):
        return((self.r-self.parent.r).pow(2).sum())
    
class CalculatedPoint(Point):
    def __init__(self, linkage, name, parent):
        super(CalculatedPoint, self).__init__(linkage, name)
        self.parent = parent
        
    def __repr__(self):
        label = self.__class__.__name__[:-5]
        return('[{}]Point_{}(from={})'.format(label, self.name, str(self.parent.p1)))
        
    @property
    def r(self):
        if self.parent.ux is None:
            ax = torch.tensor([1,0,0], requires_grad=False).to(torch.float)
        elif type(self.parent.ux) is list:
            ax = torch.tensor(ux, requires_grad=False).to(torch.float)
        elif type(self.parent.ux).__bases__[0].__name__ is 'Line':
            ax = self.parent.ux.u
        else:
            raise Exception('ux must be None, a list, or a Line.')
        if self.parent.uz is None:
            az = torch.tensor([0,0,1], requires_grad=False).to(torch.float)
        elif type(self.parent.uz) is list:
            az = torch.tensor(uz, requires_grad=False).to(torch.float)
        elif type(self.parent.uz).__bases__[0].__name__ is 'Line':
            az = self.parent.uz.u
        else:
            raise Exception('uz must be None, a list, or a Line.')
        ay = torch.cross(az, ax)
        if self.linkage.use_manual_params:
            theta = self.parent._params.theta()*10
            phi = self.parent._params.phi()*10            
        else:
            theta = self.parent.params.theta()*10
            phi = self.parent.params.phi()*10
        ux = torch.sin(phi)*torch.cos(theta)
        uy = torch.sin(phi)*torch.sin(theta)
        uz = torch.cos(phi)
        dr = self.parent.L * torch.cat([ux, uy, uz])
        R = torch.stack([ax, ay, az], dim=1)
        r = self.parent.p1.r + torch.matmul(R,dr)
        return(r)
    
    def root(self):
        return(self)
    
    def E(self):
        return(0)
    
class CalculatedAnteriorPoint(Point):
    def __init__(self, linkage, name, parent):
        super(CalculatedAnteriorPoint, self).__init__(linkage, name)
        self.parent = parent
        
    def __repr__(self):
        label = self.__class__.__name__[:-5]
        return('[{}]Point_{}(from={})'.format(label, self.name, str(self.parent.parent)))
        
    @property
    def r(self):
        if self.parent.ux is None:
            ax = torch.tensor([1,0,0], requires_grad=False).to(torch.float)
        elif type(self.parent.ux) is list:
            ax = torch.tensor(ux, requires_grad=False).to(torch.float)
        elif type(self.parent.ux).__bases__[0].__name__ is 'Line':
            ax = self.parent.ux.u
        else:
            raise Exception('ux must be None, a list, or a Line.')
        if self.parent.uz is None:
            az = torch.tensor([0,0,1], requires_grad=False).to(torch.float)
        elif type(self.parent.uz) is list:
            az = torch.tensor(uz, requires_grad=False).to(torch.float)
        elif type(self.parent.uz).__bases__[0].__name__ is 'Line':
            az = self.parent.uz.u
        else:
            raise Exception('uz must be None, a list, or a Line.')
        ay = torch.cross(az, ax)
        if self.linkage.use_manual_params:
            theta = self.parent._params.theta()*10
            phi = self.parent._params.phi()*10         
        else:
            theta = self.parent.params.theta()*10
            phi = self.parent.params.phi()*10
        ux = torch.sin(phi)*torch.cos(theta)
        uy = torch.sin(phi)*torch.sin(theta)
        uz = torch.cos(phi)
        dr = self.parent.L * torch.cat([ux, uy, uz])
        R = torch.stack([ax, ay, az], dim=1)
        if self.linkage.use_manual_params:
            beta = self.parent._params.beta()
        else:
            beta = self.parent.params.beta()
        r = self.parent.parent.r - beta * torch.matmul(R,dr)
        return(r)
    
    def root(self):
        return(self)
    
    def E(self):
        return(0)
    
class CalculatedPosteriorPoint(Point):
    def __init__(self, linkage, name, parent):
        super(CalculatedPosteriorPoint, self).__init__(linkage, name)
        self.parent = parent
        
    def __repr__(self):
        label = self.__class__.__name__[:-5]
        return('[{}]Point_{}(from={})'.format(label, self.name, str(self.parent.parent)))
        
    @property
    def r(self):
        if self.parent.ux is None:
            ax = torch.tensor([1,0,0], requires_grad=False).to(torch.float)
        elif type(self.parent.ux) is list:
            ax = torch.tensor(ux, requires_grad=False).to(torch.float)
        elif type(self.parent.ux).__bases__[0].__name__ is 'Line':
            ax = self.parent.ux.u
        else:
            raise Exception('ux must be None, a list, or a Line.')
        if self.parent.uz is None:
            az = torch.tensor([0,0,1], requires_grad=False).to(torch.float)
        elif type(self.parent.uz) is list:
            az = torch.tensor(uz, requires_grad=False).to(torch.float)
        elif type(self.parent.uz).__bases__[0].__name__ is 'Line':
            az = self.parent.uz.u
        else:
            raise Exception('uz must be None, a list, or a Line.')
        ay = torch.cross(az, ax)
        if self.linkage.use_manual_params:
            theta = self.parent._params.theta()*10
            phi = self.parent._params.phi()*10 
        else:
            theta = self.parent.params.theta()*10
            phi = self.parent.params.phi()*10
        ux = torch.sin(phi)*torch.cos(theta)
        uy = torch.sin(phi)*torch.sin(theta)
        uz = torch.cos(phi)
        dr = self.parent.L * torch.cat([ux, uy, uz])
        R = torch.stack([ax, ay, az], dim=1)
        if self.linkage.use_manual_params:
            beta = self.parent._params.beta()
        else:
            beta = self.parent.params.beta()
        r = self.parent.parent.r + (1-beta) * torch.matmul(R,dr)
        return(r)
    
    def root(self):
        return(self)
    
    def E(self):
        return(0)
    
class Line(torch.nn.Module):
    def __init__(self, linkage, name):
        super(Line, self).__init__()
        self.linkage = linkage
        self.name = name
        self.parent = None
        self.p1 = None
        self.p2 = None
        self.params = Munch({})
        self._params = Munch({})
        
    def __repr__(self):
        raise Exception('Override this method.')
        
    @property
    def r(self):
        return(self.p2.r-self.p1.r)
    
    def E(self):
        raise Exception('Override this method.')
    
    def is_constrained(self):
        raise Exception('Override this method.')
        
    def add_onlinepoint(self, alpha=None):
        new_point = self.linkage.add_onlinepoint(self, alpha)
        return(new_point)
    
    def get_free_params(self):
        free_params = []
        for param in self.params.values():
            if not param.locked:
                free_params.append(param)
        return(free_params)
    
    def set_parameter(self, param_name, value):
        self.params[param_name].tensor = value
        self._params[param_name].tensor = value
        self.linkage.plot.update()
        try:
            self.linkage.update()
        except:
            pass
        
    def lock(self):
        for param_name in self.params.keys():
            self.params[param_name].lock()
            self._params[param_name].lock()
            
    def unlock(self):
        for param_name in self.params.keys():
            self.params[param_name].unlock()
            self._params[param_name].unlock()
    
class FromPointLine(Line):
    def __init__(self, linkage, name, parent, L, theta, phi=None, ux=None, uz=None, locked=False):
        super(FromPointLine, self).__init__(linkage, name)
        self.parent = parent
        self.locked = locked
        self.L = L
        phi = np.pi/2 if phi is None else phi*np.pi/180
        self.ux = ux
        self.uz = uz
        self.p1 = OnPointPoint(self.linkage, '{}.{}'.format(self.name, '1'), parent=parent)
        self.p2 = CalculatedPoint(self.linkage, '{}.{}'.format(self.name, '2'), parent=self)
        self.params.theta = Parameter([theta*np.pi/180/10], locked=self.locked)
        self.params.phi = Parameter([phi/10], locked=True)
        self._params.theta = ManualParameter([theta*np.pi/180/10], locked=self.locked)
        self._params.phi = ManualParameter([phi/10], locked=True)
        
    def __repr__(self):
        label = self.__class__.__name__[:-4]
        return('[{}]Line_{}(p1={}, p2={})'.format(label, self.name, self.p1.name, self.p2.name))
    
    @property
    def u(self):
        return(self.r/self.L)
    
    def E(self):
        return(0)
    
    def is_length_constrained(self):
        return(True)
    
class FromPointsLine(Line):
    def __init__(self, linkage, name, parent1, parent2):
        super(FromPointsLine, self).__init__(linkage, name)
        self.p1 = OnPointPoint(self.linkage, '{}.{}'.format(self.name, '1'), parent=parent1)
        self.p2 = OnPointPoint(self.linkage, '{}.{}'.format(self.name, '2'), parent=parent2)
        self.target_length = None
        
    def __repr__(self):
        label = self.__class__.__name__[:-4]
        return('[{}]Line_{}(p1={}, p2={})'.format(label, self.name, self.p1.name, self.p2.name))
    
    def E(self):
        if self.is_length_constrained() and self.target_length is not None:
            #E = ((self.p2.r-self.p1.r).pow(2).sum().pow(0.5)-self.target_length).pow(2)
            E = (self.p2.r-self.p1.r).pow(2).sum()-torch.tensor(self.target_length).pow(2)
            E = (torch.abs(E)).pow(0.5)
            return(E)
        return(0)
    
    def is_length_constrained(self):
        if self.target_length is not None:
            return(True)
        elif self.p1.root().__class__.__name__ is 'AnchorPoint':
            if self.p2.root().__class__.__name__ is 'AnchorPoint':
                return(True)
        return(False)
    
    def constrain_length(self, L):
        if self.p1.root().__class__.__name__ is 'AnchorPoint':
            if self.p2.root().__class__.__name__ is 'AnchorPoint':
                raise Exception('Cannot constrain the length of a line with anchored endpoints.')
        self.target_length = L
        self.linkage.update()
    
class OnLinePoint(Point):
    def __init__(self, linkage, name, parent, alpha):
        super(OnLinePoint, self).__init__(linkage, name)
        self.parent = parent
        alpha = 0.5 if alpha is None else alpha
        self.params.alpha = Parameter([alpha], locked=False)
        self._params.alpha = ManualParameter([alpha], locked=False)
        
    def __repr__(self):
        label = self.__class__.__name__[:-5]
        return('[{}]Point_{}(on={})'.format(label, self.name, str(self.parent)))
    
    @property
    def r(self):
        if self.linkage.use_manual_params:
            alpha = self._params.alpha()
        else:
            alpha = self.params.alpha()
        return((1-alpha)*self.parent.p1.r+alpha*self.parent.p2.r)

    def root(self):
        return(self)
    
    def E(self):
        return(0)
    
class OnPointLine(Line):
    def __init__(self, linkage, name, parent, L, theta, phi=None, ux=None, uz=None, beta=None):
        super(OnPointLine, self).__init__(linkage, name)
        self.parent = parent
        self.L = L
        phi = np.pi/2 if phi is None else phi*np.pi/180
        self.ux = ux
        self.uz = uz
        beta = 0.5 if beta is None else beta
        self.p1 = CalculatedAnteriorPoint(self.linkage, '{}.{}'.format(self.name, '1'), parent=self)
        self.p2 = CalculatedPosteriorPoint(self.linkage, '{}.{}'.format(self.name, '2'), parent=self)
        self.params.theta = Parameter([theta*np.pi/180/10], locked=False)
        self.params.phi = Parameter([phi/10], locked=True)
        self.params.beta = Parameter([beta], locked=False)
        self._params.theta = ManualParameter([theta*np.pi/180/10], locked=False)
        self._params.phi = ManualParameter([phi/10], locked=True)
        self._params.beta = ManualParameter([beta], locked=False)
        
    def is_length_constrained(self):
        return(True)
    
class Linkage():
    def __init__(self, show_origin=True):       
        self.points = Munch(torch.nn.ModuleDict({}))
        self.lines = Munch(torch.nn.ModuleDict({}))
        self.names = {}
        for _type in ['point', 'line']:
            self.names[_type] = []
            letters = string.ascii_letters[-26:]
            if _type is 'line':
                letters = letters.lower()
            for n in range(3):
                for t in itertools.product(letters, repeat=n):
                    self.names[_type].append(''.join(t))
            self.names[_type] = iter(self.names[_type][1:])
        self.plot = LinkagePlot(self, show_origin)
        self.tolerance = 0.0001
        self.use_manual_params = False
    
    ######################################## Points ########################################
    
    def add_atpoint(self, at):
        name = next(self.names['point'])
        self.points[name] = AtPoint(self, name, at)
        self.plot.update()
        return(self.points[name])
    
    def add_anchorpoint(self, at):
        name = next(self.names['point'])
        self.points[name] = AnchorPoint(self, name, at)
        self.plot.update()
        return(self.points[name])
    
    def add_onpointpoint(self, parent):
        name = next(self.names['point'])
        self.points[name] = OnPointPoint(self, name, parent)
        self.plot.update()
        return(self.points[name])
    
    def add_topointpoint(self, at, parent):
        name = next(self.names['point'])
        self.points[name] = ToPointPoint(self, name, at, parent)
        self.plot.update()
        return(self.points[name])
    
    def add_onlinepoint(self, parent, alpha=None):
        name = next(self.names['point'])
        self.points[name] = OnLinePoint(self, name, parent, alpha)
        self.plot.update()
        return(self.points[name])
    
    ######################################## Lines #########################################
    
    def add_frompointline(self, parent, L, theta, phi=None, ux=None, uz=None, locked=False):
        name = next(self.names['line'])
        self.lines[name] = FromPointLine(self, name, parent, L, theta, phi, ux, uz, locked)
        self.plot.update()
        return(self.lines[name])
    
    def add_frompointsline(self, parent1, parent2):
        name = next(self.names['line'])
        self.lines[name] = FromPointsLine(self, name, parent1, parent2)
        self.plot.update()
        return(self.lines[name])
    
    def add_onpointline(self, parent, L, theta, phi=None, ux=None, uz=None, beta=None):
        name = next(self.names['line'])
        self.lines[name] = OnPointLine(self, name, parent, L, theta, phi, ux, uz, beta)
        self.plot.update()
        return(self.lines[name])
        
    @property
    def N(self):
        N = 0
        N += len(self.points)
        N += 2*len(self.lines)
        return(N)
    
    @property
    def M(self):
        return(len(self.lines))
    
    def set_parameter(self, obj_type, obj_name, param_name, value):
        if obj_type in ['Point', 'point']:
            obj = self.points[obj_name]
        elif obj_type in ['Line', 'line']:
            obj = self.lines[obj_name]
        else:
            raise Exception('Object type must be Point or Line.')
        obj.set_parameter(param_name, value)
    
    def get_param_dict(self):
        parameters = {}
        for point in self.points.values():
            for param_name in point.params.keys():
                param = point.params[param_name]
                for _param in param.parameters():
                    label = 'point_{}_{}'.format(point.name, param_name)
                    parameters[label] = _param
        for line in self.lines.values():
            for param_name in line.params.keys():
                param = line.params[param_name]
                for _param in param.parameters():
                    label = 'line_{}_{}'.format(line.name, param_name)
                    parameters[label] = _param
        return(parameters)
       
    def get_manual_param_dict(self):
        parameters = {}
        for point in self.points.values():
            for param_name in point._params.keys():
                param = point._params[param_name]
                if param.locked:
                    continue
                label = 'point_{}_{}'.format(point.name, param_name)
                parameters[label] = param
        for line in self.lines.values():
            for param_name in line._params.keys():
                param = line._params[param_name]
                if param.locked:
                    continue
                label = 'line_{}_{}'.format(line.name, param_name)
                parameters[label] = param
        return(parameters)
        
    def get_manual_param_list(self):
        manual_param_dict = self.get_manual_param_dict()
        manual_param_list = []
        for param in manual_param_dict.values():
            manual_param_list.append(param.tensor.tolist())
        manual_param_list = list(itertools.chain(*manual_param_list))    
        return(manual_param_list)
        
    def energy(self, use_manual_params=False):
        self.use_manual_params = use_manual_params
        E = 0.0
        for point in self.points.values():
            E += point.E()
        for line in self.lines.values():
            E += line.E()
        self.use_manual_params = False
        return(E)
        
    def set_manual_params(self, x):
        manual_param_dict = self.get_manual_param_dict()
        for i, param_name in enumerate(manual_param_dict.keys()):
            manual_param_dict[param_name].tensor = torch.tensor([x.tolist()[i]], requires_grad=True).to(torch.float)
        
    def apply_manual_params(self):
        for point in self.points.values():
            for param_name in point.params.keys():
                #point.set_parameter(param_name, point._params[param_name].tensor.tolist())
                point.params[param_name] = Parameter(point._params[param_name].tensor.tolist(), 
                                                     locked=point._params[param_name].locked) 
        for line in self.lines.values():
            for param_name in line.params.keys():
                #line.set_parameter(param_name, line._params[param_name].tensor.tolist())
                line.params[param_name] = Parameter(line._params[param_name].tensor.tolist(),
                                                    locked=line._params[param_name].locked) 
      
    '''
    def get_dof_tensor(self):
        manual_param_dict = self.get_manual_param_dict()
        d = 0
        for manual_param in manual_param_dict.values():
            d += len(manual_param.tensor)
        dof_tensor = torch.tensor(torch.zeros(d).tolist(), dtype=torch.float, requires_grad=True)
        for counter, manual_param in enumerate(manual_param_dict.values()):
            dof_tensor[counter] += manual_param.tensor[0]  
        return(dof_tensor)
    '''
    
    def _energy(self, x):
        self.set_manual_params(x)
        E = self.energy(use_manual_params=True)
        return(E)
        
    def _forces(self):
        #E = self.energy(use_manual_params=True)
        #dof_tensor = self.get_dof_tensor()
        manual_param_dict = self.get_manual_param_dict()
        F_list = []
        for manual_param in manual_param_dict.values():
            E = self.energy(use_manual_params=True)
            F = torch.autograd.grad(E, manual_param.tensor,
                retain_graph=True, create_graph=False, allow_unused=True)[0]
            F = torch.tensor([0.0], requires_grad=False) if F is None else F
            F_list.append(F)
        F = torch.cat(F_list).tolist()
        return(F)
        
    def _error_vec(self, x):
        f = np.zeros(len(x))
        f[0] += self._energy(x)
        return(f)
        
    def _error_vec_jacobian(self, x):
        F = self._forces()
        J = np.zeros((len(x),len(x)))
        J[0] += F
        return(J)
        
    '''
    def update(self, max_num_epochs=10000):
        optimizer = torch.optim.SGD(self.get_param_dict().values(), lr=0.001)
        for epoch in range(max_num_epochs):
            optimizer.zero_grad()
            E = self.energy(use_manual_params=False)
            E.backward()
            optimizer.step()
            self.plot.E_list.append(E.item())
            if E <= self.tolerance:
                break
        if False:
            if (E > self.tolerance or E.isnan()):
                raise Exception('Could not solve all constraints.')
        self.plot.update()
        time.sleep(0.01)
    '''
        
    def update(self):
        x0 = self.get_manual_param_list()
        solver = optimize.root(self._error_vec, x0=x0, jac=self._error_vec_jacobian, method='hybr',
                               options={'maxfev': 1000, 'factor': 0.1, 'xtol': 1.0e-04}) 
        xf = solver.x
        self.apply_manual_params()
        if not solver.success:
            raise Exception('Could not solve all constraints.')
        self.plot.E_list.append(0)
        self.plot.update()
        time.sleep(0.01)
        
    def create_controller(self, manual=False):
        
        linkage = self

        if False:
            obj_type_widget = widgets.Dropdown(options=['obj_type', 'point', 'line'])
            obj_name_widget = widgets.Dropdown(options=['obj_name'])
            param_name_widget = widgets.Dropdown(options=['param_name'])
        else:
            obj_type_widget = widgets.Dropdown(options=['point', 'line'])
            obj_name_widget = widgets.Dropdown(options=['D'])
            param_name_widget = widgets.Dropdown(options=['alpha'])    
        value_widget = widgets.FloatSlider(min=0, max=1, step=0.05, value=0.15)

        def update_obj_name_options(*args):
            avail_obj_names = []
            if obj_type_widget.value is 'point':
                for point in linkage.points.values():
                    if list(point.params):
                        avail_obj_names.append(point.name)
            elif obj_type_widget.value is 'line':
                for line in linkage.lines.values():
                    if list(line.params):
                        avail_obj_names.append(line.name)
            else:
                avail_obj_names.append('obj_name')
            obj_name_widget.options = avail_obj_names
        obj_type_widget.observe(update_obj_name_options, 'value')

        def update_param_name_options(*args):
            if obj_type_widget.value is 'point':
                obj = linkage.points[obj_name_widget.value]
                param_name_widget.options = obj.params.keys()
            elif obj_type_widget.value is 'line':
                obj = linkage.lines[obj_name_widget.value]
                param_name_widget.options = obj.params.keys()
            else:
                param_name_widget.options = ['param_name']
        obj_name_widget.observe(update_param_name_options, 'value')

        def update_param_bounds(*args):
            if param_name_widget.value in ['alpha', 'beta']:
                _min = 0
                _max = 1
                _value = 0.15
            elif param_name_widget.value in ['theta', 'phi']:
                _min = 0
                _max = (2*np.pi)/10
                _value = (np.pi/2)/10
            else:
                _min = 0
                _max = 1
                _value = 0.5
            value_widget.min = _min
            value_widget.max = _max
            value_widget.value = _value
        param_name_widget.observe(update_param_bounds, 'value')

        if manual:
            interact_manual(
                linkage.set_parameter,
                obj_type=obj_type_widget,
                obj_name=obj_name_widget,
                param_name=param_name_widget,
                value=value_widget
            );
        else:
            interact(
                linkage.set_parameter,
                obj_type=obj_type_widget,
                obj_name=obj_name_widget,
                param_name=param_name_widget,
                value=value_widget
            );
        
class LinkagePlot():
    def __init__(self, linkage, show_origin=True):
        self.linkage = linkage
        self.origin = torch.tensor([0,0,0])
        self.E_list = [1.0]
        
        # Set up figure and axis
        self.size = 5
        self.lim = 2
        self.fig = plt.figure(figsize=(2*self.size,self.size))
        self.ax1 = self.fig.add_subplot(121, autoscale_on=False,
            xlim=(-self.lim,self.lim),
            ylim=(-self.lim,self.lim))
        self.ax2 = self.fig.add_subplot(122, autoscale_on=False,
            xlim=(0,1),
            ylim=(0,1))
        self.ax1.set_title('Configuration')
        self.ax2.set_title('Energy')
        
        if show_origin:
            self.ax1.scatter(
                [self.origin[0]], [self.origin[1]],
                marker='+', s=50, c='black', alpha=1, label='origin')
        
        self.points, self.anchors, self.lines = {}, {}, {}
            
        #self.lnE_line, = self.ax2.plot([], [], 'b-', markersize=3, lw=0.5, label='log10(E)')
        time_template = ' t={:.0f}\n E={:.2f}\n T={:.5f}\n theta={:.0f}\n'
        self.time_text = self.ax1.text(0.05, 0.7, '', transform=self.ax1.transAxes)
        
    def create_energy_plot(self):
        if 'd' not in self.linkage.lines.keys():
            return()
        if self.linkage.lines['d'].target_length is None:
            return()
        L = 4
        alpha = self.linkage.points['D'].params.alpha().item()
        d_target = self.linkage.lines['d'].target_length
        theta = np.linspace(0, 2*np.pi, 1000)
        beta = np.linspace(0, 2*np.pi, 1000)
        THETA, BETA = np.meshgrid(theta, beta)
        d2 = (L**2/16)*((2-4*alpha+np.cos(THETA)+np.cos(BETA))**2 + (np.sin(THETA)+np.sin(BETA))**2)
        E = ((d2 - d_target**2)**2)**0.5
        E = E**0.5
        self.ax2.contourf(THETA, BETA, E, levels=50, cmap='coolwarm') #gnuplot #gist_stern #coolwarm
        self.ax2.set_xlim([0,2*np.pi])
        self.ax2.set_ylim([0,2*np.pi])
        self.ax2.set_xlabel('theta (rad)')
        self.ax2.set_ylabel('beta (rad)')
        theta = self.linkage.lines['b'].params.theta().item()*10
        beta = self.linkage.lines['c'].params.theta().item()*10
        self.ax2.scatter(x=[theta], y=[beta], c='white', s=25)
        #self.ax2.colorbar()
        
    def update(self):
        
        for point_name in self.linkage.points.keys():
            if self.linkage.points[point_name].__class__.__name__ is 'AnchorPoint':
                color = 'blue'
                size = 150
            else:
                color = 'limegreen'
                size = 50
            if point_name not in self.points.keys():
                point = self.ax1.scatter([], [], s=size, c=color,
                    zorder=0, label=point_name)
                self.points[point_name] = point
            point = self.linkage.points[point_name]
            self.points[point_name].set_offsets(
                [[point.r[0],point.r[1]]])
                
        for line_name in self.linkage.lines.keys():
            ls, lw = ':', 1
            if self.linkage.lines[line_name].is_length_constrained():
                ls, lw = '-', 1
            if line_name not in self.lines.keys():
                line, = self.ax1.plot([], [], linestyle=ls, markersize=3, lw=lw, c='black',
                    zorder=0, label=line_name)
                self.lines[line_name] = line
            line = self.linkage.lines[line_name]
            self.lines[line_name].set_data(
                [line.p1.r[0],line.p2.r[0]],
                [line.p1.r[1],line.p2.r[1]])
            self.lines[line_name].set_linestyle(ls)
            self.lines[line_name].set_linewidth(lw)
            for p in [line.p1, line.p2]:
                if p.name not in self.points.keys():
                    point = self.ax1.scatter([], [], s=10, c='red',
                        zorder=0, label=p.name)
                    self.points[p.name] = point
                self.points[p.name].set_offsets(
                    [[p.r[0],p.r[1]]])
            
        #self.lnE_line.set_xdata(torch.arange(0,len(self.E_list)))
        #self.lnE_line.set_ydata(torch.log10(torch.tensor(self.E_list)))
        #self.ax2.set_xlabel('Epoch')
        #self.ax2.set_xlim(0,len(self.E_list))
        #self.ax2.set_ylim(-10,10)
        self.time_text.set_text('')
        self.create_energy_plot()
        self.fig.canvas.draw()