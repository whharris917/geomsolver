import torch
from base import BaseGeometry
from param import Parameter, ManualParameter
from settings import *

class Point(BaseGeometry):
    def __init__(self, linkage, name):
        super(Point, self).__init__(linkage, name)

    @property
    def type(self):
        return('point')
    
    @property
    def r(self):
        raise Exception('Override this property.')

    def root(self):
        raise Exception('Override this method.')
        
    def add_frompointline(self, L, theta, phi=None, ux=None, uz=None, locked=False):
        new_line = self.linkage.add_frompointline(self, L, theta, phi, ux, uz, locked)
        return(new_line)
    
    def add_onpointline(self, L, theta, phi=None, ux=None, uz=None, beta=None):
        new_line = self.linkage.add_onpointline(self, L, theta, phi, ux, uz, beta)
        return(new_line)
    
class AtPoint(Point):
    def __init__(self, linkage, name, at):
        super(AtPoint, self).__init__(linkage, name)
        self.params.x = Parameter(at[0], self, 'x', range=[-10,10], units='m', locked=False)
        self.params.y = Parameter(at[1], self, 'y', range=[-10,10], units='m', locked=False)
        self.params.z = Parameter(at[2], self, 'z', range=[-10,10], units='m', locked=False)
        self._params.x = ManualParameter(at[0], self, 'x', range=[-10,10], units='m', locked=False)
        self._params.y = ManualParameter(at[1], self, 'y', range=[-10,10], units='m', locked=False)
        self._params.z = ManualParameter(at[2], self, 'z', range=[-10,10], units='m', locked=False)
    
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
        self.params.x = Parameter(at[0], self, 'x', range=[-10,10], units='m', locked=True)
        self.params.y = Parameter(at[1], self, 'y', range=[-10,10], units='m', locked=True)
        self.params.z = Parameter(at[2], self, 'z', range=[-10,10], units='m', locked=True)
        self._params.x = ManualParameter(at[0], self, 'x', range=[-10,10], units='m', locked=True)
        self._params.y = ManualParameter(at[1], self, 'y', range=[-10,10], units='m', locked=True)
        self._params.z = ManualParameter(at[2], self, 'z', range=[-10,10], units='m', locked=True)
        
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
        self.params.x = Parameter(at[0], self, 'x', range=[-10,10], units='m', locked=False)
        self.params.y = Parameter(at[1], self, 'y', range=[-10,10], units='m', locked=False)
        self.params.z = Parameter(at[2], self, 'z', range=[-10,10], units='m', locked=False)
        self._params.x = ManualParameter(at[0], self, 'x', range=[-10,10], units='m', locked=False)
        self._params.y = ManualParameter(at[1], self, 'y', range=[-10,10], units='m', locked=False)
        self._params.z = ManualParameter(at[2], self, 'z', range=[-10,10], units='m', locked=False)
        
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
    
    def get_dr(self):
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
            theta = self.parent._params.theta()*ANGLE_FACTOR
            phi = self.parent._params.phi()*ANGLE_FACTOR      
        else:
            theta = self.parent.params.theta()*ANGLE_FACTOR
            phi = self.parent.params.phi()*ANGLE_FACTOR
        theta = theta.view(-1,1)
        phi = phi.view(1,-1)
        ux = torch.sin(phi)*torch.cos(theta)
        uy = torch.sin(phi)*torch.sin(theta)
        uz = torch.cos(phi).expand(ux.shape[0],ux.shape[1])
        dr = self.parent.L * torch.stack([ux, uy, uz], dim=2) #torch.cat([ux, uy, uz])
        R = torch.stack([ax, ay, az], dim=1)
        dr = torch.matmul(R.unsqueeze(0).unsqueeze(0), dr.unsqueeze(3)) #torch.matmul(R,dr)
        dr = dr.squeeze()
        return(dr)
    
    def root(self):
        return(self)
    
    def E(self):
        return(0)
    
class CalculatedAlphaPoint(CalculatedPoint):
    def __init__(self, linkage, name, parent):
        super(CalculatedAlphaPoint, self).__init__(linkage, name, parent)
        
    def __repr__(self):
        label = self.__class__.__name__[:-5]
        return('[{}]Point_{}(from={})'.format(label, self.name, str(self.parent.p1)))
        
    @property
    def r(self):
        dr = self.get_dr()
        r = self.parent.p1.r + dr
        return(r)
    
class CalculatedAnteriorPoint(CalculatedPoint):
    def __init__(self, linkage, name, parent):
        super(CalculatedAnteriorPoint, self).__init__(linkage, name, parent)
        
    def __repr__(self):
        label = self.__class__.__name__[:-5]
        return('[{}]Point_{}(from={})'.format(label, self.name, str(self.parent.parent)))
        
    @property
    def r(self):
        dr = self.get_dr()
        if self.linkage.use_manual_params:
            beta = self.parent._params.beta()
        else:
            beta = self.parent.params.beta()
        r = self.parent.parent.r - beta * dr
        return(r)
    
class CalculatedAnteriorGammaPoint(CalculatedPoint):
    def __init__(self, linkage, name, parent):
        super(CalculatedAnteriorGammaPoint, self).__init__(linkage, name, parent)
        
    def __repr__(self):
        label = self.__class__.__name__[:-5]
        return('[{}]Point_{}(from={})'.format(label, self.name, str(self.parent.parent)))
        
    @property
    def r(self):   
        if self.linkage.use_manual_params:
            gamma = self.parent._params.gamma()
        else:
            gamma = self.parent.params.gamma()
        gamma = 0.5*(1+torch.tanh(10*(gamma-0.5)))
        dr = self.parent.parent2.r - self.parent.parent1.r
        u = dr/(dr.pow(2).sum().pow(0.5))
        R = self.parent.L*u
        r = self.parent.parent1.r - gamma*(R-dr)
        return(r)
    
class CalculatedPosteriorPoint(CalculatedPoint):
    def __init__(self, linkage, name, parent):
        super(CalculatedPosteriorPoint, self).__init__(linkage, name, parent)
        
    def __repr__(self):
        label = self.__class__.__name__[:-5]
        return('[{}]Point_{}(from={})'.format(label, self.name, str(self.parent.parent)))
        
    @property
    def r(self):
        dr = self.get_dr()
        if dr.dim() == 1:
            dr = dr.view(-1,3)
        if self.linkage.use_manual_params:
            beta = self.parent._params.beta()
        else:
            beta = self.parent.params.beta()
        beta = beta.unsqueeze(beta.dim())
        for d in range(dr.dim()-1):
            beta = beta.unsqueeze(0)
            dr = dr.unsqueeze(dr.dim()-1)
        r = self.parent.parent.r + (1-beta) * dr
        if not self.linkage.use_manual_params:
            r = r.squeeze()
        return(r)

class CalculatedPosteriorGammaPoint(CalculatedPoint):
    def __init__(self, linkage, name, parent):
        super(CalculatedPosteriorGammaPoint, self).__init__(linkage, name, parent)
        
    def __repr__(self):
        label = self.__class__.__name__[:-5]
        return('[{}]Point_{}(from={})'.format(label, self.name, str(self.parent.parent)))
        
    @property
    def r(self):
        dr = self.get_dr()
        if self.linkage.use_manual_params:
            beta = self.parent._params.beta()
        else:
            beta = self.parent.params.beta()
        r = self.parent.parent.r + (1-beta) * dr
        return(r)
    
    @property
    def r(self):   
        if self.linkage.use_manual_params:
            gamma = self.parent._params.gamma()
        else:
            gamma = self.parent.params.gamma()
        gamma = 0.5*(1+torch.tanh(10*(gamma-0.5)))
        dr = self.parent.parent2.r - self.parent.parent1.r
        u = dr/(dr.pow(2).sum().pow(0.5))
        R = self.parent.L*u
        r = self.parent.parent2.r + (1-gamma)*(R-dr)
        return(r)
    
class OnLinePoint(Point):
    def __init__(self, linkage, name, parent, alpha):
        super(OnLinePoint, self).__init__(linkage, name)
        self.parent = parent
        alpha = 0.5 if alpha is None else alpha
        self.params.alpha = Parameter([alpha], self, 'alpha', range=[0,1], units=None, locked=False)
        self._params.alpha = ManualParameter([alpha], self, 'alpha', range=[0,1], units=None, locked=False)
        
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