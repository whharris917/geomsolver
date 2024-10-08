import torch
import numpy as np
from base import BaseGeometry
from param import Parameter
from point import (
    Point, AtPoint, AnchorPoint, OnPointPoint, ToPointPoint, OnLinePoint,
    CalculatedAlphaPoint, CalculatedAnteriorPoint, CalculatedPosteriorPoint,
    CalculatedAnteriorGammaPoint, CalculatedPosteriorGammaPoint)
from settings import *

class Line(BaseGeometry):
    def __init__(self, linkage, name):
        super(Line, self).__init__(linkage, name)
        #self.parent = None
        self.p1 = None
        self.p2 = None
        
    @property
    def type(self):
        return('line')
        
    @property
    def r(self):
        return(self.p2.r-self.p1.r)
    
    def get_length(self):
        try:
            return(self.L)
        except:
            return(self.r.pow(2).sum().pow(0.5))
    
    @property
    def u(self):
        return(self.r/self.get_length())
    
    def is_constrained(self):
        raise Exception('Override this method.')
    
    def add_onlinepoint(self, alpha=None):
        new_point = self.linkage.add_onlinepoint(self, alpha)
        return(new_point)
    
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
        self.p2 = CalculatedAlphaPoint(self.linkage, '{}.{}'.format(self.name, '2'), parent=self)
        self.params.theta = Parameter([theta*np.pi/180/ANGLE_FACTOR],
            self, 'theta', range=[0,2*np.pi], units='rad', locked=self.locked)
        self.params.phi = Parameter([phi/ANGLE_FACTOR],
            self, 'phi', range=[0,2*np.pi], units='rad', locked=True)
        
    def __repr__(self):
        label = self.__class__.__name__[:-4]
        return('[{}]Line_{}(p1={}, p2={})'.format(label, self.name, self.p1.name, self.p2.name))
    
    def info(self):
        print('\t', self)
        self.param_info()
    
    def E(self):
        E = 0
        for param in self.params.values():
            E += param.constraint_E()
        return(E)
    
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
    
    def info(self):
        print('\t', self)
        self.param_info()
    
    def constrain_length(self, L):
        if self.p1.root().__class__.__name__ == 'AnchorPoint':
            if self.p2.root().__class__.__name__ == 'AnchorPoint':
                raise Exception('Cannot constrain the length of a line with anchored endpoints.')
        self.target_length = L
        if self.linkage.solve:
            self.linkage.update()
        self.linkage.energy_updated = False
        
    def E(self):
        if self.is_length_constrained() and self.target_length is not None:
            r1 = self.p1.r
            r2 = self.p2.r
            if not self.linkage.use_explicit_coords:
                for d in range(r1.dim()-1):
                    r2 = r2.unsqueeze(0)
                    r1 = r1.unsqueeze(r1.dim()-1)
            E = ((r2-r1).pow(2).sum(-1).pow(0.5)-self.target_length).pow(2)
            #E = E.pow(0.5) ###########################################################
            if not self.linkage.use_manual_params:
                E = E.squeeze()
            return(E)
        return(0)
    
    def is_length_constrained(self):
        if self.target_length is not None:
            return(True)
        '''
        elif self.p1.root().__class__.__name__ == 'AnchorPoint':
            if self.p2.root().__class__.__name__ == 'AnchorPoint':
                return(True)
        '''
        return(False)
        
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
        self.params.theta = Parameter([theta*np.pi/180/ANGLE_FACTOR],
            self, 'theta', range=[0,2*np.pi], units='rad', locked=False)
        self.params.phi = Parameter([phi/ANGLE_FACTOR],
            self, 'phi', range=[0,2*np.pi], units='rad', locked=True)
        self.params.beta = Parameter([beta],
            self, 'beta', range=[0,1], units=None, locked=False)
    
    def __repr__(self):
        return('Debug this.')
    
    def info(self):
        print('\t', self)
        self.param_info()  
        
    def E(self):
        E = 0
        for param in self.params.values():
            E += param.constraint_E()
        return(E)
    
    def is_length_constrained(self):
        return(True)
    
class OnPointsLine(Line):
    def __init__(self, linkage, name, parent1, parent2, L, gamma=None):
        super(OnPointsLine, self).__init__(linkage, name)
        self.parent1 = parent1
        self.parent2 = parent2
        self.L = L
        gamma = 0.5 if gamma is None else gamma
        self.p1 = CalculatedAnteriorGammaPoint(self.linkage, '{}.{}'.format(self.name, '1'), parent=self)
        self.p2 = CalculatedPosteriorGammaPoint(self.linkage, '{}.{}'.format(self.name, '2'), parent=self)
        self.params.gamma = Parameter([gamma], self, 'gamma', range=[0,1], units=None, locked=False)
    
    def __repr__(self):
        return('Debug this.')
    
    def info(self):
        print('\t', self)
        self.param_info()
    
    def E(self):
        E = 0
        for param in self.params.values():
            E += param.constraint_E()
        return(E)
    
    def is_length_constrained(self):
        return(True)