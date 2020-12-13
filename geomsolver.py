import numpy as np
from scipy import optimize
import torch, itertools, string, time
from munch import Munch
from point import AtPoint, AnchorPoint, OnPointPoint, ToPointPoint, OnLinePoint
from line import FromPointLine, FromPointsLine, OnPointLine, OnPointsLine
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
import matplotlib.pyplot as plt
from settings import *
from param import Parameter

class Linkage():
    def __init__(self):       
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
        self.config_plot = None
        self.energy_plot = None
        self.tolerance = TOLERANCE
        self.use_manual_params = False
        
    ######################################## Plots #########################################

    def show_configuration(self, show_origin=True):
        self.config_plot = LinkagePlot(self, show_origin)
    
    def show_energy_plot(self):
        self.energy_plot = EnergyPlot(self)
    
    ######################################## Points ########################################
    
    def add_atpoint(self, at):
        name = next(self.names['point'])
        self.points[name] = AtPoint(self, name, at)
        self.config_plot.update()
        return(self.points[name])
    
    def add_anchorpoint(self, at):
        name = next(self.names['point'])
        self.points[name] = AnchorPoint(self, name, at)
        self.config_plot.update()
        return(self.points[name])
    
    def add_onpointpoint(self, parent):
        name = next(self.names['point'])
        self.points[name] = OnPointPoint(self, name, parent)
        self.config_plot.update()
        return(self.points[name])
    
    def add_topointpoint(self, at, parent):
        name = next(self.names['point'])
        self.points[name] = ToPointPoint(self, name, at, parent)
        self.config_plot.update()
        return(self.points[name])
    
    def add_onlinepoint(self, parent, alpha=None):
        name = next(self.names['point'])
        self.points[name] = OnLinePoint(self, name, parent, alpha)
        self.config_plot.update()
        return(self.points[name])
    
    ######################################## Lines #########################################
    
    def add_frompointline(self, parent, L, theta, phi=None, ux=None, uz=None, locked=False):
        name = next(self.names['line'])
        self.lines[name] = FromPointLine(self, name, parent, L, theta, phi, ux, uz, locked)
        self.config_plot.update()
        return(self.lines[name])
    
    def add_frompointsline(self, parent1, parent2):
        name = next(self.names['line'])
        self.lines[name] = FromPointsLine(self, name, parent1, parent2)
        self.config_plot.update()
        return(self.lines[name])
    
    def add_onpointline(self, parent, L, theta, phi=None, ux=None, uz=None, beta=None):
        name = next(self.names['line'])
        self.lines[name] = OnPointLine(self, name, parent, L, theta, phi, ux, uz, beta)
        self.config_plot.update()
        return(self.lines[name])
        
    def add_onpointsline(self, parent1, parent2, L, gamma=None):
        name = next(self.names['line'])
        self.lines[name] = OnPointsLine(self, name, parent1, parent2, L, gamma)
        self.config_plot.update()
        return(self.lines[name]) 
        
    ########################################################################################
        
    @property
    def N(self):
        N = 0
        N += len(self.points)
        N += 2*len(self.lines)
        return(N)
    
    @property
    def M(self):
        return(len(self.lines))
    
    def get_parameter(self, full_param_name):
        obj_type, obj_name, param_name = full_param_name.split('.')
        if obj_type == 'point':
            return(self.points[obj_name].params[param_name])
        elif obj_type == 'line':
            return(self.lines[obj_name].params[param_name])
        else:
            raise Exception('Invalid parameter name.')
        
    def set_parameter(self, full_param_name, value, manual=False, solve=True, update=True):
        obj_type, obj_name, param_name = full_param_name.split('.')
        if obj_type in ['Point', 'point']:
            obj = self.points[obj_name]
        elif obj_type in ['Line', 'line']:
            obj = self.lines[obj_name]
        else:
            raise Exception('Object type must be point or line.')
        obj.set_parameter(param_name, value, manual, solve, update)
    
    def get_param_dict(self):
        parameters = {}
        for point in self.points.values():
            for param_name in point.params.keys():
                param = point.params[param_name]
                for _param in param.parameters():
                    label = 'point.{}.{}'.format(point.name, param_name)
                    parameters[label] = _param
        for line in self.lines.values():
            for param_name in line.params.keys():
                param = line.params[param_name]
                for _param in param.parameters():
                    label = 'line.{}.{}'.format(line.name, param_name)
                    parameters[label] = _param
        return(parameters)
       
    def energy(self, use_manual_params=False):
        self.use_manual_params = use_manual_params
        E = 0.0
        for point in self.points.values():
            E += point.E()
        for line in self.lines.values():
            E += line.E()
        self.use_manual_params = False
        return(E)
        
    def update(self, max_num_epochs=10000):
        optimizer = torch.optim.SGD(self.get_param_dict().values(), lr=LR)
        for epoch in range(max_num_epochs):
            optimizer.zero_grad()
            E = self.energy(use_manual_params=False)
            E.backward()
            optimizer.step()
            if E <= self.tolerance:
                break
            if epoch % N_UPDATE == 0:
                self.config_plot.update()
                time.sleep(0.01)
        if False:
            if (E > self.tolerance or E.isnan()):
                raise Exception('Could not solve all constraints.')
        self.config_plot.update()
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
                _max = (2*np.pi)/ANGLE_FACTOR
                _value = (np.pi/2)/ANGLE_FACTOR
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
        self.show_origin = show_origin
        self.origin = torch.tensor([0,0,0])
        self.fig_size = FIGSIZE
        self.fig_lim = FIGLIM
        self.build_plot()
        self.points, self.anchors, self.lines = {}, {}, {}
        
    def build_plot(self):
        self.fig = plt.figure(figsize=(self.fig_size,self.fig_size))
        self.ax = self.fig.add_subplot(111, autoscale_on=False,
            xlim=(-self.fig_lim,self.fig_lim),
            ylim=(-self.fig_lim,self.fig_lim))
        self.ax.set_title('Configuration')
        if self.show_origin:
            self.ax.scatter(
                [self.origin[0]], [self.origin[1]],
                marker='+', s=50, c='black', alpha=1, label='origin')
        #time_template = ' t={:.0f}\n E={:.2f}\n T={:.5f}\n theta={:.0f}\n'
        #self.time_text = self.ax.text(0.05, 0.7, '', transform=self.ax.transAxes)
        self.fig.canvas.draw()
    
    def update(self):
        
        for point_name in self.linkage.points.keys():
            if self.linkage.points[point_name].__class__.__name__ is 'AnchorPoint':
                color = 'blue'
                size = 150
            else:
                color = 'limegreen'
                size = 50
            if point_name not in self.points.keys():
                point = self.ax.scatter([], [], s=size, c=color,
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
                line, = self.ax.plot([], [],
                    linestyle=ls, markersize=3, lw=lw, 
                    c='black', zorder=0, label=line_name)
                self.lines[line_name] = line
            line = self.linkage.lines[line_name]
            self.lines[line_name].set_data(
                [line.p1.r[0],line.p2.r[0]],
                [line.p1.r[1],line.p2.r[1]])
            self.lines[line_name].set_linestyle(ls)
            self.lines[line_name].set_linewidth(lw)
            for p in [line.p1, line.p2]:
                if p.name not in self.points.keys():
                    point = self.ax.scatter([], [],
                        s=10, c='red', zorder=0, label=p.name)
                    self.points[p.name] = point
                self.points[p.name].set_offsets(
                    [[p.r[0],p.r[1]]])
            
        #self.time_text.set_text('')
        self.fig.canvas.draw()
        
class EnergyPlot():
    def __init__(self, linkage):
        self.linkage = linkage
        self.fig_size = FIGSIZE
        self.fig_lim = FIGLIM
        self.num_param_steps = NUM_PARAM_STEPS
        self.num_contour_levels = NUM_CONTOUR_LEVELS
        self.cmap = CMAP
        self.x = None
        self.y = None
        self.build_plot()
        
    def build_plot(self):
        self.fig = plt.figure(figsize=(self.fig_size,self.fig_size))
        self.ax = self.fig.add_subplot(111, autoscale_on=False,
            xlim=(0,1),
            ylim=(0,1))
        self.ax.set_title('Energy')
        self.fig.canvas.draw()
        
    def show_controller(self, manual=True, show_configs=False):
        param_dict = self.linkage.get_param_dict()
        x_widget = widgets.Dropdown(options=param_dict.keys())
        y_widget = widgets.Dropdown(options=param_dict.keys())
        if manual:
            interact_manual(self.update, x_name=x_widget, y_name=y_widget, show_configs=show_configs)
        else:
            interact(self.update, x_name=x_widget, y_name=y_widget, show_configs=show_configs)
    
    def update(self, x_name, y_name, show_configs=False):
        self.x = self.linkage.get_parameter(x_name)
        self.y = self.linkage.get_parameter(y_name)
        self.draw_axes()
        self.draw_contour_plot(show_configs)
        
    def draw_axes(self):
        self.ax.set_xlim([self.x.min,self.x.max])
        self.ax.set_ylim([self.y.min,self.y.max])
        self.ax.set_xlabel('{} ({})'.format(self.x.full_name, self.x.units))
        self.ax.set_ylabel('{} ({})'.format(self.y.full_name, self.y.units))
    
    def draw_contour_plot(self, show_configs=False):
        L = 4
        alpha = 0.8
        d_target = 0.3
        x = np.linspace(self.x.min, self.x.max, self.num_param_steps)
        y = np.linspace(self.y.min, self.y.max, self.num_param_steps)
        X, Y = np.meshgrid(x, y)        
        E = np.zeros((X.shape[0], X.shape[1]))
        x0 = self.linkage.get_parameter(self.x.full_name).tensor.tolist()
        y0 = self.linkage.get_parameter(self.y.full_name).tensor.tolist()
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                x, y = X[i][j], Y[i][j]
                self.linkage.set_parameter(self.x.full_name, X[i][j], solve=False, update=show_configs)
                self.linkage.set_parameter(self.y.full_name, Y[i][j], solve=False, update=show_configs)
                E[i][j] = self.linkage.energy()
        self.linkage.set_parameter(self.x.full_name, x0, solve=False, update=True)
        self.linkage.set_parameter(self.y.full_name, y0, solve=False, update=True)
        contourmap = self.ax.contourf(X, Y, E, levels=self.num_contour_levels, cmap=self.cmap)
        x = self.x.tensor.item()
        y = self.y.tensor.item()
        self.ax.scatter(x=[x], y=[y], c='white', s=25)
        #self.fig.colorbar(contourmap)
        self.fig.canvas.draw()