    '''
    def get_manual_param_dict(self):
        parameters = {}
        for point in self.points.values():
            for param_name in point._params.keys():
                param = point._params[param_name]
                if param.locked:
                    continue
                label = 'point.{}.{}'.format(point.name, param_name)
                parameters[label] = param
        for line in self.lines.values():
            for param_name in line._params.keys():
                param = line._params[param_name]
                if param.locked:
                    continue
                label = 'line.{}.{}'.format(line.name, param_name)
                parameters[label] = param
        return(parameters)
       
    def get_manual_param_list(self):
        manual_param_dict = self.get_manual_param_dict()
        manual_param_list = []
        for param in manual_param_dict.values():
            manual_param_list.append(param.tensor.tolist())
        manual_param_list = list(itertools.chain(*manual_param_list))    
        return(manual_param_list)
        
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
    
    def get_dof_tensor(self):
        manual_param_dict = self.get_manual_param_dict()
        d = 0
        for manual_param in manual_param_dict.values():
            d += len(manual_param.tensor)
        dof_tensor = torch.tensor(torch.zeros(d).tolist(), dtype=torch.float, requires_grad=True)
        for counter, manual_param in enumerate(manual_param_dict.values()):
            dof_tensor[counter] += manual_param.tensor[0]  
        return(dof_tensor)
    
    def _energy(self, x):
        self.set_manual_params(x)
        E = self.energy(use_manual_params=True)
        return(E)
        
    def _forces(self):
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
        
    def update(self):
        x0 = self.get_manual_param_list()
        solver = optimize.root(
            self._error_vec, x0=x0, jac=self._error_vec_jacobian, method='hybr',
            options={'maxfev': 1000, 'factor': 0.1, 'xtol': XTOL}) 
        xf = solver.x
        self.apply_manual_params()
        if not solver.success:
            raise Exception('Could not solve all constraints.')
        self.config_plot.update()
        time.sleep(0.01)
    '''