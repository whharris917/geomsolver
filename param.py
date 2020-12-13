from base import BaseParameter

class Parameter(BaseParameter):
    def __init__(self, tensor, parent, name, range, units, locked=False):
        super(Parameter, self).__init__(tensor, parent, name, range, units, locked)
        self.manual = ManualParameter(tensor, parent, name, range, units, locked)
        
    def reset_manual(self):
        self.manual.tensor = self.tensor.tolist()
        
class ManualParameter(BaseParameter):
    def __init__(self, tensor, parent, name, range, units, locked=False):
        super(ManualParameter, self).__init__(tensor, parent, name, range, units, locked)