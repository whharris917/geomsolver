from base import BaseParameter

class Parameter(BaseParameter):
    def __init__(self, tensor, parent, name, range, units, locked=False):
        super(Parameter, self).__init__(tensor, parent, name, range, units, locked)
        self.manual = ManualParameter(tensor, parent, name, range, units, locked)
        self.backup = ManualParameter([], parent, name, range, units, locked)
       
    def reset(self):
        self.backup.tensor = self.manual.tensor.tolist()
        self.manual.tensor = self.tensor.tolist()
        
    def restore(self):
        self.manual.tensor = self.backup.tensor.tolist()
        self.backup.tensor = []
        
class ManualParameter(BaseParameter):
    def __init__(self, tensor, parent, name, range, units, locked=False):
        super(ManualParameter, self).__init__(tensor, parent, name, range, units, locked)