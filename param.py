from base import BaseParameter

class Parameter(BaseParameter):
    def __init__(self, tensor, parent, name, range, units, locked=False):
        super(Parameter, self).__init__(tensor, parent, name, range, units, locked)
        
class ManualParameter(BaseParameter):
    def __init__(self, tensor, parent, name, range, units, locked=False):
        super(ManualParameter, self).__init__(tensor, parent, name, range, units, locked)