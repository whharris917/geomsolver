from base import BaseParameter

class Parameter(BaseParameter):
    def __init__(self, tensor, locked=False):
        super(Parameter, self).__init__(tensor, locked)
        
class ManualParameter(BaseParameter):
    def __init__(self, tensor, locked=False):
        super(ManualParameter, self).__init__(tensor, locked)