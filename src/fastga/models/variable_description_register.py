import fastoad
import numpy as np
from fastoad.module_management.service_registry import RegisterOpenMDAOSystem
from fastoad.module_management.constants import ModelDomain

from openmdao.core.explicitcomponent import ExplicitComponent


@RegisterOpenMDAOSystem("fastga.variable_description.register", domain=ModelDomain.OTHER)
class VariableDescriptionRegister(ExplicitComponent):
    def setup(self):
        self.add_input("data:geometry:cabin:seats:passenger:NPAX_max", val=np.nan)
