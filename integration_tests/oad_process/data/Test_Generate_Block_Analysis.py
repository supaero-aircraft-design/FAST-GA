import numpy as np
import openmdao.api as om
from fastoad.openmdao.variables import VariableList


def list_all_subsystem(model, model_address, dict_subsystems):

    try:
        subsystem_list = model._proc_info.keys()
        for subsystem in subsystem_list:
            dict_subsystems = list_all_subsystem(getattr(model, subsystem), model_address+'.'+subsystem, dict_subsystems)
    except:
        dict_subsystems[model_address] = get_type(model)

    return dict_subsystems


def get_type(model):
    raw_type = model.msginfo.split('<')[-1]
    type_alone = raw_type.split(' ')[-1]
    model_type = type_alone[:-1]

    return model_type


class Disc1(om.ExplicitComponent):
    """ An OpenMDAO component to encapsulate Disc1 discipline """

    def setup(self):
        self.add_input("x", val=np.nan, desc="")  # NaN as default for testing connexion check
        self.add_input("z", val=[5, 2], desc="", units="m**2")  # for testing non-None units
        self.add_input("y2", val=1.0, desc="")

        self.add_output("y1", val=1.0, desc="")
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        """
        Evaluates the equation
        y1 = z1**2 + z2 + x1 - 0.2*y2
        """
        z1 = inputs["z"][0]
        z2 = inputs["z"][1]
        x1 = inputs["x"]
        y2 = inputs["y2"]

        outputs["y1"] = z1 ** 2 + z2 + x1 - 0.2 * y2


class Disc2(om.ExplicitComponent):
    """ An OpenMDAO component to encapsulate Disc2 discipline """

    def setup(self):
        self.add_input("y1", val=1.0, desc="")

        self.add_output("y2", val=1.0, desc="")
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        """
        Evaluates the equation
        y2 = y1**(.5) + z1 + z2
        """

        y1 = inputs["y1"]

        # Note: this may cause some issues. However, y1 is constrained to be
        # above 3.16, so lets just let it converge, and the optimizer will
        # throw it out
        if y1.real < 0.0:
            y1 *= -1

        outputs["y2"] = y1 ** 0.5 + 2.0


group = om.Group()
ivc = om.IndepVarComp()
ivc.add_output("x", val=0.0)
ivc.add_output("z", val=[6.0, 3.0], units="m**2")
group.add_subsystem("test_ivc", ivc, promotes=["*"])
group.add_subsystem("disc1", Disc1(), promotes=["x", "z"])

global_group = om.Group()
global_group.add_subsystem("inside_groupe", group, promotes=["*"])
global_group.add_subsystem("disc2", Disc2(), promotes=["*"])
ivc2 = om.IndepVarComp()
ivc2.add_output("test", val=12.0)
global_group.add_subsystem("test_ivc_2", ivc2, promotes=["*"])

problem = om.Problem(global_group)
problem.setup()

vars = VariableList.from_problem(problem, use_initial_values=False, get_promoted_names=True)

# Should be False
print(vars["x"].is_input)
dict = {}
subsystem_list = list_all_subsystem(global_group, "global_group", dict)
list_ivc_index = np.where(np.array(list(subsystem_list.values())) == "IndepVarComp")
test = 1