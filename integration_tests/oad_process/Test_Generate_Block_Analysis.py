import numpy as np
import openmdao.api as om
import os.path as pth
from fastoad.openmdao.variables import VariableList
from tests.testing_utilities import run_system, get_indep_var_comp, list_inputs
from fastga.command import api

DATA_FOLDER_PATH = pth.join(pth.dirname(__file__), "data")


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
        self.add_input("data:y4", val=np.nan, desc="")
        self.add_input("foo", val=1.0, desc="")

        self.add_output("y2", val=1.0, desc="")
        self.add_output("y3", val=1.0, desc="")
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        """
        Evaluates the equation
        y2 = y1**(.5) + z1 + z2
        """

        y4 = inputs["data:y4"]
        foo = inputs["foo"]

        # Note: this may cause some issues. However, y1 is constrained to be
        # above 3.16, so lets just let it converge, and the optimizer will
        # throw it out
        if y4.real < 0.0:
            y4 *= -1

        outputs["y2"] = y4 ** 0.5 + 2.0 * foo
        outputs["y3"] = 2 * y4 + np.sqrt(foo)


class Disc3(om.ExplicitComponent):
    """ An OpenMDAO component to encapsulate Disc2 discipline """

    def setup(self):
        self.add_input("y1", val=1.0, desc="")

        self.add_output("data:y4", desc="")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        outputs["data:y4"] = inputs["y1"]


group = om.Group()
ivc = om.IndepVarComp()
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


class InsideGroup(om.Group):

    def setup(self):
        self.add_subsystem("test_ivc", ivc, promotes=["*"])
        self.add_subsystem("disc1", Disc1(), promotes=["x", "y2", "z", "y1"])


class TestBlockAnalysis(om.Group):

    def setup(self):
        self.add_subsystem("inside_group", InsideGroup(), promotes=["*"])
        self.add_subsystem("disc3", Disc3(), promotes=["y1"])
        self.add_subsystem("disc2", Disc2(), promotes=["foo", "y2", "y3"])

        self.connect("disc3.data:y4", "disc2.data:y4")


vars = VariableList.from_problem(problem, use_initial_values=False, get_promoted_names=True)

# Should be False
print(vars["x"].is_input)
dict = {}
subsystem_list = list_all_subsystem(global_group, "global_group", dict)
list_ivc_index = np.where(np.array(list(subsystem_list.values())) == "IndepVarComp")

ref_inputs = pth.join(DATA_FOLDER_PATH, "test.xml")

compute_aero_HS = api.generate_block_analysis(TestBlockAnalysis(), ["foo"], ref_inputs, False)

input_dict = {"foo": (1.0, None)}

outputs_dict = compute_aero_HS(input_dict)

print(outputs_dict)

try:
    ivc_run_system = get_indep_var_comp(list_inputs(TestBlockAnalysis()), __file__, ref_inputs)

    # Run problem and check obtained value(s) is/(are) correct
    # noinspection PyTypeChecker
    problem = run_system(TestBlockAnalysis(), ivc_run_system)
    print(problem["y3"])
except:
    print("Pas de run system")


