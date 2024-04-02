import numpy as np
import openmdao.api as om


class AirCoefficientReader(om.ExplicitComponent):
    """
    Some classes used for the computation of the off-design point will have to be slightly
    altered for the design point because the name of their input is a sizing parameter. To reduce
    the amount of code necessary, when the formula for the output does not change, we will simply
    add an option instead or rewriting the component.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.cv_t_coefficients = None
        self.cp_t_coefficients = None
        self.gamma_coefficients = None

        self.d_cv_t_coefficients = None
        self.d_cp_t_coefficients = None
        self.d_gamma_coefficients = None

        self.input_name = "total_temperature"

    def initialize(self):

        self.options.declare("number_of_points", types=int, default=250)
        self.options.declare(
            "temperature_list",
            types=np.ndarray,
            default=np.linspace(200.0, 2000.0, 91),
        )
        self.options.declare(
            "cv_list",
            types=np.ndarray,
            default=np.linspace(715.0, 963.0, 91),
        )
        self.options.declare(
            "cp_list",
            types=np.ndarray,
            default=np.linspace(1002.0, 1050.0, 91),
        )
        self.options.declare(
            "gamma_list",
            types=np.ndarray,
            default=np.linspace(1.400, 1.298, 91),
        )
        self.options.declare("design_point", types=bool, default=False)

    def setup(self):

        if self.options["design_point"]:
            self.input_name = "data:propulsion:turboprop:design_point:turbine_entry_temperature"

        n = self.options["number_of_points"]

        self.add_input(self.input_name, units="K", shape=n, val=np.nan)

        self.add_output("cp", shape=n, val=1024)
        self.add_output("cv", shape=n, val=731)
        self.add_output("gamma", shape=n, val=1.4)

        self.cv_t_coefficients = np.polyfit(
            self.options["temperature_list"], self.options["cv_list"], 15
        )
        self.cp_t_coefficients = np.polyfit(
            self.options["temperature_list"], self.options["cp_list"], 15
        )
        self.gamma_coefficients = np.polyfit(
            self.options["temperature_list"], self.options["gamma_list"], 15
        )

        self.d_cv_t_coefficients = np.polyder(self.cv_t_coefficients)
        self.d_cp_t_coefficients = np.polyder(self.cp_t_coefficients)
        self.d_gamma_coefficients = np.polyder(self.gamma_coefficients)

        self.declare_partials(of="*", wrt=self.input_name, method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        total_temperature = inputs[self.input_name]

        cv = np.polyval(self.cv_t_coefficients, total_temperature)
        cp = np.polyval(self.cp_t_coefficients, total_temperature)
        gamma = np.polyval(self.gamma_coefficients, total_temperature)

        outputs["cp"] = cp
        outputs["cv"] = cv
        outputs["gamma"] = gamma

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        total_temperature = inputs[self.input_name]

        partials["cp", self.input_name] = np.diag(
            np.polyval(self.d_cp_t_coefficients, total_temperature)
        )
        partials["cv", self.input_name] = np.diag(
            np.polyval(self.d_cv_t_coefficients, total_temperature)
        )
        partials["gamma", self.input_name] = np.diag(
            np.polyval(self.d_gamma_coefficients, total_temperature)
        )
