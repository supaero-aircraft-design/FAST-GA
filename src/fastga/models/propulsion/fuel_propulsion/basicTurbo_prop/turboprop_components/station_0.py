import numpy as np
import openmdao.api as om
from stdatm import Atmosphere


class Station0(om.ExplicitComponent):
    """
    Some classes used for the computation of the off-design point will have to be slightly
    altered for the design point because the name of their input is a sizing parameter. To reduce
    the amount of code necessary, when the formula for the output does not change, we will simply
    add an option instead or rewriting the component.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.input_mach_name = "mach_0"

    def initialize(self):

        self.options.declare("number_of_points", types=int, default=250)
        self.options.declare("design_point", types=bool, default=False)

    def setup(self):

        n = self.options["number_of_points"]

        if self.options["design_point"]:
            self.input_mach_name = "data:propulsion:turboprop:design_point:mach"

        self.add_input(self.input_mach_name, val=np.nan, shape=n)
        self.add_input("static_temperature_0", units="K", shape=n, val=np.nan)
        self.add_input("static_pressure_0", units="Pa", shape=n, val=np.nan)

        self.add_output("total_temperature_0", units="K", shape=n)
        self.add_output("total_pressure_0", units="Pa", shape=n)

        self.declare_partials(
            of="total_temperature_0",
            wrt=[self.input_mach_name, "static_temperature_0"],
            method="exact",
        )
        self.declare_partials(
            of="total_pressure_0",
            wrt=[self.input_mach_name, "static_pressure_0"],
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        mach_0 = inputs[self.input_mach_name]

        static_temperature_0 = inputs["static_temperature_0"]
        static_pressure_0 = inputs["static_pressure_0"]

        gamma = 1.4

        total_factor = 1.0 + (gamma - 1.0) / 2.0 * mach_0 ** 2.0

        outputs["total_temperature_0"] = static_temperature_0 * total_factor
        outputs["total_pressure_0"] = static_pressure_0 * total_factor ** (gamma / (gamma - 1.0))

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        mach_0 = inputs[self.input_mach_name]

        gamma = 1.4

        static_temperature_0 = inputs["static_temperature_0"]
        static_pressure_0 = inputs["static_pressure_0"]

        total_factor = 1.0 + (gamma - 1.0) / 2.0 * mach_0 ** 2.0

        d_total_factor_d_mach_0 = (gamma - 1.0) * mach_0

        partials["total_temperature_0", self.input_mach_name] = np.diag(
            static_temperature_0 * d_total_factor_d_mach_0
        )
        partials["total_temperature_0", "static_temperature_0"] = np.diag(total_factor)

        partials["total_pressure_0", self.input_mach_name] = np.diag(
            static_pressure_0
            * gamma
            / (gamma - 1.0)
            * total_factor ** (gamma / (gamma - 1.0) - 1.0)
            * d_total_factor_d_mach_0
        )
        partials["total_pressure_0", "static_pressure_0"] = np.diag(
            total_factor ** (gamma / (gamma - 1.0) - 1.0)
        )


class Station0Static(om.ExplicitComponent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.atm = None
        self.input_alt_name = "altitude"

    def initialize(self):

        self.options.declare("number_of_points", types=int, default=250)
        self.options.declare("design_point", types=bool, default=False)

    def setup(self):

        n = self.options["number_of_points"]

        if self.options["design_point"]:
            self.input_alt_name = "data:propulsion:turboprop:design_point:altitude"

        self.add_input(self.input_alt_name, units="m", shape=n, val=np.nan)

        self.add_output("static_temperature_0", units="K", shape=n)
        self.add_output("static_pressure_0", units="Pa", shape=n)

        self.declare_partials(of="*", wrt="*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        # TODO: In later version of stdatm a class with analytic computation of partials exists.
        #  Increasing the minimum version to allow that would however require us to drop the
        #  support for Python 3.7 which is another task on its own.
        self.atm = Atmosphere(altitude=inputs[self.input_alt_name], altitude_in_feet=False)

        outputs["static_temperature_0"] = self.atm.temperature
        outputs["static_pressure_0"] = self.atm.pressure
