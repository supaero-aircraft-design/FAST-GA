import numpy as np
import openmdao.api as om
from stdatm import AtmosphereWithPartials


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

        self.add_input(self.input_mach_name, val=np.nan, shape=n, units="unitless")
        self.add_input("static_temperature_0", units="K", shape=n, val=np.nan)
        self.add_input("static_pressure_0", units="Pa", shape=n, val=np.nan)

        self.add_output("total_temperature_0", units="K", shape=n)
        self.add_output("total_pressure_0", units="Pa", shape=n)

    # pylint: disable=missing-function-docstring
    # Overriding OpenMDAO setup_partials
    def setup_partials(self):
        n = self.options["number_of_points"]
        self.declare_partials(
            of="total_temperature_0",
            wrt=[self.input_mach_name, "static_temperature_0"],
            method="exact",
            rows=np.arange(n),
            cols=np.arange(n),
        )
        self.declare_partials(
            of="total_pressure_0",
            wrt=[self.input_mach_name, "static_pressure_0"],
            method="exact",
            rows=np.arange(n),
            cols=np.arange(n),
        )

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute, not all arguments are used
    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        mach_0 = inputs[self.input_mach_name]

        static_temperature_0 = inputs["static_temperature_0"]
        static_pressure_0 = inputs["static_pressure_0"]

        gamma = 1.4

        total_factor = 1.0 + (gamma - 1.0) / 2.0 * mach_0**2.0

        outputs["total_temperature_0"] = static_temperature_0 * total_factor
        outputs["total_pressure_0"] = static_pressure_0 * total_factor ** (gamma / (gamma - 1.0))

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute_partials, not all arguments are used
    def compute_partials(self, inputs, partials, discrete_inputs=None):
        mach_0 = inputs[self.input_mach_name]

        gamma = 1.4

        static_temperature_0 = inputs["static_temperature_0"]
        static_pressure_0 = inputs["static_pressure_0"]

        total_factor = 1.0 + (gamma - 1.0) / 2.0 * mach_0**2.0

        d_total_factor_d_mach_0 = (gamma - 1.0) * mach_0

        partials["total_temperature_0", self.input_mach_name] = (
            static_temperature_0 * d_total_factor_d_mach_0
        )
        partials["total_temperature_0", "static_temperature_0"] = total_factor

        partials["total_pressure_0", self.input_mach_name] = (
            static_pressure_0
            * gamma
            / (gamma - 1.0)
            * total_factor ** (gamma / (gamma - 1.0) - 1.0)
            * d_total_factor_d_mach_0
        )
        partials["total_pressure_0", "static_pressure_0"] = total_factor ** (
            gamma / (gamma - 1.0) - 1.0
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

    # pylint: disable=missing-function-docstring
    # Overriding OpenMDAO setup_partials
    def setup_partials(self):
        n = self.options["number_of_points"]
        self.declare_partials(
            of="*",
            wrt="*",
            method="exact",
            rows=np.arange(n),
            cols=np.arange(n),
        )

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute, not all arguments are used
    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        self.atm = AtmosphereWithPartials(
            altitude=inputs[self.input_alt_name], altitude_in_feet=False
        )

        outputs["static_temperature_0"] = self.atm.temperature
        outputs["static_pressure_0"] = self.atm.pressure

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute_partials, not all arguments are used
    def compute_partials(self, inputs, partials, discrete_inputs=None):

        partials["static_temperature_0", self.input_alt_name] = (
            self.atm.partial_temperature_altitude
        )
        partials["static_pressure_0", self.input_alt_name] = self.atm.partial_pressure_altitude
