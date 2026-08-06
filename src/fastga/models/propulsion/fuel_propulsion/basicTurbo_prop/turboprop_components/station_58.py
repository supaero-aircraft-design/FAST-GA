import numpy as np
import openmdao.api as om


class Station58Pressure(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("number_of_points", types=int, default=250)

    def setup(self):
        n = self.options["number_of_points"]

        self.add_input("gamma_5", shape=n, val=np.nan, units="unitless")
        self.add_input("static_pressure_0", units="Pa", shape=n, val=np.nan)
        self.add_input("mach_8", shape=n, val=np.nan, units="unitless")

        self.add_output("total_pressure_5", units="Pa", shape=n, val=1e6)

    # pylint: disable=missing-function-docstring
    # Overriding OpenMDAO setup_partials
    def setup_partials(self):
        n = self.options["number_of_points"]

        self.declare_partials(
            of="total_pressure_5",
            wrt=["static_pressure_0", "mach_8"],
            method="exact",
            rows=np.arange(n),
            cols=np.arange(n),
        )
        self.declare_partials(
            of="total_pressure_5",
            wrt="gamma_5",
            method="fd",
            rows=np.arange(n),
            cols=np.arange(n),
        )

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute, not all arguments are used
    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        mach_8 = inputs["mach_8"]
        static_pressure_0 = inputs["static_pressure_0"]
        gamma_5 = inputs["gamma_5"]

        total_pressure_5 = static_pressure_0 * (1.0 + (gamma_5 - 1.0) / 2.0 * mach_8**2.0) ** (
            gamma_5 / (gamma_5 - 1.0)
        )

        outputs["total_pressure_5"] = total_pressure_5

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        mach_8 = inputs["mach_8"]
        static_pressure_0 = inputs["static_pressure_0"]
        gamma_5 = inputs["gamma_5"]

        partials["total_pressure_5", "mach_8"] = (
            static_pressure_0
            * gamma_5
            / (gamma_5 - 1.0)
            * (1.0 + (gamma_5 - 1.0) / 2.0 * mach_8**2.0) ** (gamma_5 / (gamma_5 - 1.0) - 1.0)
            * (gamma_5 - 1.0)
            * mach_8
        )
        partials["total_pressure_5", "static_pressure_0"] = (
            1.0 + (gamma_5 - 1.0) / 2.0 * mach_8**2.0
        ) ** (gamma_5 / (gamma_5 - 1.0))
