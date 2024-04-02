import numpy as np
import openmdao.api as om


class Station58Pressure(om.ExplicitComponent):
    def initialize(self):

        self.options.declare("number_of_points", types=int, default=250)

    def setup(self):

        n = self.options["number_of_points"]

        self.add_input("gamma_5", shape=n, val=np.nan)
        self.add_input("static_pressure_0", units="Pa", shape=n, val=np.nan)
        self.add_input("mach_8", shape=n, val=np.nan)

        self.add_output("total_pressure_5", units="Pa", shape=n, val=1e6)

        self.declare_partials(
            of="total_pressure_5",
            wrt=["static_pressure_0", "mach_8"],
            method="exact",
        )
        self.declare_partials(
            of="total_pressure_5",
            wrt="gamma_5",
            method="fd",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        mach_8 = inputs["mach_8"]
        static_pressure_0 = inputs["static_pressure_0"]
        gamma_5 = inputs["gamma_5"]

        total_pressure_5 = static_pressure_0 * (1.0 + (gamma_5 - 1.0) / 2.0 * mach_8 ** 2.0) ** (
            gamma_5 / (gamma_5 - 1.0)
        )

        outputs["total_pressure_5"] = total_pressure_5

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        mach_8 = inputs["mach_8"]
        static_pressure_0 = inputs["static_pressure_0"]
        gamma_5 = inputs["gamma_5"]

        partials["total_pressure_5", "mach_8"] = np.diag(
            static_pressure_0
            * gamma_5
            / (gamma_5 - 1.0)
            * (1.0 + (gamma_5 - 1.0) / 2.0 * mach_8 ** 2.0) ** (gamma_5 / (gamma_5 - 1.0) - 1.0)
            * (gamma_5 - 1.0)
            * mach_8
        )
        partials["total_pressure_5", "static_pressure_0"] = np.diag(
            1.0 + (gamma_5 - 1.0) / 2.0 * mach_8 ** 2.0
        ) ** (gamma_5 / (gamma_5 - 1.0))
