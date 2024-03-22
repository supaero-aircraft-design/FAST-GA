import numpy as np
import openmdao.api as om


class Station02(om.ExplicitComponent):
    def initialize(self):

        self.options.declare("number_of_points", types=int, default=250)

    def setup(self):

        n = self.options["number_of_points"]

        self.add_input("total_temperature_0", units="K", shape=n, val=np.nan)
        self.add_input("total_pressure_0", units="Pa", shape=n, val=np.nan)
        self.add_input("total_pressure_loss_02", shape=1, val=1.0)

        self.add_output("total_temperature_2", units="K", shape=n, val=3e2)
        self.add_output("total_pressure_2", units="Pa", shape=n, val=0.5e6)

        self.declare_partials(of="total_temperature_2", wrt="total_temperature_0", method="exact")
        self.declare_partials(
            of="total_pressure_2",
            wrt=["total_pressure_0", "total_pressure_loss_02"],
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        total_temperature_0 = inputs["total_temperature_0"]
        total_pressure_0 = inputs["total_pressure_0"]

        pi_02 = inputs["total_pressure_loss_02"]

        outputs["total_temperature_2"] = total_temperature_0
        outputs["total_pressure_2"] = total_pressure_0 * pi_02

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        n = self.options["number_of_points"]

        partials["total_temperature_2", "total_temperature_0"] = np.eye(n)

        partials["total_pressure_2", "total_pressure_0"] = inputs[
            "total_pressure_loss_02"
        ] * np.eye(n)
        partials["total_pressure_2", "total_pressure_loss_02"] = inputs["total_pressure_0"]
