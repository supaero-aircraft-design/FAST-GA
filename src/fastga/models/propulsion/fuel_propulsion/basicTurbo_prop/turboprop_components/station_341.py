import numpy as np
import openmdao.api as om


class Station341Pressure(om.ExplicitComponent):
    def initialize(self):

        self.options.declare("number_of_points", types=int, default=250)

    def setup(self):

        n = self.options["number_of_points"]

        self.add_input("total_pressure_3", units="Pa", shape=n, val=np.nan)
        self.add_input("pressure_loss_34", shape=1, val=1.0)

        self.add_output("total_pressure_41", units="Pa", shape=n, val=1e5)

        self.declare_partials(
            of="total_pressure_41",
            wrt=["total_pressure_3", "pressure_loss_34"],
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        total_pressure_3 = inputs["total_pressure_3"]
        pressure_loss_34 = inputs["pressure_loss_34"]

        outputs["total_pressure_41"] = total_pressure_3 * pressure_loss_34

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        n = self.options["number_of_points"]

        total_pressure_3 = inputs["total_pressure_3"]
        pressure_loss_34 = inputs["pressure_loss_34"]

        partials["total_pressure_41", "total_pressure_3"] = np.eye(n) * pressure_loss_34
        partials["total_pressure_41", "pressure_loss_34"] = total_pressure_3
