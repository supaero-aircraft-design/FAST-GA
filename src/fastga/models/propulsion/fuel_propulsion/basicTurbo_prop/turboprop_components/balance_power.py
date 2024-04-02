import numpy as np
import openmdao.api as om


class BalancePower(om.ImplicitComponent):
    def initialize(self):

        self.options.declare("number_of_points", types=int, default=250)

    def setup(self):

        n = self.options["number_of_points"]

        self.add_input("shaft_power", units="kW", shape=n, val=np.nan)
        self.add_input("required_shaft_power", units="kW", shape=n, val=np.nan)

        self.add_output("fuel_mass_flow", units="kg/h", shape=n, val=150.0)

        self.declare_partials(of="*", wrt="*", method="exact")

    def apply_nonlinear(
        self, inputs, outputs, residuals, discrete_inputs=None, discrete_outputs=None
    ):

        residuals["fuel_mass_flow"] = 1.0 - inputs["shaft_power"] / inputs["required_shaft_power"]

    def linearize(self, inputs, outputs, jacobian, discrete_inputs=None, discrete_outputs=None):

        jacobian["fuel_mass_flow", "required_shaft_power"] = np.diag(
            inputs["shaft_power"] / inputs["required_shaft_power"] ** 2.0
        )
        jacobian["fuel_mass_flow", "shaft_power"] = -np.diag(1.0 / inputs["required_shaft_power"])
