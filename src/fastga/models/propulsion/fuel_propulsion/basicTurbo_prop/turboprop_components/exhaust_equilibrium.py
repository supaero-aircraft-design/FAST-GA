import numpy as np
import openmdao.api as om


class ExhaustEquilibrium(om.ImplicitComponent):
    def initialize(self):

        self.options.declare("number_of_points", types=int, default=250)

    def setup(self):

        n = self.options["number_of_points"]

        self.add_input("total_pressure_45", units="Pa", shape=n, val=np.nan)
        self.add_input("total_pressure_5", units="Pa", shape=n, val=np.nan)
        self.add_input("total_temperature_45", units="K", shape=n, val=np.nan)
        self.add_input("gamma_45", shape=n, val=np.nan)
        self.add_input("eta_455", shape=1, val=1.0)

        self.add_output("total_temperature_5", units="K", shape=n, val=0.75e3)

        self.declare_partials(of="*", wrt="*", method="exact")

    def apply_nonlinear(
        self, inputs, outputs, residuals, discrete_inputs=None, discrete_outputs=None
    ):

        total_pressure_45 = inputs["total_pressure_45"]
        total_pressure_5 = inputs["total_pressure_5"]
        total_temperature_45 = inputs["total_temperature_45"]
        gamma_45 = inputs["gamma_45"]
        eta_455 = inputs["eta_455"]

        total_temperature_5 = outputs["total_temperature_5"]

        residuals["total_temperature_5"] = 1.0 - total_temperature_45 / total_temperature_5 * (
            total_pressure_5 / total_pressure_45
        ) ** ((gamma_45 - 1.0) / gamma_45 * eta_455)

    def linearize(self, inputs, outputs, jacobian, discrete_inputs=None, discrete_outputs=None):

        total_pressure_45 = inputs["total_pressure_45"]
        total_pressure_5 = inputs["total_pressure_5"]
        total_temperature_45 = inputs["total_temperature_45"]
        total_temperature_5 = outputs["total_temperature_5"]
        gamma_45 = inputs["gamma_45"]
        eta_455 = inputs["eta_455"]

        jacobian["total_temperature_5", "total_temperature_5"] = np.diag(
            total_temperature_45
            / total_temperature_5 ** 2.0
            * (total_pressure_5 / total_pressure_45) ** ((gamma_45 - 1.0) / gamma_45 * eta_455)
        )
        jacobian["total_temperature_5", "total_temperature_45"] = -np.diag(
            (total_pressure_5 / total_pressure_45) ** ((gamma_45 - 1.0) / gamma_45 * eta_455)
            / total_temperature_5
        )
        jacobian["total_temperature_5", "total_pressure_5"] = np.diag(
            -total_temperature_45
            / total_temperature_5
            * (gamma_45 - 1.0)
            / gamma_45
            * eta_455
            * (total_pressure_5 / total_pressure_45)
            ** ((gamma_45 - 1.0) / gamma_45 * eta_455 - 1.0)
            / total_pressure_45
        )
        jacobian["total_temperature_5", "total_pressure_45"] = -np.diag(
            -total_temperature_45
            / total_temperature_5
            * (gamma_45 - 1.0)
            / gamma_45
            * eta_455
            * (total_pressure_5 / total_pressure_45)
            ** ((gamma_45 - 1.0) / gamma_45 * eta_455 - 1.0)
            * total_pressure_5
            / total_pressure_45 ** 2.0
        )
        jacobian["total_temperature_5", "gamma_45"] = np.diag(
            -total_temperature_45
            / total_temperature_5
            * np.log(total_pressure_5 / total_pressure_45)
            * (total_pressure_5 / total_pressure_45) ** ((gamma_45 - 1.0) / gamma_45 * eta_455)
            * eta_455
            / gamma_45 ** 2.0
        )
        jacobian["total_temperature_5", "eta_455"] = (
            -total_temperature_45
            / total_temperature_5
            * np.log(total_pressure_5 / total_pressure_45)
            * (total_pressure_5 / total_pressure_45) ** ((gamma_45 - 1.0) / gamma_45 * eta_455)
            * (gamma_45 - 1.0)
            / gamma_45
        )
