import numpy as np
import openmdao.api as om


class Station225Pressure(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("number_of_points", types=int, default=250)

    def setup(self):
        n = self.options["number_of_points"]

        self.add_input("total_temperature_2", units="K", shape=n, val=np.nan)
        self.add_input("total_temperature_25", units="K", shape=n, val=np.nan)
        self.add_input("total_pressure_2", units="Pa", shape=n, val=np.nan)
        self.add_input("gamma_2", shape=n, val=np.nan, units="unitless")
        self.add_input("eta_225", shape=1, val=1.0, units="unitless")

        self.add_output("total_pressure_25", units="Pa", shape=n, val=1e6)

    # pylint: disable=missing-function-docstring
    # Overriding OpenMDAO setup_partials
    def setup_partials(self):
        n = self.options["number_of_points"]

        self.declare_partials(
            of="total_pressure_25",
            wrt=["total_temperature_2", "total_temperature_25", "total_pressure_2", "gamma_2"],
            method="exact",
            rows=np.arange(n),
            cols=np.arange(n),
        )
        self.declare_partials(
            of="total_pressure_25",
            wrt="eta_225",
            method="exact",
            rows=np.arange(n),
            cols=np.zeros(n),
        )

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute, not all arguments are used
    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        total_temperature_2 = inputs["total_temperature_2"]
        total_temperature_25 = inputs["total_temperature_25"]
        total_pressure_2 = inputs["total_pressure_2"]

        gamma_2 = inputs["gamma_2"]
        eta_225 = inputs["eta_225"]

        total_pressure_25 = total_pressure_2 * (total_temperature_25 / total_temperature_2) ** (
            gamma_2 * eta_225 / (gamma_2 - 1.0)
        )

        outputs["total_pressure_25"] = total_pressure_25

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        total_temperature_2 = inputs["total_temperature_2"]
        total_temperature_25 = inputs["total_temperature_25"]
        total_pressure_2 = inputs["total_pressure_2"]

        gamma_2 = inputs["gamma_2"]
        eta_225 = inputs["eta_225"]

        partials["total_pressure_25", "total_pressure_2"] = (
            total_temperature_25 / total_temperature_2
        ) ** (gamma_2 * eta_225 / (gamma_2 - 1.0))
        partials["total_pressure_25", "total_temperature_25"] = (
            total_pressure_2
            * (gamma_2 * eta_225 / (gamma_2 - 1.0))
            * (total_temperature_25 / total_temperature_2)
            ** (gamma_2 * eta_225 / (gamma_2 - 1.0) - 1.0)
            / total_temperature_2
        )
        partials["total_pressure_25", "total_temperature_2"] = -(
            total_pressure_2
            * (gamma_2 * eta_225 / (gamma_2 - 1.0))
            * (total_temperature_25 / total_temperature_2)
            ** (gamma_2 * eta_225 / (gamma_2 - 1.0) - 1.0)
            * total_temperature_25
            / total_temperature_2**2.0
        )
        partials["total_pressure_25", "eta_225"] = (
            total_pressure_2
            * np.log(total_temperature_25 / total_temperature_2)
            * (total_temperature_25 / total_temperature_2) ** (gamma_2 * eta_225 / (gamma_2 - 1.0))
            * gamma_2
            / (gamma_2 - 1.0)
        )
        partials["total_pressure_25", "gamma_2"] = -(
            total_pressure_2
            * np.log(total_temperature_25 / total_temperature_2)
            * (total_temperature_25 / total_temperature_2) ** (gamma_2 * eta_225 / (gamma_2 - 1.0))
            * eta_225
            / (gamma_2 - 1.0) ** 2.0
        )


class Station225DesignPoint(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("number_of_points", types=int, default=250)

    def setup(self):
        n = self.options["number_of_points"]

        self.add_input("total_temperature_2", units="K", shape=n, val=np.nan)
        self.add_input("total_pressure_2", units="Pa", shape=n, val=np.nan)
        self.add_input("opr_1", shape=n, val=np.nan, units="unitless")
        self.add_input("gamma_2", shape=n, val=np.nan, units="unitless")
        self.add_input("eta_225", shape=1, val=1.0, units="unitless")

        self.add_output("total_temperature_25", units="K", shape=n, val=500.0)
        self.add_output("total_pressure_25", units="Pa", shape=n, val=1e6)

    # pylint: disable=missing-function-docstring
    # Overriding OpenMDAO setup_partials
    def setup_partials(self):
        n = self.options["number_of_points"]
        self.declare_partials(
            of="total_pressure_25",
            wrt=["total_pressure_2", "opr_1"],
            method="exact",
            rows=np.arange(n),
            cols=np.arange(n),
        )
        self.declare_partials(
            of="total_temperature_25",
            wrt=["total_temperature_2", "opr_1", "gamma_2"],
            method="exact",
            rows=np.arange(n),
            cols=np.arange(n),
        )
        self.declare_partials(
            of="total_temperature_25",
            wrt="eta_225",
            method="exact",
            rows=np.arange(n),
            cols=np.zeros(n),
        )

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute, not all arguments are used
    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        total_temperature_2 = inputs["total_temperature_2"]
        total_pressure_2 = inputs["total_pressure_2"]
        opr_1 = inputs["opr_1"]
        gamma_2 = inputs["gamma_2"]
        eta_225 = inputs["eta_225"]

        outputs["total_pressure_25"] = total_pressure_2 * opr_1
        outputs["total_temperature_25"] = total_temperature_2 * opr_1 ** (
            (gamma_2 - 1) / gamma_2 / eta_225
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        total_temperature_2 = inputs["total_temperature_2"]
        total_pressure_2 = inputs["total_pressure_2"]
        opr_1 = inputs["opr_1"]
        gamma_2 = inputs["gamma_2"]
        eta_225 = inputs["eta_225"]

        partials["total_pressure_25", "total_pressure_2"] = opr_1
        partials["total_pressure_25", "opr_1"] = total_pressure_2

        partials["total_temperature_25", "total_temperature_2"] = opr_1 ** (
            (gamma_2 - 1) / gamma_2 / eta_225
        )
        partials["total_temperature_25", "opr_1"] = (
            total_temperature_2
            * (gamma_2 - 1)
            / gamma_2
            / eta_225
            * opr_1 ** ((gamma_2 - 1) / gamma_2 / eta_225 - 1.0)
        )
        partials["total_temperature_25", "gamma_2"] = (
            total_temperature_2
            * opr_1 ** ((gamma_2 - 1) / gamma_2 / eta_225)
            * np.log(opr_1)
            / gamma_2**2.0
            / eta_225
        )
        partials["total_temperature_25", "eta_225"] = (
            -total_temperature_2
            * opr_1 ** ((gamma_2 - 1) / gamma_2 / eta_225)
            * np.log(opr_1)
            * (gamma_2 - 1)
            / gamma_2
            / eta_225**2.0
        )
