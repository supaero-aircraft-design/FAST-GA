import numpy as np
import openmdao.api as om


class Station253Pressure(om.ExplicitComponent):
    def initialize(self):

        self.options.declare("number_of_points", types=int, default=250)

    def setup(self):

        n = self.options["number_of_points"]

        self.add_input("total_pressure_25", units="Pa", shape=n, val=np.nan)
        self.add_input("total_pressure_2", units="Pa", shape=n, val=np.nan)
        self.add_input("data:propulsion:turboprop:design_point:opr_2_opr_1", val=np.nan)

        self.add_output("total_pressure_3", units="Pa", shape=n, val=1e6)

        self.declare_partials(
            of="total_pressure_3",
            wrt=[
                "total_pressure_25",
                "total_pressure_2",
                "data:propulsion:turboprop:design_point:opr_2_opr_1",
            ],
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        total_pressure_25 = inputs["total_pressure_25"]
        total_pressure_2 = inputs["total_pressure_2"]

        opr_ratio = inputs["data:propulsion:turboprop:design_point:opr_2_opr_1"]

        total_pressure_3 = total_pressure_25 * opr_ratio * total_pressure_25 / total_pressure_2

        outputs["total_pressure_3"] = total_pressure_3

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        total_pressure_25 = inputs["total_pressure_25"]
        total_pressure_2 = inputs["total_pressure_2"]

        opr_ratio = inputs["data:propulsion:turboprop:design_point:opr_2_opr_1"]

        partials["total_pressure_3", "total_pressure_25"] = np.diag(
            2.0 * total_pressure_25 / total_pressure_2 * opr_ratio
        )
        partials["total_pressure_3", "total_pressure_2"] = np.diag(
            -((total_pressure_25 / total_pressure_2) ** 2.0) * opr_ratio
        )
        partials["total_pressure_3", "data:propulsion:turboprop:design_point:opr_2_opr_1"] = (
            total_pressure_25 ** 2.0 / total_pressure_2
        )


class Station253Temperature(om.ExplicitComponent):
    def initialize(self):

        self.options.declare("number_of_points", types=int, default=250)

    def setup(self):

        n = self.options["number_of_points"]

        self.add_input("total_pressure_3", units="Pa", shape=n, val=np.nan)
        self.add_input("total_pressure_25", units="Pa", shape=n, val=np.nan)
        self.add_input("total_temperature_25", units="K", shape=n, val=np.nan)
        self.add_input("gamma_25", shape=n, val=np.nan)
        self.add_input("eta_253", shape=1, val=1.0)

        self.add_output("total_temperature_3", units="K", shape=n, val=0.5e3)

        self.declare_partials(
            of="total_temperature_3",
            wrt=[
                "total_pressure_3",
                "total_pressure_25",
                "total_temperature_25",
                "gamma_25",
                "eta_253",
            ],
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        total_temperature_25 = inputs["total_temperature_25"]
        total_pressure_3 = inputs["total_pressure_3"]
        total_pressure_25 = inputs["total_pressure_25"]
        gamma_25 = inputs["gamma_25"]

        eta_253 = inputs["eta_253"]

        total_temperature_3 = total_temperature_25 * (total_pressure_3 / total_pressure_25) ** (
            (gamma_25 - 1) / (gamma_25 * eta_253)
        )

        outputs["total_temperature_3"] = total_temperature_3

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        total_temperature_25 = inputs["total_temperature_25"]
        total_pressure_3 = inputs["total_pressure_3"]
        total_pressure_25 = inputs["total_pressure_25"]
        gamma_25 = inputs["gamma_25"]

        eta_253 = inputs["eta_253"]

        partials["total_temperature_3", "total_temperature_25"] = np.diag(
            (total_pressure_3 / total_pressure_25) ** ((gamma_25 - 1) / (gamma_25 * eta_253))
        )
        partials["total_temperature_3", "total_pressure_3"] = np.diag(
            total_temperature_25
            * ((gamma_25 - 1) / (gamma_25 * eta_253))
            * (total_pressure_3 / total_pressure_25)
            ** ((gamma_25 - 1) / (gamma_25 * eta_253) - 1.0)
            / total_pressure_25
        )
        partials["total_temperature_3", "total_pressure_25"] = -np.diag(
            total_temperature_25
            * ((gamma_25 - 1) / (gamma_25 * eta_253))
            * (total_pressure_3 / total_pressure_25)
            ** ((gamma_25 - 1) / (gamma_25 * eta_253) - 1.0)
            * total_pressure_3
            / total_pressure_25 ** 2.0
        )
        partials["total_temperature_3", "gamma_25"] = np.diag(
            total_temperature_25
            * np.log(total_pressure_3 / total_pressure_25)
            * (total_pressure_3 / total_pressure_25) ** ((gamma_25 - 1) / (gamma_25 * eta_253))
            * 1.0
            / (gamma_25 ** 2.0 * eta_253)
        )
        partials["total_temperature_3", "eta_253"] = -(
            total_temperature_25
            * np.log(total_pressure_3 / total_pressure_25)
            * (total_pressure_3 / total_pressure_25) ** ((gamma_25 - 1) / (gamma_25 * eta_253))
            * (gamma_25 - 1.0)
            / (gamma_25 * eta_253 ** 2.0)
        )


class Station253PressureDesignPoint(om.ExplicitComponent):
    def initialize(self):

        self.options.declare("number_of_points", types=int, default=250)

    def setup(self):

        n = self.options["number_of_points"]

        self.add_input("total_pressure_25", units="Pa", shape=n, val=np.nan)
        self.add_input("opr_2", shape=n, val=np.nan)

        self.add_output("total_pressure_3", units="Pa", shape=n, val=1e6)

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        total_pressure_25 = inputs["total_pressure_25"]
        opr_2 = inputs["opr_2"]

        outputs["total_pressure_3"] = total_pressure_25 * opr_2

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        total_pressure_25 = inputs["total_pressure_25"]
        opr_2 = inputs["opr_2"]

        partials["total_pressure_3", "total_pressure_25"] = np.diag(opr_2)
        partials["total_pressure_3", "opr_2"] = np.diag(total_pressure_25)
