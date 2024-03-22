import numpy as np
import openmdao.api as om


class Station4145Temperature(om.ExplicitComponent):
    def initialize(self):

        self.options.declare("number_of_points", types=int, default=250)

    def setup(self):

        n = self.options["number_of_points"]

        self.add_input("total_temperature_41", units="K", shape=n, val=np.nan)
        self.add_input("data:propulsion:turboprop:design_point:alpha", shape=1, val=0.8)

        self.add_output("total_temperature_45", units="K", shape=n, val=1.2e3)

        self.declare_partials(
            of="total_temperature_45",
            wrt=[
                "total_temperature_41",
                "data:propulsion:turboprop:design_point:alpha",
            ],
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        outputs["total_temperature_45"] = (
            inputs["data:propulsion:turboprop:design_point:alpha"] * inputs["total_temperature_41"]
        )
        # print("ITT", outputs["total_temperature_45"])

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        n = self.options["number_of_points"]

        partials["total_temperature_45", "total_temperature_41"] = np.diag(
            np.full(n, inputs["data:propulsion:turboprop:design_point:alpha"])
        )
        partials["total_temperature_45", "data:propulsion:turboprop:design_point:alpha"] = inputs[
            "total_temperature_41"
        ]


class Station4145Pressure(om.ExplicitComponent):
    def initialize(self):

        self.options.declare("number_of_points", types=int, default=250)

    def setup(self):

        n = self.options["number_of_points"]

        self.add_input("total_pressure_41", units="Pa", shape=n, val=np.nan)
        self.add_input("data:propulsion:turboprop:design_point:alpha_p", shape=1, val=0.8)

        self.add_output("total_pressure_45", units="Pa", shape=n, val=1.2e3)

        self.declare_partials(
            of="total_pressure_45",
            wrt=[
                "total_pressure_41",
                "data:propulsion:turboprop:design_point:alpha_p",
            ],
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        outputs["total_pressure_45"] = (
            inputs["data:propulsion:turboprop:design_point:alpha_p"] * inputs["total_pressure_41"]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        n = self.options["number_of_points"]

        partials["total_pressure_45", "total_pressure_41"] = np.diag(
            np.full(n, inputs["data:propulsion:turboprop:design_point:alpha_p"])
        )
        partials["total_pressure_45", "data:propulsion:turboprop:design_point:alpha_p"] = inputs[
            "total_pressure_41"
        ]
