import numpy as np
import openmdao.api as om


class OverallPressureRatio(om.ExplicitComponent):
    def initialize(self):

        self.options.declare("number_of_points", types=int, default=250)

    def setup(self):

        n = self.options["number_of_points"]

        self.add_input("total_pressure_2", units="Pa", shape=n, val=np.nan)
        self.add_input("total_pressure_25", units="Pa", shape=n, val=np.nan)
        self.add_input("total_pressure_3", units="Pa", shape=n, val=np.nan)

        self.add_output("opr_1", shape=n)
        self.add_output("opr_2", shape=n)
        self.add_output("opr", shape=n, upper=12.0)

        self.declare_partials(
            of="opr_1", wrt=["total_pressure_25", "total_pressure_2"], method="exact"
        )
        self.declare_partials(
            of="opr_2", wrt=["total_pressure_3", "total_pressure_25"], method="exact"
        )
        self.declare_partials(
            of="opr", wrt=["total_pressure_3", "total_pressure_2"], method="exact"
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        outputs["opr_1"] = inputs["total_pressure_25"] / inputs["total_pressure_2"]
        outputs["opr_2"] = inputs["total_pressure_3"] / inputs["total_pressure_25"]
        outputs["opr"] = inputs["total_pressure_3"] / inputs["total_pressure_2"]
        # print("OPR", outputs["opr"])

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        partials["opr_1", "total_pressure_25"] = np.diag(1.0 / inputs["total_pressure_2"])
        partials["opr_1", "total_pressure_2"] = -np.diag(
            inputs["total_pressure_25"] / inputs["total_pressure_2"] ** 2.0
        )

        partials["opr_2", "total_pressure_3"] = np.diag(1.0 / inputs["total_pressure_25"])
        partials["opr_2", "total_pressure_25"] = -np.diag(
            inputs["total_pressure_3"] / inputs["total_pressure_25"] ** 2.0
        )

        partials["opr", "total_pressure_3"] = np.diag(1.0 / inputs["total_pressure_2"])
        partials["opr", "total_pressure_2"] = -np.diag(
            inputs["total_pressure_3"] / inputs["total_pressure_2"] ** 2.0
        )


class OverallPressureRatioDesignPoint(om.ExplicitComponent):
    def initialize(self):

        self.options.declare("number_of_points", types=int, default=250)

    def setup(self):

        n = self.options["number_of_points"]

        self.add_input(
            "settings:propulsion:turboprop:design_point:first_stage_pressure_ratio",
            val=0.25,
        )
        self.add_input(
            "data:propulsion:turboprop:design_point:OPR",
            shape=n,
            val=np.full(n, np.nan),
        )

        self.add_output("opr_1", shape=n)
        self.add_output("opr_2", shape=n)

        self.declare_partials(of="opr_1", wrt="*", method="exact")
        self.declare_partials(
            of="opr_2",
            wrt="settings:propulsion:turboprop:design_point:first_stage_pressure_ratio",
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        opr_design = inputs["data:propulsion:turboprop:design_point:OPR"]
        opr_ratio_design = inputs[
            "settings:propulsion:turboprop:design_point:first_stage_pressure_ratio"
        ]

        outputs["opr_1"] = opr_design * opr_ratio_design
        outputs["opr_2"] = 1.0 / opr_ratio_design

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        n = self.options["number_of_points"]

        opr_design = inputs["data:propulsion:turboprop:design_point:OPR"]
        opr_ratio_design = inputs[
            "settings:propulsion:turboprop:design_point:first_stage_pressure_ratio"
        ]

        partials["opr_1", "data:propulsion:turboprop:design_point:OPR"] = (
            np.eye(n) * opr_ratio_design
        )
        partials[
            "opr_1",
            "settings:propulsion:turboprop:design_point:first_stage_pressure_ratio",
        ] = opr_design

        partials[
            "opr_2",
            "settings:propulsion:turboprop:design_point:first_stage_pressure_ratio",
        ] = (
            -np.ones(n) / opr_ratio_design ** 2.0
        )
