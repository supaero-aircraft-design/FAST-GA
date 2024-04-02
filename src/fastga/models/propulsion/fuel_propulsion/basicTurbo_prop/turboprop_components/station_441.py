import numpy as np
import openmdao.api as om


class Station441Temperature(om.ExplicitComponent):
    def initialize(self):

        self.options.declare("number_of_points", types=int, default=250)

    def setup(self):

        n = self.options["number_of_points"]

        self.add_input("total_temperature_41", units="K", shape=n, val=np.nan)
        self.add_input("total_temperature_3", units="K", shape=n, val=np.nan)
        self.add_input("fuel_air_ratio", shape=n, val=np.nan)
        self.add_input("compressor_bleed_ratio", shape=n, val=np.nan)
        self.add_input("cooling_bleed_ratio", shape=n, val=np.nan)
        self.add_input("pressurization_bleed_ratio", shape=n, val=np.nan)

        self.add_output("total_temperature_4", units="K", shape=n, val=1e3)

        self.declare_partials(
            of="total_temperature_4",
            wrt=[
                "total_temperature_3",
                "total_temperature_41",
                "fuel_air_ratio",
                "compressor_bleed_ratio",
                "cooling_bleed_ratio",
                "pressurization_bleed_ratio",
            ],
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        total_temperature_3 = inputs["total_temperature_3"]
        total_temperature_41 = inputs["total_temperature_41"]

        fuel_air_ratio = inputs["fuel_air_ratio"]
        compressor_bleed_ratio = inputs["compressor_bleed_ratio"]
        cooling_bleed_ratio = inputs["cooling_bleed_ratio"]
        pressurization_bleed_ratio = inputs["pressurization_bleed_ratio"]

        total_temperature_4 = (
            total_temperature_41
            * (1.0 + fuel_air_ratio - pressurization_bleed_ratio - compressor_bleed_ratio)
            - total_temperature_3 * cooling_bleed_ratio
        ) / (
            1.0
            + fuel_air_ratio
            - pressurization_bleed_ratio
            - cooling_bleed_ratio
            - compressor_bleed_ratio
        )

        outputs["total_temperature_4"] = total_temperature_4

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        total_temperature_3 = inputs["total_temperature_3"]
        total_temperature_41 = inputs["total_temperature_41"]

        fuel_air_ratio = inputs["fuel_air_ratio"]
        compressor_bleed_ratio = inputs["compressor_bleed_ratio"]
        cooling_bleed_ratio = inputs["cooling_bleed_ratio"]
        pressurization_bleed_ratio = inputs["pressurization_bleed_ratio"]

        partials["total_temperature_4", "total_temperature_41"] = np.diag(
            (1.0 + fuel_air_ratio - pressurization_bleed_ratio - compressor_bleed_ratio)
            / (
                1.0
                + fuel_air_ratio
                - pressurization_bleed_ratio
                - cooling_bleed_ratio
                - compressor_bleed_ratio
            )
        )
        partials["total_temperature_4", "total_temperature_3"] = -np.diag(
            cooling_bleed_ratio
            / (
                1.0
                + fuel_air_ratio
                - pressurization_bleed_ratio
                - cooling_bleed_ratio
                - compressor_bleed_ratio
            )
        )
        partials["total_temperature_4", "fuel_air_ratio"] = np.diag(
            (total_temperature_3 - total_temperature_41)
            * cooling_bleed_ratio
            / (
                1.0
                + fuel_air_ratio
                - pressurization_bleed_ratio
                - cooling_bleed_ratio
                - compressor_bleed_ratio
            )
            ** 2.0
        )
        partials["total_temperature_4", "pressurization_bleed_ratio"] = -np.diag(
            (total_temperature_3 - total_temperature_41)
            * cooling_bleed_ratio
            / (
                1.0
                + fuel_air_ratio
                - pressurization_bleed_ratio
                - cooling_bleed_ratio
                - compressor_bleed_ratio
            )
            ** 2.0
        )
        partials["total_temperature_4", "compressor_bleed_ratio"] = -np.diag(
            (total_temperature_3 - total_temperature_41)
            * cooling_bleed_ratio
            / (
                1.0
                + fuel_air_ratio
                - pressurization_bleed_ratio
                - cooling_bleed_ratio
                - compressor_bleed_ratio
            )
            ** 2.0
        )
        partials["total_temperature_4", "cooling_bleed_ratio"] = -np.diag(
            (1.0 + fuel_air_ratio - pressurization_bleed_ratio - compressor_bleed_ratio)
            * (total_temperature_3 - total_temperature_41)
            / (
                1.0
                + fuel_air_ratio
                - pressurization_bleed_ratio
                - cooling_bleed_ratio
                - compressor_bleed_ratio
            )
            ** 2.0
        )


class Station441PressureDesignPoint(om.ExplicitComponent):
    def initialize(self):

        self.options.declare("number_of_points", types=int, default=250)

    def setup(self):

        n = self.options["number_of_points"]

        self.add_input("total_pressure_41", units="Pa", shape=n, val=np.nan)

        self.add_output("total_pressure_4", units="Pa", shape=n, val=1e5)

        self.declare_partials(
            of="total_pressure_4",
            wrt="total_pressure_41",
            val=np.eye(n),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        outputs["total_pressure_4"] = inputs["total_pressure_41"]
