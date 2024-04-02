import numpy as np
import openmdao.api as om


class MassFlow(om.ExplicitComponent):
    def initialize(self):

        self.options.declare("number_of_points", types=int, default=250)

    def setup(self):

        n = self.options["number_of_points"]

        self.add_input("air_mass_flow", units="kg/s", val=np.nan, shape=n)
        self.add_input("fuel_mass_flow", units="kg/s", val=np.nan, shape=n)
        self.add_input("compressor_bleed_mass_flow", units="kg/s", val=np.nan, shape=n)
        self.add_input("pressurization_mass_flow", units="kg/s", val=np.nan, shape=n)

        self.add_output("fuel_air_ratio", shape=n)
        self.add_output("compressor_bleed_ratio", shape=n)
        self.add_output("pressurization_bleed_ratio", shape=n)

        self.declare_partials(
            of="fuel_air_ratio", wrt=["air_mass_flow", "fuel_mass_flow"], method="exact"
        )
        self.declare_partials(
            of="compressor_bleed_ratio",
            wrt=["air_mass_flow", "compressor_bleed_mass_flow"],
            method="exact",
        )
        self.declare_partials(
            of="pressurization_bleed_ratio",
            wrt=["air_mass_flow", "pressurization_mass_flow"],
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        outputs["fuel_air_ratio"] = inputs["fuel_mass_flow"] / inputs["air_mass_flow"]
        outputs["compressor_bleed_ratio"] = (
            inputs["compressor_bleed_mass_flow"] / inputs["air_mass_flow"]
        )
        outputs["pressurization_bleed_ratio"] = (
            inputs["pressurization_mass_flow"] / inputs["air_mass_flow"]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        partials["fuel_air_ratio", "fuel_mass_flow"] = np.diag(1.0 / inputs["air_mass_flow"])
        partials["fuel_air_ratio", "air_mass_flow"] = -np.diag(
            inputs["fuel_mass_flow"] / inputs["air_mass_flow"] ** 2.0
        )

        partials["compressor_bleed_ratio", "compressor_bleed_mass_flow"] = np.diag(
            1.0 / inputs["air_mass_flow"]
        )
        partials["compressor_bleed_ratio", "air_mass_flow"] = -np.diag(
            inputs["compressor_bleed_mass_flow"] / inputs["air_mass_flow"] ** 2.0
        )

        partials["pressurization_bleed_ratio", "pressurization_mass_flow"] = np.diag(
            1.0 / inputs["air_mass_flow"]
        )
        partials["pressurization_bleed_ratio", "air_mass_flow"] = -np.diag(
            inputs["pressurization_mass_flow"] / inputs["air_mass_flow"] ** 2.0
        )
