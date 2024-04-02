import numpy as np
import openmdao.api as om


class ShaftPower(om.ExplicitComponent):
    def initialize(self):

        self.options.declare("number_of_points", types=int, default=250)

    def setup(self):

        n = self.options["number_of_points"]

        self.add_input("air_mass_flow", units="kg/s", val=np.nan, shape=n)
        self.add_input("fuel_air_ratio", shape=n, val=np.nan)
        self.add_input("compressor_bleed_ratio", shape=n, val=np.nan)
        self.add_input("pressurization_bleed_ratio", shape=n, val=np.nan)

        self.add_input("cp_45", shape=n, val=np.nan)
        self.add_input("total_temperature_45", units="K", shape=n, val=np.nan)
        self.add_input("cp_5", shape=n, val=np.nan)
        self.add_input("total_temperature_5", units="K", shape=n, val=np.nan)

        self.add_input(
            "settings:propulsion:turboprop:efficiency:gearbox",
            val=0.98,
        )

        self.add_output("shaft_power", units="W", shape=n, val=300e3)

        self.declare_partials(
            of="shaft_power",
            wrt=["*"],
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        air_mass_flow = inputs["air_mass_flow"]
        fuel_air_ratio = inputs["fuel_air_ratio"]
        compressor_bleed_ratio = inputs["compressor_bleed_ratio"]
        pressurization_bleed_ratio = inputs["pressurization_bleed_ratio"]

        cp_45 = inputs["cp_45"]
        total_temperature_45 = inputs["total_temperature_45"]
        cp_5 = inputs["cp_5"]
        total_temperature_5 = inputs["total_temperature_5"]

        gearbox_efficiency = inputs["settings:propulsion:turboprop:efficiency:gearbox"]

        shaft_power = (
            air_mass_flow
            * (1.0 + fuel_air_ratio - pressurization_bleed_ratio - compressor_bleed_ratio)
            * (cp_45 * total_temperature_45 - cp_5 * total_temperature_5)
            * gearbox_efficiency
        )

        outputs["shaft_power"] = shaft_power
        # print("shaft_power", outputs["shaft_power"])

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        air_mass_flow = inputs["air_mass_flow"]
        fuel_air_ratio = inputs["fuel_air_ratio"]
        compressor_bleed_ratio = inputs["compressor_bleed_ratio"]
        pressurization_bleed_ratio = inputs["pressurization_bleed_ratio"]

        cp_45 = inputs["cp_45"]
        total_temperature_45 = inputs["total_temperature_45"]
        cp_5 = inputs["cp_5"]
        total_temperature_5 = inputs["total_temperature_5"]

        gearbox_efficiency = inputs["settings:propulsion:turboprop:efficiency:gearbox"]

        partials["shaft_power", "air_mass_flow"] = np.diag(
            (1.0 + fuel_air_ratio - pressurization_bleed_ratio - compressor_bleed_ratio)
            * (cp_45 * total_temperature_45 - cp_5 * total_temperature_5)
            * gearbox_efficiency
        )
        partials["shaft_power", "fuel_air_ratio"] = np.diag(
            air_mass_flow
            * (cp_45 * total_temperature_45 - cp_5 * total_temperature_5)
            * gearbox_efficiency
        )
        partials["shaft_power", "compressor_bleed_ratio"] = -np.diag(
            air_mass_flow
            * (cp_45 * total_temperature_45 - cp_5 * total_temperature_5)
            * gearbox_efficiency
        )
        partials["shaft_power", "pressurization_bleed_ratio"] = -np.diag(
            air_mass_flow
            * (cp_45 * total_temperature_45 - cp_5 * total_temperature_5)
            * gearbox_efficiency
        )
        partials["shaft_power", "cp_45"] = np.diag(
            air_mass_flow
            * (1.0 + fuel_air_ratio - pressurization_bleed_ratio - compressor_bleed_ratio)
            * total_temperature_45
            * gearbox_efficiency
        )
        partials["shaft_power", "total_temperature_45"] = np.diag(
            air_mass_flow
            * (1.0 + fuel_air_ratio - pressurization_bleed_ratio - compressor_bleed_ratio)
            * cp_45
            * gearbox_efficiency
        )
        partials["shaft_power", "cp_5"] = -np.diag(
            air_mass_flow
            * (1.0 + fuel_air_ratio - pressurization_bleed_ratio - compressor_bleed_ratio)
            * total_temperature_5
            * gearbox_efficiency
        )
        partials["shaft_power", "total_temperature_5"] = -np.diag(
            air_mass_flow
            * (1.0 + fuel_air_ratio - pressurization_bleed_ratio - compressor_bleed_ratio)
            * cp_5
            * gearbox_efficiency
        )
        partials["shaft_power", "settings:propulsion:turboprop:efficiency:gearbox"] = (
            air_mass_flow
            * (1.0 + fuel_air_ratio - pressurization_bleed_ratio - compressor_bleed_ratio)
            * (cp_45 * total_temperature_45 - cp_5 * total_temperature_5)
        )
