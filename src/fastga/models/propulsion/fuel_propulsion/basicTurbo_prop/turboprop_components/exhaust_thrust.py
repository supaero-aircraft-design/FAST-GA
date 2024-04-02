import numpy as np
import openmdao.api as om


class ExhaustThrust(om.ExplicitComponent):
    def initialize(self):

        self.options.declare("number_of_points", types=int, default=250)

    def setup(self):

        n = self.options["number_of_points"]

        self.add_input("air_mass_flow", units="kg/s", val=np.nan, shape=n)
        self.add_input("fuel_air_ratio", shape=n, val=np.nan)
        self.add_input("compressor_bleed_ratio", shape=n, val=np.nan)
        self.add_input("pressurization_bleed_ratio", shape=n, val=np.nan)

        self.add_input("velocity_8", units="m/s", shape=n, val=np.nan)
        self.add_input("mach_0", val=np.nan, shape=n)
        self.add_input("static_temperature_0", units="K", shape=n, val=np.nan)

        self.add_output("exhaust_thrust", units="N", shape=n, val=2e2)

        self.declare_partials(
            of="exhaust_thrust",
            wrt=["*"],
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        r_g = 287.0  # Perfect gas constant
        gamma = 1.4  # Gamma taken at its usual value

        air_mass_flow = inputs["air_mass_flow"]
        fuel_air_ratio = inputs["fuel_air_ratio"]
        compressor_bleed_ratio = inputs["compressor_bleed_ratio"]
        pressurization_bleed_ratio = inputs["pressurization_bleed_ratio"]

        velocity_8 = inputs["velocity_8"]
        static_temperature_0 = inputs["static_temperature_0"]
        mach_0 = inputs["mach_0"]

        exhaust_thrust = (
            air_mass_flow
            * (1.0 + fuel_air_ratio - pressurization_bleed_ratio - compressor_bleed_ratio)
            * (velocity_8 - mach_0 * np.sqrt(gamma * r_g * static_temperature_0))
        )

        outputs["exhaust_thrust"] = exhaust_thrust

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        r_g = 287.0  # Perfect gas constant
        gamma = 1.4  # Gamma taken at its usual value

        air_mass_flow = inputs["air_mass_flow"]
        fuel_air_ratio = inputs["fuel_air_ratio"]
        compressor_bleed_ratio = inputs["compressor_bleed_ratio"]
        pressurization_bleed_ratio = inputs["pressurization_bleed_ratio"]

        velocity_8 = inputs["velocity_8"]
        static_temperature_0 = inputs["static_temperature_0"]
        mach_0 = inputs["mach_0"]

        partials["exhaust_thrust", "air_mass_flow"] = np.diag(
            (1.0 + fuel_air_ratio - pressurization_bleed_ratio - compressor_bleed_ratio)
            * (velocity_8 - mach_0 * np.sqrt(gamma * r_g * static_temperature_0))
        )
        partials["exhaust_thrust", "fuel_air_ratio"] = np.diag(
            air_mass_flow * (velocity_8 - mach_0 * np.sqrt(gamma * r_g * static_temperature_0))
        )
        partials["exhaust_thrust", "compressor_bleed_ratio"] = -np.diag(
            air_mass_flow * (velocity_8 - mach_0 * np.sqrt(gamma * r_g * static_temperature_0))
        )
        partials["exhaust_thrust", "pressurization_bleed_ratio"] = -np.diag(
            air_mass_flow * (velocity_8 - mach_0 * np.sqrt(gamma * r_g * static_temperature_0))
        )
        partials["exhaust_thrust", "velocity_8"] = np.diag(
            air_mass_flow
            * (1.0 + fuel_air_ratio - pressurization_bleed_ratio - compressor_bleed_ratio)
        )
        partials["exhaust_thrust", "mach_0"] = -np.diag(
            air_mass_flow
            * (1.0 + fuel_air_ratio - pressurization_bleed_ratio - compressor_bleed_ratio)
            * np.sqrt(gamma * r_g * static_temperature_0)
        )
        partials["exhaust_thrust", "static_temperature_0"] = -np.diag(
            air_mass_flow
            * (1.0 + fuel_air_ratio - pressurization_bleed_ratio - compressor_bleed_ratio)
            * mach_0
            / 2.0
            * np.sqrt(gamma * r_g / static_temperature_0)
        )
