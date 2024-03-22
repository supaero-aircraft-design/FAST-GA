import numpy as np
import openmdao.api as om
from stdatm import Atmosphere


class PressurizationAirFlow(om.ExplicitComponent):
    """
    Some classes used for the computation of the off-design point will have to be slightly
    altered for the design point because the name of their input is a sizing parameter. To reduce
    the amount of code necessary, when the formula for the output does not change, we will simply
    add an option instead or rewriting the component.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.input_name = "altitude"

    def initialize(self):

        self.options.declare("number_of_points", types=int, default=250)
        self.options.declare(
            "altitude_list",
            types=np.ndarray,
            default=np.linspace(0, 10000, 25),
        )
        self.options.declare(
            "cabin_altitude_list",
            types=np.ndarray,
            default=np.linspace(0, 3000, 25),
        )
        self.options.declare("design_point", types=bool, default=False)

    def setup(self):

        n = self.options["number_of_points"]

        if self.options["design_point"]:
            self.input_name = "data:propulsion:turboprop:design_point:altitude"

        self.add_input(self.input_name, units="m", shape=n, val=np.nan)
        self.add_input("data:geometry:cabin:volume", units="m**3", shape=1, val=np.nan)
        self.add_input("bleed_control", shape=n, val=1.0)
        self.add_input("cabin_air_renewal_time", units="s", shape=1, val=np.nan)

        self.add_output("pressurization_mass_flow", units="kg/s", val=0.045, shape=n)

        self.declare_partials("pressurization_mass_flow", self.input_name, method="fd", step=50)
        self.declare_partials(
            "pressurization_mass_flow",
            ["data:geometry:cabin:volume", "bleed_control", "cabin_air_renewal_time"],
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        cabin_altitude = np.interp(
            inputs[self.input_name] * 3.28084,
            self.options["altitude_list"],
            self.options["cabin_altitude_list"],
        )

        cabin_temperature = 273.0 + 20.0  # 20 degree Celsius inside the cabin

        atm = Atmosphere(altitude=cabin_altitude, altitude_in_feet=True, delta_t=20.0)
        cabin_air_density = atm.pressure / 287.0 / cabin_temperature

        renewal_time = inputs["cabin_air_renewal_time"]
        bleed_control = inputs["bleed_control"]
        cabin_volume = inputs["data:geometry:cabin:volume"]

        air_mass_flow_pressurization = (
            cabin_volume * cabin_air_density / renewal_time * (0.3 + 0.7 * bleed_control)
        )

        outputs["pressurization_mass_flow"] = air_mass_flow_pressurization

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        cabin_altitude = np.interp(
            inputs[self.input_name],
            self.options["altitude_list"],
            self.options["cabin_altitude_list"],
        )

        cabin_air_density = Atmosphere(altitude=cabin_altitude, altitude_in_feet=True).density

        renewal_time = inputs["cabin_air_renewal_time"]
        bleed_control = inputs["bleed_control"]
        cabin_volume = inputs["data:geometry:cabin:volume"]

        partials["pressurization_mass_flow", "data:geometry:cabin:volume"] = (
            cabin_air_density / renewal_time * (0.3 + 0.7 * bleed_control)
        )
        partials["pressurization_mass_flow", "bleed_control"] = np.diag(
            cabin_volume * cabin_air_density / renewal_time * 0.7
        )
        partials["pressurization_mass_flow", "cabin_air_renewal_time"] = -(
            cabin_volume * cabin_air_density / renewal_time ** 2.0 * (0.3 + 0.7 * bleed_control)
        )
