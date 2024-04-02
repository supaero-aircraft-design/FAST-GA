import numpy as np
import openmdao.api as om
from scipy.interpolate import RectBivariateSpline, interp1d
from stdatm import Atmosphere

THRUST_PTS_NB = 30
SPEED_PTS_NB = 10


class PropellerThrustRequired(om.ExplicitComponent):
    def initialize(self):

        self.options.declare("number_of_points", types=int, default=250)

    def setup(self):

        n = self.options["number_of_points"]

        self.add_input("required_thrust", units="N", shape=n, val=np.nan)
        self.add_input("exhaust_thrust", units="N", shape=n, val=np.nan)

        self.add_output("propeller_thrust", units="N", shape=n, val=2e3)

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        outputs["propeller_thrust"] = inputs["required_thrust"] - inputs["exhaust_thrust"]

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        n = self.options["number_of_points"]

        partials["propeller_thrust", "required_thrust"] = np.eye(n)
        partials["propeller_thrust", "exhaust_thrust"] = -np.eye(n)


class ShaftPowerRequired(om.ExplicitComponent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.sea_level_spline = None
        self.cruise_level_spline = None

    def initialize(self):

        self.options.declare("number_of_points", types=int, default=250)

    def setup(self):

        n = self.options["number_of_points"]

        self.add_input("altitude", units="m", shape=n, val=np.nan)
        self.add_input("mach_0", val=np.nan, shape=n)
        self.add_input("propeller_thrust", units="N", shape=n, val=np.nan)
        self.add_input("data:aerodynamics:propeller:cruise_level:altitude", np.nan, units="m")
        self.add_input(
            "data:aerodynamics:propeller:sea_level:speed",
            np.full(SPEED_PTS_NB, np.nan),
            units="m/s",
        )
        self.add_input(
            "data:aerodynamics:propeller:sea_level:thrust",
            np.full(THRUST_PTS_NB, np.nan),
            units="N",
        )
        self.add_input(
            "data:aerodynamics:propeller:sea_level:thrust_limit",
            np.full(SPEED_PTS_NB, np.nan),
            units="N",
        )
        self.add_input(
            "data:aerodynamics:propeller:sea_level:efficiency",
            np.full((SPEED_PTS_NB, THRUST_PTS_NB), np.nan),
        )
        self.add_input(
            "data:aerodynamics:propeller:cruise_level:speed",
            np.full(SPEED_PTS_NB, np.nan),
            units="m/s",
        )
        self.add_input(
            "data:aerodynamics:propeller:cruise_level:thrust",
            np.full(THRUST_PTS_NB, np.nan),
            units="N",
        )
        self.add_input(
            "data:aerodynamics:propeller:cruise_level:thrust_limit",
            np.full(SPEED_PTS_NB, np.nan),
            units="N",
        )
        self.add_input(
            "data:aerodynamics:propeller:cruise_level:efficiency",
            np.full((SPEED_PTS_NB, THRUST_PTS_NB), np.nan),
        )
        self.add_input(
            "data:aerodynamics:propeller:installation_effect:effective_efficiency:low_speed",
            val=1.0,
        )
        self.add_input(
            "data:aerodynamics:propeller:installation_effect:effective_efficiency:cruise",
            val=1.0,
        )
        self.add_input(
            "data:aerodynamics:propeller:installation_effect:effective_advance_ratio",
            val=1.0,
        )

        self.add_output("required_shaft_power", units="W", shape=n, val=500e3)

        self.declare_partials(
            of="required_shaft_power", wrt="altitude", method="fd", step=1.0, form="central"
        )
        self.declare_partials(
            of="required_shaft_power", wrt="mach_0", method="fd", step=1e-4, form="central"
        )
        self.declare_partials(
            of="required_shaft_power", wrt="propeller_thrust", method="fd", step=1.0, form="central"
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        atm = Atmosphere(inputs["altitude"], altitude_in_feet=False)
        true_airspeed = inputs["mach_0"] * atm.speed_of_sound

        thrust_sl = inputs["data:aerodynamics:propeller:sea_level:thrust"]
        thrust_limit_sl = inputs["data:aerodynamics:propeller:sea_level:thrust_limit"]
        speed_sl = inputs["data:aerodynamics:propeller:sea_level:speed"]
        efficiency_sl = inputs["data:aerodynamics:propeller:sea_level:efficiency"]

        thrust_cl = inputs["data:aerodynamics:propeller:cruise_level:thrust"]
        thrust_limit_cl = inputs["data:aerodynamics:propeller:cruise_level:thrust_limit"]
        speed_cl = inputs["data:aerodynamics:propeller:cruise_level:speed"]
        efficiency_cl = inputs["data:aerodynamics:propeller:cruise_level:efficiency"]

        if not self.sea_level_spline:
            self.sea_level_spline = RectBivariateSpline(
                thrust_sl,
                speed_sl,
                efficiency_sl.T
                * inputs[
                    "data:aerodynamics:propeller:installation_effect:effective_efficiency:low_speed"
                ],
            )
        if not self.cruise_level_spline:
            self.cruise_level_spline = RectBivariateSpline(
                thrust_cl,
                speed_cl,
                efficiency_cl.T
                * inputs[
                    "data:aerodynamics:propeller:installation_effect:effective_efficiency:cruise"
                ],
            )

        input_thrust = inputs["propeller_thrust"]

        thrust_interp_sl = np.clip(
            input_thrust,
            np.full_like(input_thrust, np.min(thrust_sl)),
            np.interp(true_airspeed, speed_sl, thrust_limit_sl),
        )
        thrust_interp_cl = np.clip(
            input_thrust,
            np.full_like(input_thrust, np.min(thrust_cl)),
            np.interp(true_airspeed, speed_cl, thrust_limit_cl),
        )

        # The input effective advance ratio is included here and not in the computation of the
        # true airspeed since it should not be used as part of the shaft power computation but
        # only as a displacement in the efficiency map reading
        lower_bound_efficiency = self.sea_level_spline(
            thrust_interp_sl,
            true_airspeed
            * inputs["data:aerodynamics:propeller:installation_effect:effective_advance_ratio"],
        )[0]
        upper_bound_efficiency = self.cruise_level_spline(
            thrust_interp_cl,
            true_airspeed
            * inputs["data:aerodynamics:propeller:installation_effect:effective_advance_ratio"],
        )[0]

        efficiency_propeller = (
            lower_bound_efficiency
            + (upper_bound_efficiency - lower_bound_efficiency)
            * inputs["altitude"]
            / inputs["data:aerodynamics:propeller:cruise_level:altitude"]
        )

        required_shaft_power = thrust_interp_sl * true_airspeed / efficiency_propeller
        outputs["required_shaft_power"] = required_shaft_power


class PropellerMaxThrust(om.ExplicitComponent):
    def initialize(self):

        self.options.declare("number_of_points", types=int, default=250)

    def setup(self):

        n = self.options["number_of_points"]

        self.add_input("altitude", units="m", shape=n, val=np.nan)
        self.add_input("mach_0", val=np.nan, shape=n)
        self.add_input("data:aerodynamics:propeller:cruise_level:altitude", np.nan, units="m")
        self.add_input(
            "data:aerodynamics:propeller:sea_level:speed",
            np.full(SPEED_PTS_NB, np.nan),
            units="m/s",
        )
        self.add_input(
            "data:aerodynamics:propeller:sea_level:thrust_limit",
            np.full(SPEED_PTS_NB, np.nan),
            units="N",
        )
        self.add_input(
            "data:aerodynamics:propeller:cruise_level:speed",
            np.full(SPEED_PTS_NB, np.nan),
            units="m/s",
        )
        self.add_input(
            "data:aerodynamics:propeller:cruise_level:thrust_limit",
            np.full(SPEED_PTS_NB, np.nan),
            units="N",
        )
        self.add_input(
            "data:aerodynamics:propeller:installation_effect:effective_advance_ratio",
            val=1.0,
        )

        self.add_output("propeller_max_thrust", units="N", shape=n, val=5000.0)

        self.declare_partials(
            of="propeller_max_thrust",
            wrt=["altitude", "mach_0"],
            method="fd",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        atm = Atmosphere(inputs["altitude"], altitude_in_feet=False)
        true_airspeed = inputs["mach_0"] * atm.speed_of_sound

        thrust_limit_sl = inputs["data:aerodynamics:propeller:sea_level:thrust_limit"]
        speed_sl = inputs["data:aerodynamics:propeller:sea_level:speed"]

        thrust_limit_cl = inputs["data:aerodynamics:propeller:cruise_level:thrust_limit"]
        speed_cl = inputs["data:aerodynamics:propeller:cruise_level:speed"]

        propeller_max_thrust_sl_func = interp1d(
            speed_sl, thrust_limit_sl, kind="cubic", bounds_error=False, fill_value="extrapolate"
        )
        propeller_max_thrust_cl_func = interp1d(
            speed_cl, thrust_limit_cl, kind="cubic", bounds_error=False, fill_value="extrapolate"
        )

        lower_bound_thrust_limit = propeller_max_thrust_sl_func(
            true_airspeed
            * inputs["data:aerodynamics:propeller:installation_effect:effective_advance_ratio"]
        )
        upper_bound_thrust_limit = propeller_max_thrust_cl_func(
            true_airspeed
            * inputs["data:aerodynamics:propeller:installation_effect:effective_advance_ratio"]
        )

        thrust_limit = (
            lower_bound_thrust_limit
            + (upper_bound_thrust_limit - lower_bound_thrust_limit)
            * np.minimum(
                inputs["altitude"],
                inputs["data:aerodynamics:propeller:cruise_level:altitude"],
            )
            / inputs["data:aerodynamics:propeller:cruise_level:altitude"]
        )

        outputs["propeller_max_thrust"] = thrust_limit
