"""Simple module for takeoff."""
#  This file is part of FAST : A framework for rapid Overall Aircraft Design
#  Copyright (C) 2020  ONERA & ISAE-SUPAERO
#  FAST is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.

import numpy as np
import math
import openmdao.api as om
import warnings
from scipy.constants import g
from typing import Union, List, Optional, Tuple

from fastoad.model_base import Atmosphere, FlightPoint
from fastoad.model_base.propulsion import FuelEngineSet

# noinspection PyProtectedMember
from fastoad.module_management._bundle_loader import BundleLoader
from fastoad.constants import EngineSetting

from fastga.models.propulsion.fuel_propulsion.base import FuelEngineSet

ALPHA_LIMIT = 13.5 * math.pi / 180.0  # Limit angle to touch tail on ground in rad
ALPHA_RATE = 3.0 * math.pi / 180.0  # Angular rotation speed in rad/s
SAFETY_HEIGHT = 50 * 0.3048  # Height in meters to reach V2 speed
TIME_STEP = 0.1  # For time dependent simulation


class TakeOffPhase(om.Group):
    def initialize(self):
        self.options.declare("propulsion_id", default="", types=str)

    def setup(self):

        self.add_subsystem(
            "compute_v2",
            _v2(propulsion_id=self.options["propulsion_id"]),
            promotes=self.get_io_names(
                _v2(propulsion_id=self.options["propulsion_id"]), iotypes="inputs"
            ),
        )
        self.add_subsystem(
            "compute_vloff",
            _vloff_from_v2(propulsion_id=self.options["propulsion_id"]),
            promotes=self.get_io_names(
                _vloff_from_v2(propulsion_id=self.options["propulsion_id"]),
                excludes=["v2:speed", "v2:angle",],
                iotypes="inputs",
            ),
        )
        self.add_subsystem(
            "compute_vr",
            _vr_from_v2(propulsion_id=self.options["propulsion_id"]),
            promotes=self.get_io_names(
                _vr_from_v2(propulsion_id=self.options["propulsion_id"]),
                excludes=["vloff:speed", "vloff:angle",],
                iotypes="inputs",
            ),
        )
        self.add_subsystem(
            "simulate_takeoff",
            _simulate_takeoff(propulsion_id=self.options["propulsion_id"]),
            promotes=self.get_io_names(
                _simulate_takeoff(propulsion_id=self.options["propulsion_id"]),
                excludes=["vr:speed", "v2:angle",],
            ),
        )
        self.connect("compute_v2.v2:speed", "compute_vloff.v2:speed")
        self.connect("compute_v2.v2:angle", "compute_vloff.v2:angle")
        self.connect("compute_vloff.vloff:speed", "compute_vr.vloff:speed")
        self.connect("compute_vloff.vloff:angle", "compute_vr.vloff:angle")
        self.connect("compute_vr.vr:speed", "simulate_takeoff.vr:speed")
        self.connect("compute_v2.v2:angle", "simulate_takeoff.v2:angle")

    @staticmethod
    def get_io_names(
        component: om.ExplicitComponent,
        excludes: Optional[Union[str, List[str]]] = None,
        iotypes: Optional[Union[str, Tuple[str]]] = ("inputs", "outputs"),
    ) -> List[str]:
        prob = om.Problem(model=component)
        prob.setup()
        data = []
        if type(iotypes) == tuple:
            data.extend(prob.model.list_inputs(out_stream=None))
            data.extend(prob.model.list_outputs(out_stream=None))
        else:
            if iotypes == "inputs":
                data.extend(prob.model.list_inputs(out_stream=None))
            else:
                data.extend(prob.model.list_outputs(out_stream=None))
        list_names = []
        for idx in range(len(data)):
            variable_name = data[idx][0]
            if excludes is None:
                list_names.append(variable_name)
            else:
                if variable_name not in list(excludes):
                    list_names.append(variable_name)

        return list_names


class _v2(om.ExplicitComponent):
    """
    Calculate V2 safety speed @ defined altitude considering a 30% safety margin on max lift capability (alpha imposed).
    Find corresponding climb rate margin for imposed thrust rate.
    Fuel burn is neglected : mass = MTOW.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._engine_wrapper = None

    def initialize(self):
        self.options.declare("propulsion_id", default="", types=str)

    def setup(self):
        self._engine_wrapper = BundleLoader().instantiate_component(self.options["propulsion_id"])
        self._engine_wrapper.setup(self)

        self.add_input("data:geometry:propulsion:count", np.nan)
        self.add_input("data:aerodynamics:wing:low_speed:CL_max_clean", np.nan)
        self.add_input("data:aerodynamics:wing:low_speed:CL0_clean", np.nan)
        self.add_input("data:aerodynamics:flaps:takeoff:CL", np.nan)
        self.add_input("data:aerodynamics:wing:low_speed:CL_alpha", np.nan, units="rad**-1")
        self.add_input("data:aerodynamics:aircraft:low_speed:CD0", np.nan)
        self.add_input("data:aerodynamics:flaps:takeoff:CD", np.nan)
        self.add_input("data:aerodynamics:wing:low_speed:induced_drag_coefficient", np.nan)
        self.add_input("data:geometry:wing:area", np.nan, units="m**2")
        self.add_input("data:geometry:wing:span", np.nan, units="m")
        self.add_input("data:geometry:landing_gear:height", np.nan, units="m")
        self.add_input("data:weight:aircraft:MTOW", np.nan, units="kg")

        self.add_output("v2:speed", units="m/s")
        self.add_output("v2:angle", units="rad")
        self.add_output("v2:climb_rate")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        propulsion_model = FuelEngineSet(
            self._engine_wrapper.get_model(inputs), inputs["data:geometry:propulsion:count"]
        )
        cl_max_clean = inputs["data:aerodynamics:wing:low_speed:CL_max_clean"]
        cl0 = (
            inputs["data:aerodynamics:wing:low_speed:CL0_clean"]
            + inputs["data:aerodynamics:flaps:takeoff:CL"]
        )
        cl_alpha = inputs["data:aerodynamics:wing:low_speed:CL_alpha"]
        cd0 = (
            inputs["data:aerodynamics:aircraft:low_speed:CD0"]
            + inputs["data:aerodynamics:flaps:takeoff:CD"]
        )
        coef_k = inputs["data:aerodynamics:wing:low_speed:induced_drag_coefficient"]
        wing_area = inputs["data:geometry:wing:area"]
        wing_span = inputs["data:geometry:wing:span"]
        lg_height = inputs["data:geometry:landing_gear:height"]
        mtow = inputs["data:weight:aircraft:MTOW"]

        # Define atmospheric condition for safety height
        atm = Atmosphere(SAFETY_HEIGHT, altitude_in_feet=False)
        # Define Cl considering 30% margin and estimate alpha
        cl = cl_max_clean / 1.2 ** 2  # V2>=1.2*VS1
        alpha_interp = np.linspace(0.0, 30.0, 31) * math.pi / 180.0
        cl_interp = cl0 + alpha_interp * cl_alpha
        alpha = np.interp(cl, cl_interp, alpha_interp)
        # Calculate drag coefficient
        k_ground = (
            33.0
            * ((lg_height + SAFETY_HEIGHT) / wing_span) ** 1.5
            / (1.0 + 33.0 * ((lg_height + SAFETY_HEIGHT) / wing_span) ** 1.5)
        )
        cd = cd0 + k_ground * coef_k * cl ** 2
        # Find v2 safety speed for 0% climb rate
        v2 = math.sqrt((mtow * g) / (0.5 * atm.density * wing_area * cl))
        # Estimate climb rate considering alpha~0° and max thrust rate for CS23.65 (loop on error)
        flight_point = FlightPoint(
            mach=v2 / atm.speed_of_sound,
            altitude=SAFETY_HEIGHT,
            engine_setting=EngineSetting.TAKEOFF,
            thrust_rate=1.0,
        )
        propulsion_model.compute_flight_points(flight_point)
        thrust = float(flight_point.thrust)
        gamma = math.asin(thrust / (mtow * g) - cd / cl)
        rel_error = 0.1
        while rel_error > 0.05:
            new_gamma = math.asin(thrust / (mtow * g) - cd / cl * math.cos(gamma))
            rel_error = abs((new_gamma - gamma) / new_gamma)
            gamma = new_gamma

        outputs["v2:speed"] = v2
        outputs["v2:angle"] = alpha
        outputs["v2:climb_rate"] = math.sin(gamma)


class _vloff_from_v2(om.ExplicitComponent):
    """
    Search alpha-angle<=alpha(v2) at which Vloff is operated such that
    aircraft reaches v>=v2 speed @ safety height with imposed rotation speed.
    Fuel burn is neglected : mass = MTOW.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._engine_wrapper = None

    def initialize(self):
        self.options.declare("propulsion_id", default="", types=str)

    def setup(self):
        self._engine_wrapper = BundleLoader().instantiate_component(self.options["propulsion_id"])
        self._engine_wrapper.setup(self)

        self.add_input("data:geometry:propulsion:count", np.nan)
        self.add_input("data:aerodynamics:wing:low_speed:CL0_clean", np.nan)
        self.add_input("data:aerodynamics:flaps:takeoff:CL", np.nan)
        self.add_input("data:aerodynamics:wing:low_speed:CL_alpha", np.nan, units="rad**-1")
        self.add_input("data:aerodynamics:aircraft:low_speed:CD0", np.nan)
        self.add_input("data:aerodynamics:flaps:takeoff:CD", np.nan)
        self.add_input("data:aerodynamics:wing:low_speed:induced_drag_coefficient", np.nan)
        self.add_input("data:geometry:wing:area", np.nan, units="m**2")
        self.add_input("data:geometry:wing:span", np.nan, units="m")
        self.add_input("data:geometry:landing_gear:height", np.nan, units="m")
        self.add_input("data:weight:aircraft:MTOW", np.nan, units="kg")
        self.add_input("data:mission:sizing:takeoff:thrust_rate", np.nan)
        self.add_input("v2:speed", np.nan, units="m/s")
        self.add_input("v2:angle", np.nan, units="rad")

        self.add_output("vloff:speed", units="m/s")
        self.add_output("vloff:angle", units="rad")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        propulsion_model = FuelEngineSet(
            self._engine_wrapper.get_model(inputs), inputs["data:geometry:propulsion:count"]
        )
        cl0 = (
            inputs["data:aerodynamics:wing:low_speed:CL0_clean"]
            + inputs["data:aerodynamics:flaps:takeoff:CL"]
        )
        cl_alpha = inputs["data:aerodynamics:wing:low_speed:CL_alpha"]
        cd0 = (
            inputs["data:aerodynamics:aircraft:low_speed:CD0"]
            + inputs["data:aerodynamics:flaps:takeoff:CD"]
        )
        coef_k = inputs["data:aerodynamics:wing:low_speed:induced_drag_coefficient"]
        wing_area = inputs["data:geometry:wing:area"]
        wing_span = inputs["data:geometry:wing:span"]
        lg_height = inputs["data:geometry:landing_gear:height"]
        mtow = inputs["data:weight:aircraft:MTOW"]
        thrust_rate = inputs["data:mission:sizing:takeoff:thrust_rate"]
        v2_target = float(inputs["v2:speed"])
        alpha_v2 = float(inputs["v2:angle"])

        # Define ground factor effect on Drag
        k_ground = lambda altitude: (
            33.0
            * ((lg_height + altitude) / wing_span) ** 1.5
            / (1.0 + 33.0 * ((lg_height + altitude) / wing_span) ** 1.5)
        )
        # Calculate v2 speed @ safety height for different alpha lift-off
        alpha = np.linspace(0.0, min(ALPHA_LIMIT, alpha_v2), num=10)
        vloff = np.zeros(np.size(alpha))
        v2 = np.zeros(np.size(alpha))
        atm_0 = Atmosphere(0.0)
        for i in range(len(alpha)):
            # Calculate lift coefficient
            cl = cl0 + cl_alpha * alpha[i]
            # Loop on estimated lift-off speed error induced by thrust estimation
            rel_error = 0.1
            vloff[i] = math.sqrt((mtow * g) / (0.5 * atm_0.density * wing_area * cl))
            while rel_error > 0.05:
                # Update thrust with vloff
                flight_point = FlightPoint(
                    mach=vloff[i] / atm_0.speed_of_sound,
                    altitude=0.0,
                    engine_setting=EngineSetting.TAKEOFF,
                    thrust_rate=thrust_rate,
                )
                propulsion_model.compute_flight_points(flight_point)
                thrust = float(flight_point.thrust)
                # Calculate vloff necessary to overcome weight
                if thrust * math.sin(alpha[i]) > mtow * g:
                    break
                else:
                    v = math.sqrt(
                        (mtow * g - thrust * math.sin(alpha[i]))
                        / (0.5 * atm_0.density * wing_area * cl)
                    )
                rel_error = abs(v - vloff[i]) / v
                vloff[i] = v
            # Perform climb with imposed rotational speed till reaching safety height
            alpha_t = alpha[i]
            gamma_t = 0.0
            v_t = float(vloff[i])
            altitude_t = 0.0
            distance_t = 0.0
            while altitude_t < SAFETY_HEIGHT:
                # Estimation of thrust
                atm = Atmosphere(altitude_t, altitude_in_feet=False)
                flight_point = FlightPoint(
                    mach=v_t / atm.speed_of_sound,
                    altitude=altitude_t,
                    engine_setting=EngineSetting.TAKEOFF,
                    thrust_rate=thrust_rate,
                )
                propulsion_model.compute_flight_points(flight_point)
                thrust = float(flight_point.thrust)
                # Calculate lift and drag
                cl = cl0 + cl_alpha * alpha_t
                lift = 0.5 * atm.density * wing_area * cl * v_t ** 2
                cd = cd0 + k_ground(altitude_t) * coef_k * cl ** 2
                drag = 0.5 * atm.density * wing_area * cd * v_t ** 2
                # Calculate acceleration on x/z air axis
                weight = mtow * g
                acc_x = (thrust * math.cos(alpha_t) - weight * math.sin(gamma_t) - drag) / mtow
                acc_z = (lift + thrust * math.sin(alpha_t) - weight * math.cos(gamma_t)) / mtow
                # Calculate gamma change and new speed
                delta_gamma = math.atan((acc_z * TIME_STEP) / (v_t + acc_x * TIME_STEP))
                v_t_new = math.sqrt((acc_z * TIME_STEP) ** 2 + (v_t + acc_x * TIME_STEP) ** 2)
                # Trapezoidal integration on distance/altitude
                delta_altitude = (
                    (v_t_new * math.sin(gamma_t + delta_gamma) + v_t * math.sin(gamma_t))
                    / 2
                    * TIME_STEP
                )
                delta_distance = (
                    (v_t_new * math.cos(gamma_t + delta_gamma) + v_t * math.cos(gamma_t))
                    / 2
                    * TIME_STEP
                )
                # Update temporal values
                alpha_t = min(alpha_v2, alpha_t + ALPHA_RATE * TIME_STEP)
                gamma_t = gamma_t + delta_gamma
                altitude_t = altitude_t + delta_altitude
                distance_t = distance_t + delta_distance
                v_t = v_t_new
            # Save obtained v2
            v2[i] = v_t
        # If v2 target speed not reachable maximum lift-off speed chosen (alpha=0°)
        if sum(v2 > v2_target) == 0:
            alpha = 0.0
            vloff = vloff[0]  # FIXME: not reachable v2
            warnings.warn("V2 @ 50ft requirement not reachable with max lift-off speed!")
        else:
            # If max alpha angle lead to v2 > v2 target take it
            if v2[-1] > v2_target:
                alpha = alpha[-1]
                vloff = vloff[-1]
            else:
                alpha = np.interp(v2_target, v2, alpha)
                vloff = np.interp(v2_target, v2, vloff)

        outputs["vloff:speed"] = vloff
        outputs["vloff:angle"] = alpha


class _vr_from_v2(om.ExplicitComponent):
    """
    Search VR for given lift-off conditions by doing reverted simulation.
    The error introduced comes from acceleration acc(t)~acc(t+dt) => v(t-dt)~V(t)-acc(t)*dt.
    Time step has been reduced by 1/5 to limit integration error.

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._engine_wrapper = None

    def initialize(self):
        self.options.declare("propulsion_id", default="", types=str)

    def setup(self):
        self._engine_wrapper = BundleLoader().instantiate_component(self.options["propulsion_id"])
        self._engine_wrapper.setup(self)

        self.add_input("data:geometry:propulsion:count", np.nan)
        self.add_input("data:aerodynamics:wing:low_speed:CL0_clean", np.nan)
        self.add_input("data:aerodynamics:flaps:takeoff:CL", np.nan)
        self.add_input("data:aerodynamics:wing:low_speed:CL_alpha", np.nan, units="rad**-1")
        self.add_input("data:aerodynamics:aircraft:low_speed:CD0", np.nan)
        self.add_input("data:aerodynamics:flaps:takeoff:CD", np.nan)
        self.add_input("data:aerodynamics:wing:low_speed:induced_drag_coefficient", np.nan)
        self.add_input("data:geometry:wing:area", np.nan, units="m**2")
        self.add_input("data:geometry:wing:span", np.nan, units="m")
        self.add_input("data:geometry:landing_gear:height", np.nan, units="m")
        self.add_input("data:weight:aircraft:MTOW", np.nan, units="kg")
        self.add_input("data:mission:sizing:takeoff:thrust_rate", np.nan)
        self.add_input("data:mission:sizing:takeoff:friction_coefficient_no_brake", np.nan)
        self.add_input("vloff:speed", np.nan, units="m/s")
        self.add_input("vloff:angle", np.nan, units="rad")

        self.add_output("vr:speed", units="m/s")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        propulsion_model = FuelEngineSet(
            self._engine_wrapper.get_model(inputs), inputs["data:geometry:propulsion:count"]
        )
        cl0 = (
            inputs["data:aerodynamics:wing:low_speed:CL0_clean"]
            + inputs["data:aerodynamics:flaps:takeoff:CL"]
        )
        cl_alpha = inputs["data:aerodynamics:wing:low_speed:CL_alpha"]
        cd0 = (
            inputs["data:aerodynamics:aircraft:low_speed:CD0"]
            + inputs["data:aerodynamics:flaps:takeoff:CD"]
        )
        coef_k = inputs["data:aerodynamics:wing:low_speed:induced_drag_coefficient"]
        wing_area = inputs["data:geometry:wing:area"]
        wing_span = inputs["data:geometry:wing:span"]
        lg_height = inputs["data:geometry:landing_gear:height"]
        mtow = inputs["data:weight:aircraft:MTOW"]
        thrust_rate = inputs["data:mission:sizing:takeoff:thrust_rate"]
        friction_coeff = inputs["data:mission:sizing:takeoff:friction_coefficient_no_brake"]
        v_t = float(inputs["vloff:speed"])
        alpha_t = float(inputs["vloff:angle"])

        # Define ground factor effect on Drag
        k_ground = (
            33.0 * (lg_height / wing_span) ** 1.5 / (1.0 + 33.0 * (lg_height / wing_span) ** 1.5)
        )
        # Start reverted calculation of flight from lift-off to 0° alpha angle
        atm = Atmosphere(0.0)
        while (alpha_t != 0.0) and (v_t != 0.0):
            # Estimation of thrust
            flight_point = FlightPoint(
                mach=v_t / atm.speed_of_sound,
                altitude=0.0,
                engine_setting=EngineSetting.TAKEOFF,
                thrust_rate=thrust_rate,
            )
            propulsion_model.compute_flight_points(flight_point)
            thrust = float(flight_point.thrust)
            # Calculate lift and drag
            cl = cl0 + cl_alpha * alpha_t
            lift = 0.5 * atm.density * wing_area * cl * v_t ** 2
            cd = cd0 + k_ground * coef_k * cl ** 2
            drag = 0.5 * atm.density * wing_area * cd * v_t ** 2
            # Calculate rolling resistance load
            friction = (mtow * g - lift - thrust * math.sin(alpha_t)) * friction_coeff
            # Calculate acceleration
            acc_x = (thrust * math.cos(alpha_t) - drag - friction) / mtow
            # Speed and angle update (feedback)
            dt = min(TIME_STEP / 5, alpha_t / ALPHA_RATE, v_t / acc_x)
            v_t = v_t - acc_x * dt
            alpha_t = alpha_t - ALPHA_RATE * dt

        outputs["vr:speed"] = v_t


class _simulate_takeoff(om.ExplicitComponent):
    """
    Simulate take-off from 0m/s speed to safety height using input VR.
    Fuel burn is supposed negligible : mass = MTOW.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._engine_wrapper = None

    def initialize(self):
        self.options.declare("propulsion_id", default="", types=str)

    def setup(self):
        self._engine_wrapper = BundleLoader().instantiate_component(self.options["propulsion_id"])
        self._engine_wrapper.setup(self)

        self.add_input("data:geometry:propulsion:count", np.nan)
        self.add_input("data:aerodynamics:wing:low_speed:CL_max_clean", np.nan)
        self.add_input("data:aerodynamics:wing:low_speed:CL0_clean", np.nan)
        self.add_input("data:aerodynamics:flaps:takeoff:CL", np.nan)
        self.add_input("data:aerodynamics:wing:low_speed:CL_alpha", np.nan, units="rad**-1")
        self.add_input("data:aerodynamics:aircraft:low_speed:CD0", np.nan)
        self.add_input("data:aerodynamics:flaps:takeoff:CD", np.nan)
        self.add_input("data:aerodynamics:wing:low_speed:induced_drag_coefficient", np.nan)
        self.add_input("data:geometry:wing:area", np.nan, units="m**2")
        self.add_input("data:geometry:wing:span", np.nan, units="m")
        self.add_input("data:geometry:landing_gear:height", np.nan, units="m")
        self.add_input("data:weight:aircraft:MTOW", np.nan, units="kg")
        self.add_input("data:mission:sizing:takeoff:thrust_rate", np.nan)
        self.add_input("data:mission:sizing:takeoff:friction_coefficient_no_brake", np.nan)
        self.add_input("vr:speed", np.nan, units="m/s")
        self.add_input("v2:angle", np.nan, units="rad")

        self.add_output("data:mission:sizing:takeoff:VR", units="m/s")
        self.add_output("data:mission:sizing:takeoff:VLOF", units="m/s")
        self.add_output("data:mission:sizing:takeoff:V2", units="m/s")
        self.add_output("data:mission:sizing:takeoff:TOFL", units="m")
        self.add_output("data:mission:sizing:takeoff:duration", units="s")
        self.add_output("data:mission:sizing:takeoff:fuel", units="kg")
        self.add_output("data:mission:sizing:initial_climb:fuel", units="kg")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        propulsion_model = FuelEngineSet(
            self._engine_wrapper.get_model(inputs), inputs["data:geometry:propulsion:count"]
        )
        cl_max_clean = inputs["data:aerodynamics:wing:low_speed:CL_max_clean"]
        cl0 = (
            inputs["data:aerodynamics:wing:low_speed:CL0_clean"]
            + inputs["data:aerodynamics:flaps:takeoff:CL"]
        )
        cl_alpha = inputs["data:aerodynamics:wing:low_speed:CL_alpha"]
        cd0 = (
            inputs["data:aerodynamics:aircraft:low_speed:CD0"]
            + inputs["data:aerodynamics:flaps:takeoff:CD"]
        )
        coef_k = inputs["data:aerodynamics:wing:low_speed:induced_drag_coefficient"]
        wing_area = inputs["data:geometry:wing:area"]
        wing_span = inputs["data:geometry:wing:span"]
        lg_height = inputs["data:geometry:landing_gear:height"]
        mtow = inputs["data:weight:aircraft:MTOW"]
        thrust_rate = inputs["data:mission:sizing:takeoff:thrust_rate"]
        friction_coeff = inputs["data:mission:sizing:takeoff:friction_coefficient_no_brake"]
        alpha_v2 = float(inputs["v2:angle"])

        # Define ground factor effect on Drag
        k_ground = lambda altitude: (
            33.0
            * ((lg_height + altitude) / wing_span) ** 1.5
            / (1.0 + 33.0 * ((lg_height + altitude) / wing_span) ** 1.5)
        )
        # Determine rotation speed from regulation CS23.51
        vs1 = math.sqrt((mtow * g) / (0.5 * Atmosphere(0).density * wing_area * cl_max_clean))
        if inputs["data:geometry:propulsion:count"] == 1.0:
            k = 1.0
        else:
            k = 1.1
        vr = max(k * vs1, float(inputs["vr:speed"]))
        # Start calculation of flight from null speed to 35ft high
        alpha_t = 0.0
        gamma_t = 0.0
        v_t = 0.0
        altitude_t = 0.0
        distance_t = 0.0
        mass_fuel1_t = 0.0
        mass_fuel2_t = 0.0
        time_t = 0.0
        vloff = 0.0
        climb = False
        while altitude_t < SAFETY_HEIGHT:
            # Estimation of thrust
            atm = Atmosphere(altitude_t, altitude_in_feet=False)
            flight_point = FlightPoint(
                mach=max(v_t, vr) / atm.speed_of_sound,
                altitude=altitude_t,
                engine_setting=EngineSetting.TAKEOFF,
                thrust_rate=thrust_rate,
            )
            # FIXME: (speed increased to vr to have feasible consumptions)
            propulsion_model.compute_flight_points(flight_point)
            thrust = float(flight_point.thrust)
            # Calculate lift and drag
            cl = cl0 + cl_alpha * alpha_t
            lift = 0.5 * atm.density * wing_area * cl * v_t ** 2
            cd = cd0 + k_ground(altitude_t) * coef_k * cl ** 2
            drag = 0.5 * atm.density * wing_area * cd * v_t ** 2
            # Check if lift-off condition reached
            if (
                (lift + thrust * math.sin(alpha_t) - mtow * g * math.cos(gamma_t)) >= 0.0
            ) and not climb:
                climb = True
                vloff = v_t
            # Calculate acceleration on x/z air axis
            if climb:
                acc_z = (lift + thrust * math.sin(alpha_t) - mtow * g * math.cos(gamma_t)) / mtow
                acc_x = (thrust * math.cos(alpha_t) - mtow * g * math.sin(gamma_t) - drag) / mtow
            else:
                friction = (mtow * g - lift - thrust * math.sin(alpha_t)) * friction_coeff
                acc_z = 0.0
                acc_x = (thrust * math.cos(alpha_t) - drag - friction) / mtow
            # Calculate gamma change and new speed
            delta_gamma = math.atan((acc_z * TIME_STEP) / (v_t + acc_x * TIME_STEP))
            v_t_new = math.sqrt((acc_z * TIME_STEP) ** 2 + (v_t + acc_x * TIME_STEP) ** 2)
            # Trapezoidal integration on distance/altitude
            delta_altitude = (
                (v_t_new * math.sin(gamma_t + delta_gamma) + v_t * math.sin(gamma_t))
                / 2
                * TIME_STEP
            )
            delta_distance = (
                (v_t_new * math.cos(gamma_t + delta_gamma) + v_t * math.cos(gamma_t))
                / 2
                * TIME_STEP
            )
            # Update temporal values
            if v_t >= vr:
                alpha_t = min(alpha_v2, alpha_t + ALPHA_RATE * TIME_STEP)
            gamma_t = gamma_t + delta_gamma
            altitude_t = altitude_t + delta_altitude
            if not climb:
                mass_fuel1_t += propulsion_model.get_consumed_mass(flight_point, TIME_STEP)
                distance_t = distance_t + delta_distance
                time_t = time_t + TIME_STEP
            else:
                mass_fuel2_t += propulsion_model.get_consumed_mass(flight_point, TIME_STEP)
                time_t = time_t + TIME_STEP
            v_t = v_t_new

        outputs["data:mission:sizing:takeoff:VR"] = vr
        outputs["data:mission:sizing:takeoff:VLOF"] = vloff
        outputs["data:mission:sizing:takeoff:V2"] = v_t
        outputs["data:mission:sizing:takeoff:TOFL"] = distance_t
        outputs["data:mission:sizing:takeoff:duration"] = time_t
        outputs["data:mission:sizing:takeoff:fuel"] = mass_fuel1_t
        outputs["data:mission:sizing:initial_climb:fuel"] = mass_fuel2_t
