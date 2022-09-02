"""Estimation of horizontal tail area."""
#  This file is part of FAST-OAD_CS23 : A framework for rapid Overall Aircraft Design
#  Copyright (C) 2022  ONERA & ISAE-SUPAERO
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

from scipy.constants import g
from typing import Union, List, Optional, Tuple

# noinspection PyProtectedMember
from fastoad.module_management._bundle_loader import BundleLoader
import fastoad.api as oad
from fastoad.constants import EngineSetting

from stdatm import Atmosphere

from fastga.command.api import list_inputs, list_outputs

from .constants import SUBMODEL_HT_AREA

_ANG_VEL = 12 * math.pi / 180  # 12 deg/s (typical for light aircraft)


@oad.RegisterSubmodel(
    SUBMODEL_HT_AREA, "fastga.submodel.handling_qualities.horizontal_tail.area.legacy"
)
class UpdateHTArea(om.Group):
    """
    Computes needed ht area to:
      - have enough rotational power during take-off phase.
      - have enough rotational power during landing phase.
    """

    def initialize(self):
        self.options.declare("propulsion_id", default="", types=str)

    def setup(self):

        self.add_subsystem(
            "aero_coeff_landing",
            _ComputeAeroCoeff(landing=True),
            promotes=self.get_io_names(_ComputeAeroCoeff(landing=True), iotypes="inputs"),
        )
        self.add_subsystem(
            "aero_coeff_takeoff",
            _ComputeAeroCoeff(),
            promotes=self.get_io_names(_ComputeAeroCoeff(), iotypes="inputs"),
        )
        self.add_subsystem(
            "ht_area",
            _UpdateArea(propulsion_id=self.options["propulsion_id"]),
            promotes=self.get_io_names(
                _UpdateArea(propulsion_id=self.options["propulsion_id"]),
                excludes=[
                    "landing:cl_htp",
                    "takeoff:cl_htp",
                    "low_speed:cl_alpha_htp_isolated",
                ],
            ),
        )
        self.add_subsystem(
            "ht_area_constraints",
            _ComputeHTPAreaConstraints(propulsion_id=self.options["propulsion_id"]),
            promotes=self.get_io_names(
                _ComputeHTPAreaConstraints(propulsion_id=self.options["propulsion_id"]),
                excludes=[
                    "landing:cl_htp",
                    "takeoff:cl_htp",
                    "low_speed:cl_alpha_htp_isolated",
                ],
            ),
        )

        self.connect("aero_coeff_landing.cl_htp", "ht_area.landing:cl_htp")
        self.connect("aero_coeff_takeoff.cl_htp", "ht_area.takeoff:cl_htp")
        self.connect(
            "aero_coeff_takeoff.cl_alpha_htp_isolated", "ht_area.low_speed:cl_alpha_htp_isolated"
        )

        self.connect("aero_coeff_landing.cl_htp", "ht_area_constraints.landing:cl_htp")
        self.connect("aero_coeff_takeoff.cl_htp", "ht_area_constraints.takeoff:cl_htp")
        self.connect(
            "aero_coeff_takeoff.cl_alpha_htp_isolated",
            "ht_area_constraints.low_speed:cl_alpha_htp_isolated",
        )

    @staticmethod
    def get_io_names(
        component: om.ExplicitComponent,
        excludes: Optional[Union[str, List[str]]] = None,
        iotypes: Optional[Union[str, Tuple[str, str]]] = ("inputs", "outputs"),
    ) -> List[str]:

        list_names = []
        if isinstance(iotypes, tuple):
            list_names.extend(list_inputs(component))
            list_names.extend(list_outputs(component))
        else:
            if iotypes == "inputs":
                list_names.extend(list_inputs(component))
            else:
                list_names.extend(list_outputs(component))
        if excludes is not None:
            list_names = [x for x in list_names if x not in excludes]

        return list_names


class HTPConstraints(om.ExplicitComponent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._engine_wrapper = None

    def initialize(self):
        self.options.declare("propulsion_id", default="", types=str)

    def takeoff_rotation(self, inputs):

        n_engines = inputs["data:geometry:propulsion:engine:count"]
        wing_area = inputs["data:geometry:wing:area"]
        wing_mac = inputs["data:geometry:wing:MAC:length"]
        lp_ht = inputs["data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25"]

        cg_range = inputs["settings:weight:aircraft:CG:range"]

        takeoff_t_rate = inputs["data:mission:sizing:takeoff:thrust_rate"]

        x_wing_aero_center = inputs["data:geometry:wing:MAC:at25percent:x"]

        mtow = inputs["data:weight:aircraft:MTOW"]
        x_cg_aft = inputs["data:weight:aircraft:CG:aft:x"]
        x_lg = inputs["data:weight:airframe:landing_gear:main:CG:x"]
        z_cg_aircraft = inputs["data:weight:aircraft_empty:CG:z"]
        z_cg_engine = inputs["data:weight:propulsion:engine:CG:z"]

        cl0_clean = inputs["data:aerodynamics:wing:low_speed:CL0_clean"]
        cl_max_clean = inputs["data:aerodynamics:wing:low_speed:CL_max_clean"]
        cl_max_takeoff = inputs["data:aerodynamics:aircraft:takeoff:CL_max"]
        cl_flaps_takeoff = inputs["data:aerodynamics:flaps:takeoff:CL"]
        tail_efficiency_factor = inputs["data:aerodynamics:horizontal_tail:efficiency"]
        cl_htp_takeoff = inputs["takeoff:cl_htp"]
        cm_takeoff = (
            inputs["data:aerodynamics:wing:low_speed:CM0_clean"]
            + inputs["data:aerodynamics:flaps:takeoff:CM"]
        )
        cl_alpha_htp_isolated = inputs["low_speed:cl_alpha_htp_isolated"]

        z_eng = z_cg_aircraft - z_cg_engine

        # Conditions for calculation
        atm = Atmosphere(0.0)
        rho = atm.density

        propulsion_model = self._engine_wrapper.get_model(inputs)

        # Calculation of take-off minimum speed
        weight = mtow * g
        vs0 = math.sqrt(weight / (0.5 * rho * wing_area * cl_max_takeoff))
        vs1 = math.sqrt(weight / (0.5 * rho * wing_area * cl_max_clean))
        # Rotation speed requirement from FAR 23.51 (depends on number of engines)
        if n_engines == 1:
            v_r = vs1 * 1.0
        else:
            v_r = vs1 * 1.1
        # Definition of max forward gravity center position
        x_cg = x_cg_aft - cg_range * wing_mac
        # Definition of horizontal tail global position
        x_ht = x_wing_aero_center + lp_ht
        # Calculation of wheel factor
        flight_point = oad.FlightPoint(
            mach=v_r / atm.speed_of_sound,
            altitude=0.0,
            engine_setting=EngineSetting.TAKEOFF,
            thrust_rate=takeoff_t_rate,
        )
        propulsion_model.compute_flight_points(flight_point)
        thrust = float(flight_point.thrust)
        fact_wheel = (
            (x_lg - x_cg - z_eng * thrust / weight) / wing_mac * (vs0 / v_r) ** 2
        )  # FIXME: not clear if vs0 or vs1 should be used in formula
        # Compute aerodynamic coefficients for takeoff @ 0° aircraft angle
        cl0_takeoff = cl0_clean + cl_flaps_takeoff
        # Calculation of correction coefficient n_h and n_q
        n_h = (
            (x_ht - x_lg) / lp_ht * tail_efficiency_factor
        )  # tail_efficiency_factor: dynamic pressure reduction at
        # tail (typical value)
        n_q = 1 + cl_alpha_htp_isolated / cl_htp_takeoff * _ANG_VEL * (x_ht - x_lg) / v_r
        # Calculation of volume coefficient based on Torenbeek formula
        coeff_vol = (
            cl_max_takeoff
            / (n_h * n_q * cl_htp_takeoff)
            * (cm_takeoff / cl_max_takeoff - fact_wheel)
            + cl0_takeoff / cl_htp_takeoff * (x_lg - x_wing_aero_center) / wing_mac
        )
        # Calculation of equivalent area
        area = coeff_vol * wing_area * wing_mac / lp_ht

        return area

    def landing(self, inputs):

        propulsion_model = self._engine_wrapper.get_model(inputs)

        wing_area = inputs["data:geometry:wing:area"]
        x_wing_aero_center = inputs["data:geometry:wing:MAC:at25percent:x"]
        lp_ht = inputs["data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25"]
        wing_mac = inputs["data:geometry:wing:MAC:length"]

        mlw = inputs["data:weight:aircraft:MLW"]
        cg_range = inputs["settings:weight:aircraft:CG:range"]
        x_cg_aft = inputs["data:weight:aircraft:CG:aft:x"]
        z_cg_aircraft = inputs["data:weight:aircraft_empty:CG:z"]
        z_cg_engine = inputs["data:weight:propulsion:engine:CG:z"]
        x_lg = inputs["data:weight:airframe:landing_gear:main:CG:x"]

        cl0_clean = inputs["data:aerodynamics:wing:low_speed:CL0_clean"]
        cl_max_landing = inputs["data:aerodynamics:aircraft:landing:CL_max"]
        cl_flaps_landing = inputs["data:aerodynamics:flaps:landing:CL"]
        tail_efficiency_factor = inputs["data:aerodynamics:horizontal_tail:efficiency"]
        cl_htp_landing = inputs["landing:cl_htp"]
        cm_landing = (
            inputs["data:aerodynamics:wing:low_speed:CM0_clean"]
            + inputs["data:aerodynamics:flaps:landing:CM"]
        )
        cl_alpha_htp_isolated = inputs["low_speed:cl_alpha_htp_isolated"]

        z_eng = z_cg_aircraft - z_cg_engine

        # Conditions for calculation
        atm = Atmosphere(0.0)
        rho = atm.density

        # Definition of max forward gravity center position
        x_cg = x_cg_aft - cg_range * wing_mac
        # Definition of horizontal tail global position
        x_ht = x_wing_aero_center + lp_ht

        # Calculation of take-off minimum speed
        weight = mlw * g
        vs0 = math.sqrt(weight / (0.5 * rho * wing_area * cl_max_landing))
        # Rotation speed requirement from FAR 23.73
        v_r = vs0 * 1.3
        # Calculation of wheel factor
        flight_point = oad.FlightPoint(
            mach=v_r / atm.speed_of_sound,
            altitude=0.0,
            engine_setting=EngineSetting.IDLE,
            thrust_rate=0.1,
        )  # FIXME: fixed thrust rate (should depend on wished descent rate)
        propulsion_model.compute_flight_points(flight_point)
        thrust = float(flight_point.thrust)
        fact_wheel = (
            (x_lg - x_cg - z_eng * thrust / weight) / wing_mac * (vs0 / v_r) ** 2
        )  # FIXME: not clear if vs0 or vs1 should be used in formula
        # Evaluate aircraft overall angle (aoa)
        cl0_landing = cl0_clean + cl_flaps_landing
        # Calculation of correction coefficient n_h and n_q
        n_h = (
            (x_ht - x_lg) / lp_ht * tail_efficiency_factor
        )  # tail_efficiency_factor: dynamic pressure reduction at
        # tail (typical value)
        n_q = 1 + cl_alpha_htp_isolated / cl_htp_landing * _ANG_VEL * (x_ht - x_lg) / v_r
        # Calculation of volume coefficient based on Torenbeek formula
        coeff_vol = (
            cl_max_landing
            / (n_h * n_q * cl_htp_landing)
            * (cm_landing / cl_max_landing - fact_wheel)
            + cl0_landing / cl_htp_landing * (x_lg - x_wing_aero_center) / wing_mac
        )
        # Calculation of equivalent area
        area = coeff_vol * wing_area * wing_mac / lp_ht

        return area


class _UpdateArea(HTPConstraints):
    """Computes area of horizontal tail plane (internal function)."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._engine_wrapper = None

    def initialize(self):
        self.options.declare("propulsion_id", default="", types=str)

    def setup(self):
        self._engine_wrapper = BundleLoader().instantiate_component(self.options["propulsion_id"])
        self._engine_wrapper.setup(self)

        self.add_input("settings:weight:aircraft:CG:range", val=0.3)
        self.add_input("data:mission:sizing:takeoff:thrust_rate", val=np.nan)
        self.add_input("data:geometry:wing:area", val=np.nan, units="m**2")
        self.add_input("data:geometry:wing:MAC:at25percent:x", val=np.nan, units="m")
        self.add_input(
            "data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25", val=np.nan, units="m"
        )
        self.add_input("data:geometry:wing:MAC:length", val=np.nan, units="m")
        self.add_input("data:geometry:propulsion:nacelle:height", val=np.nan, units="m")
        self.add_input("data:weight:aircraft:MTOW", val=np.nan, units="kg")
        self.add_input("data:weight:aircraft:MLW", val=np.nan, units="kg")
        self.add_input("data:weight:aircraft:CG:aft:x", val=np.nan, units="m")
        self.add_input("data:weight:airframe:landing_gear:main:CG:x", val=np.nan, units="m")
        self.add_input("data:weight:aircraft_empty:CG:z", val=np.nan, units="m")
        self.add_input("data:weight:propulsion:engine:CG:z", val=np.nan, units="m")
        self.add_input("data:aerodynamics:wing:low_speed:CL0_clean", val=np.nan)
        self.add_input("data:aerodynamics:wing:low_speed:CM0_clean", val=np.nan)
        self.add_input("data:aerodynamics:aircraft:landing:CL_max", val=np.nan)
        self.add_input("data:aerodynamics:aircraft:takeoff:CL_max", val=np.nan)
        self.add_input("data:aerodynamics:wing:low_speed:CL_max_clean", val=np.nan)
        self.add_input("data:aerodynamics:flaps:landing:CL", val=np.nan)
        self.add_input("data:aerodynamics:flaps:takeoff:CL", val=np.nan)
        self.add_input("data:aerodynamics:flaps:landing:CM", val=np.nan)
        self.add_input("data:aerodynamics:flaps:takeoff:CM", val=np.nan)
        self.add_input(
            "data:aerodynamics:horizontal_tail:low_speed:CL_alpha", val=np.nan, units="rad**-1"
        )
        self.add_input("data:aerodynamics:horizontal_tail:efficiency", val=np.nan)

        self.add_input("landing:cl_htp", val=np.nan)
        self.add_input("takeoff:cl_htp", val=np.nan)
        self.add_input("low_speed:cl_alpha_htp_isolated", val=np.nan)

        self.add_output("data:geometry:horizontal_tail:area", val=4.0, units="m**2")

        self.declare_partials(
            "*", "*", method="fd"
        )  # FIXME: write partial avoiding discrete parameters

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        # Sizing constraints for the horizontal tail (methods from Torenbeek).
        # Limiting cases: Rotating power at takeoff/landing, with the most
        # forward CG position. Returns maximum area.

        # CASE1: TAKE-OFF ##########################################################################
        # method extracted from Torenbeek 1982 p325

        # Calculation of take-off minimum speed
        area_1 = self.takeoff_rotation(inputs)

        # CASE2: LANDING ###########################################################################
        # method extracted from Torenbeek 1982 p325

        # Calculation of equivalent area
        area_2 = self.landing(inputs)

        if max(area_1, area_2) < 0.0:
            print("Warning: HTP area estimated negative (in ComputeHTArea) forced to 1m²!")
            outputs["data:geometry:horizontal_tail:area"] = 1.0
        else:
            outputs["data:geometry:horizontal_tail:area"] = max(area_1, area_2)


class _ComputeHTPAreaConstraints(HTPConstraints):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._engine_wrapper = None

    def initialize(self):
        self.options.declare("propulsion_id", default="", types=str)

    def setup(self):
        self._engine_wrapper = BundleLoader().instantiate_component(self.options["propulsion_id"])
        self._engine_wrapper.setup(self)

        self.add_input("settings:weight:aircraft:CG:range", val=0.3)
        self.add_input("data:mission:sizing:takeoff:thrust_rate", val=np.nan)
        self.add_input("data:geometry:wing:area", val=np.nan, units="m**2")
        self.add_input("data:geometry:wing:MAC:at25percent:x", val=np.nan, units="m")
        self.add_input(
            "data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25", val=np.nan, units="m"
        )
        self.add_input("data:geometry:horizontal_tail:area", val=np.nan, units="m**2")
        self.add_input("data:geometry:wing:MAC:length", val=np.nan, units="m")
        self.add_input("data:geometry:propulsion:nacelle:height", val=np.nan, units="m")
        self.add_input("data:weight:aircraft:MTOW", val=np.nan, units="kg")
        self.add_input("data:weight:aircraft:MLW", val=np.nan, units="kg")
        self.add_input("data:weight:aircraft:CG:aft:x", val=np.nan, units="m")
        self.add_input("data:weight:airframe:landing_gear:main:CG:x", val=np.nan, units="m")
        self.add_input("data:weight:aircraft_empty:CG:z", val=np.nan, units="m")
        self.add_input("data:weight:propulsion:engine:CG:z", val=np.nan, units="m")
        self.add_input("data:aerodynamics:wing:low_speed:CL0_clean", val=np.nan)
        self.add_input("data:aerodynamics:wing:low_speed:CM0_clean", val=np.nan)
        self.add_input("data:aerodynamics:aircraft:landing:CL_max", val=np.nan)
        self.add_input("data:aerodynamics:aircraft:takeoff:CL_max", val=np.nan)
        self.add_input("data:aerodynamics:wing:low_speed:CL_max_clean", val=np.nan)
        self.add_input("data:aerodynamics:flaps:landing:CL", val=np.nan)
        self.add_input("data:aerodynamics:flaps:takeoff:CL", val=np.nan)
        self.add_input("data:aerodynamics:flaps:landing:CM", val=np.nan)
        self.add_input("data:aerodynamics:flaps:takeoff:CM", val=np.nan)
        self.add_input(
            "data:aerodynamics:horizontal_tail:low_speed:CL_alpha", val=np.nan, units="rad**-1"
        )
        self.add_input("data:aerodynamics:horizontal_tail:efficiency", val=np.nan)

        self.add_input("landing:cl_htp", val=np.nan)
        self.add_input("takeoff:cl_htp", val=np.nan)
        self.add_input("low_speed:cl_alpha_htp_isolated", val=np.nan)

        self.add_output("data:constraints:horizontal_tail:takeoff_rotation", units="m**2")
        self.add_output("data:constraints:horizontal_tail:landing", units="m**2")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        # Sizing constraints margin for the horizontal tail (methods from Torenbeek).
        # Limiting cases: Rotating power at takeoff/landing, with the most
        # forward CG position. Returns maximum area.

        area_htp = inputs["data:geometry:horizontal_tail:area"]

        # CASE1: TAKE-OFF ##########################################################################
        # method extracted from Torenbeek 1982 p325

        # Calculation of take-off minimum speed
        area_diff_1 = area_htp - self.takeoff_rotation(inputs)

        # CASE2: LANDING ###########################################################################
        # method extracted from Torenbeek 1982 p325

        # Calculation of equivalent area
        area_diff_2 = area_htp - self.landing(inputs)

        outputs["data:constraints:horizontal_tail:takeoff_rotation"] = area_diff_1
        outputs["data:constraints:horizontal_tail:landing"] = area_diff_2


class _ComputeAeroCoeff(om.ExplicitComponent):
    """
    Adapts aero-coefficients (reference surface is tail area for cl_ht)
    """

    def initialize(self):
        self.options.declare("landing", default=False, types=bool)

    def setup(self):

        self.add_input("data:geometry:wing:area", val=np.nan, units="m**2")
        self.add_input("data:geometry:horizontal_tail:area", val=2.0, units="m**2")
        self.add_input("data:weight:aircraft:MLW", val=np.nan, units="kg")
        self.add_input("data:weight:aircraft:MTOW", val=np.nan, units="kg")
        self.add_input("data:aerodynamics:horizontal_tail:low_speed:CL0", val=np.nan)
        self.add_input(
            "data:aerodynamics:horizontal_tail:low_speed:CL_alpha_isolated",
            val=np.nan,
            units="rad**-1",
        )
        self.add_input(
            "data:aerodynamics:horizontal_tail:low_speed:CL_alpha", val=np.nan, units="rad**-1"
        )
        self.add_input("data:aerodynamics:wing:low_speed:CL0_clean", val=np.nan)
        self.add_input("data:aerodynamics:wing:low_speed:CL_alpha", val=np.nan, units="rad**-1")
        self.add_input("data:aerodynamics:flaps:landing:CL", val=np.nan)
        self.add_input("data:aerodynamics:flaps:takeoff:CL", val=np.nan)
        self.add_input("data:aerodynamics:aircraft:landing:CL_max", val=np.nan)
        self.add_input("data:aerodynamics:wing:low_speed:CL_max_clean", val=np.nan)
        self.add_input("data:aerodynamics:elevator:low_speed:CL_delta", val=np.nan, units="rad**-1")
        self.add_input("data:mission:sizing:landing:elevator_angle", val=np.nan, units="rad")
        self.add_input("data:mission:sizing:takeoff:elevator_angle", val=np.nan, units="rad")

        self.add_output("cl_htp")
        self.add_output("cl_alpha_htp_isolated")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        wing_area = inputs["data:geometry:wing:area"]
        ht_area = inputs["data:geometry:horizontal_tail:area"]
        mlw = inputs["data:weight:aircraft:MLW"]
        cl0_htp = inputs["data:aerodynamics:horizontal_tail:low_speed:CL0"]
        cl_alpha_htp_isolated = inputs[
            "data:aerodynamics:horizontal_tail:low_speed:CL_alpha_isolated"
        ]
        cl_alpha_htp = inputs["data:aerodynamics:horizontal_tail:low_speed:CL_alpha"]
        cl0_clean_wing = inputs["data:aerodynamics:wing:low_speed:CL0_clean"]
        cl_alpha_wing = inputs["data:aerodynamics:wing:low_speed:CL_alpha"]
        cl_flaps_landing = inputs["data:aerodynamics:flaps:landing:CL"]
        cl_max_landing = inputs["data:aerodynamics:aircraft:landing:CL_max"]
        cl_delta_elev = inputs["data:aerodynamics:elevator:low_speed:CL_delta"]

        # Conditions for calculation
        atm = Atmosphere(0.0)
        rho = atm.density

        # Calculate elevator max. additional lift
        if self.options["landing"]:
            elev_angle = inputs["data:mission:sizing:landing:elevator_angle"]
        else:
            elev_angle = inputs["data:mission:sizing:takeoff:elevator_angle"]
        cl_elev = cl_delta_elev * elev_angle
        # Define alpha angle depending on phase
        if self.options["landing"]:
            # Calculation of take-off minimum speed
            weight = mlw * g
            vs0 = math.sqrt(weight / (0.5 * rho * wing_area * cl_max_landing))
            # Rotation speed correction
            v_r = vs0 * 1.3
            # Evaluate aircraft overall angle (aoa)
            cl0_landing = cl0_clean_wing + cl_flaps_landing
            cl_landing = weight / (0.5 * rho * v_r ** 2 * wing_area)
            alpha = (cl_landing - cl0_landing) / cl_alpha_wing * 180 / math.pi
        else:
            # Define aircraft overall angle (aoa)
            alpha = 0.0
        # Interpolate cl/cm and define with ht reference surface
        cl_htp = (cl0_htp + (alpha * math.pi / 180) * cl_alpha_htp + cl_elev) * wing_area / ht_area
        # Define Cl_alpha with htp reference surface
        cl_alpha_htp_isolated = cl_alpha_htp_isolated * wing_area / ht_area

        outputs["cl_htp"] = cl_htp
        outputs["cl_alpha_htp_isolated"] = cl_alpha_htp_isolated

    @staticmethod
    def _extrapolate(x, xp, yp) -> float:
        """
        Extrapolate linearly out of range x-value
        """
        if (x >= xp[0]) and (x <= xp[-1]):
            result = float(np.interp(x, xp, yp))
        elif x < xp[0]:
            result = float(yp[0] + (x - xp[0]) * (yp[1] - yp[0]) / (xp[1] - xp[0]))
        else:
            result = float(yp[-1] + (x - xp[-1]) * (yp[-1] - yp[-2]) / (xp[-1] - xp[-2]))

        if result is None:
            result = np.array([np.nan])

        return result
