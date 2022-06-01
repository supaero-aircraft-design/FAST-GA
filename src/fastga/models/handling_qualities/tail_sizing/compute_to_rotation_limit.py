"""Estimation of the position of the CG that limits takeoff rotation."""
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

_ANG_VEL = 12 * math.pi / 180  # 12 deg/s (typical for light aircraft)


class ComputeTORotationLimitGroup(om.Group):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._engine_wrapper = None

    def initialize(self):
        self.options.declare("propulsion_id", default="", types=str)

    def setup(self):
        self.add_subsystem(
            "aero_coeff_to",
            _ComputeAeroCoeffTO(),
            promotes=self.get_io_names(_ComputeAeroCoeffTO(), iotypes="inputs"),
        )
        self.add_subsystem(
            "to_rotation_limit",
            ComputeTORotationLimit(propulsion_id=self.options["propulsion_id"]),
            promotes=self.get_io_names(
                ComputeTORotationLimit(propulsion_id=self.options["propulsion_id"]),
                excludes=[
                    "takeoff:cl_htp",
                    "takeoff:cm_wing",
                    "low_speed:cl_alpha_htp",
                ],
            ),
        )
        self.connect("aero_coeff_to.cl_htp", "to_rotation_limit.takeoff:cl_htp")
        self.connect("aero_coeff_to.cm_wing", "to_rotation_limit.takeoff:cm_wing")
        self.connect("aero_coeff_to.cl_alpha_htp", "to_rotation_limit.low_speed:cl_alpha_htp")

    @staticmethod
    def get_io_names(
        component: om.ExplicitComponent,
        excludes: Optional[Union[str, List[str]]] = None,
        iotypes: Optional[Union[str, Tuple[str]]] = ("inputs", "outputs"),
    ) -> List[str]:
        prob = om.Problem(model=component)
        prob.setup()
        data = []
        if isinstance(iotypes, tuple):
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


class ComputeTORotationLimit(om.ExplicitComponent):
    """
    Computes area of horizontal tail plane (internal function).
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._engine_wrapper = None

    def initialize(self):
        self.options.declare("propulsion_id", default="", types=str)

    def setup(self):
        self._engine_wrapper = BundleLoader().instantiate_component(self.options["propulsion_id"])
        self._engine_wrapper.setup(self)

        self.add_input("data:geometry:wing:area", val=np.nan, units="m**2")
        self.add_input("data:geometry:wing:MAC:length", val=np.nan, units="m")
        self.add_input("data:geometry:wing:MAC:at25percent:x", val=np.nan, units="m")
        self.add_input(
            "data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25", val=np.nan, units="m"
        )
        self.add_input("data:geometry:horizontal_tail:area", val=np.nan, units="m**2")
        self.add_input("data:geometry:propulsion:nacelle:height", val=np.nan, units="m")
        self.add_input("data:weight:aircraft:MTOW", val=np.nan, units="kg")
        self.add_input("data:weight:airframe:landing_gear:main:CG:x", val=np.nan, units="m")
        self.add_input("data:weight:aircraft_empty:CG:z", val=np.nan, units="m")
        self.add_input("data:weight:propulsion:engine:CG:z", val=np.nan, units="m")
        self.add_input("data:aerodynamics:wing:low_speed:CL0_clean", val=np.nan)
        self.add_input("data:aerodynamics:aircraft:takeoff:CL_max", val=np.nan)
        self.add_input("data:aerodynamics:wing:low_speed:CL_max_clean", val=np.nan)
        self.add_input("data:aerodynamics:flaps:takeoff:CL", val=np.nan)
        self.add_input("data:aerodynamics:flaps:takeoff:CM", val=np.nan)
        self.add_input(
            "data:aerodynamics:horizontal_tail:low_speed:CL_alpha_isolated",
            val=np.nan,
            units="rad**-1",
        )
        self.add_input("data:aerodynamics:horizontal_tail:efficiency", val=np.nan)

        self.add_input("takeoff:cl_htp", val=np.nan)
        self.add_input("takeoff:cm_wing", val=np.nan)
        self.add_input("low_speed:cl_alpha_htp", val=np.nan)

        self.add_output("data:handling_qualities:to_rotation_limit:x", units="m")
        self.add_output("data:handling_qualities:to_rotation_limit:MAC_position")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        cl_max_takeoff = inputs["data:aerodynamics:wing:low_speed:CL_max_clean"]
        cl0_clean = inputs["data:aerodynamics:wing:low_speed:CL0_clean"]
        cl_flaps_takeoff = inputs["data:aerodynamics:flaps:takeoff:CL"]
        cm_takeoff = inputs["takeoff:cm_wing"]
        cl_alpha_htp_isolated = inputs[
            "data:aerodynamics:horizontal_tail:low_speed:CL_alpha_isolated"
        ]
        cl_htp = inputs["takeoff:cl_htp"]
        tail_efficiency_factor = inputs["data:aerodynamics:horizontal_tail:efficiency"]

        n_engines = inputs["data:geometry:propulsion:engine:count"]
        x_wing_aero_center = inputs["data:geometry:wing:MAC:at25percent:x"]
        wing_area = inputs["data:geometry:wing:area"]
        wing_mac = inputs["data:geometry:wing:MAC:length"]
        ht_area = inputs["data:geometry:horizontal_tail:area"]
        lp_ht = inputs["data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25"]

        mtow = inputs["data:weight:aircraft:MTOW"]

        x_lg = inputs["data:weight:airframe:landing_gear:main:CG:x"]
        z_cg_aircraft = inputs["data:weight:aircraft_empty:CG:z"]
        z_cg_engine = inputs["data:weight:propulsion:engine:CG:z"]

        propulsion_model = self._engine_wrapper.get_model(inputs)

        # Conditions for calculation
        atm = Atmosphere(0.0)
        rho = atm.density
        sos = atm.speed_of_sound

        # Calculation of take-off minimum speed
        weight = mtow * g
        vs1 = math.sqrt(weight / (0.5 * rho * wing_area * cl_max_takeoff))

        if n_engines == 1.0:
            vr = 1.10 * vs1
        else:
            vr = 1.0 * vs1

        mach_r = vr / sos

        flight_point = oad.FlightPoint(
            mach=mach_r, altitude=0.0, engine_setting=EngineSetting.TAKEOFF, thrust_rate=1.0
        )
        propulsion_model.compute_flight_points(flight_point)
        thrust = float(flight_point.thrust)

        x_ht = x_wing_aero_center + lp_ht

        # Compute aerodynamic coefficients for takeoff @ 0° aircraft angle
        cl0_takeoff = cl0_clean + cl_flaps_takeoff

        eta_q = 1.0 + cl_alpha_htp_isolated / cl_htp * _ANG_VEL * (x_ht - x_lg) / vr
        eta_h = (x_ht - x_lg) / lp_ht * tail_efficiency_factor

        k_cl = cl_max_takeoff / (eta_q * eta_h * cl_htp)

        tail_volume_coefficient = ht_area * lp_ht / (wing_area * wing_mac)

        zt = z_cg_aircraft - z_cg_engine
        engine_contribution = zt * thrust / weight

        x_cg = (
            (
                1.0
                / k_cl
                * (tail_volume_coefficient - cl0_takeoff / cl_htp * (x_lg / wing_mac - 0.25))
                - cm_takeoff / cl_max_takeoff
            )
            * (vr / vs1) ** 2.0
            + x_lg
            - engine_contribution
        )

        outputs["data:handling_qualities:to_rotation_limit:x"] = x_cg

        x_cg_ratio = (x_cg - x_wing_aero_center + 0.25 * wing_mac) / wing_mac

        outputs["data:handling_qualities:to_rotation_limit:MAC_position"] = x_cg_ratio


class _ComputeAeroCoeffTO(om.ExplicitComponent):
    """
    Adapts aero-coefficients (reference surface is tail area for cl_ht).
    """

    def initialize(self):
        self.options.declare("landing", default=False, types=bool)

    def setup(self):

        self.add_input("data:geometry:wing:area", val=np.nan, units="m**2")
        self.add_input("data:geometry:horizontal_tail:area", val=2.0, units="m**2")
        self.add_input("data:aerodynamics:wing:low_speed:CM0_clean", val=np.nan)
        self.add_input("data:aerodynamics:horizontal_tail:low_speed:CL0", val=np.nan)
        self.add_input(
            "data:aerodynamics:horizontal_tail:low_speed:CL_alpha", val=np.nan, units="rad**-1"
        )
        self.add_input(
            "data:aerodynamics:horizontal_tail:low_speed:CL_alpha_isolated", units="rad**-1"
        )
        self.add_input("data:aerodynamics:elevator:low_speed:CL_delta", val=np.nan, units="rad**-1")
        self.add_input("data:mission:sizing:takeoff:elevator_angle", val=np.nan, units="rad")

        self.add_output("cl_htp")
        self.add_output("cm_wing")
        self.add_output("cl_alpha_htp")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        wing_area = inputs["data:geometry:wing:area"]
        ht_area = inputs["data:geometry:horizontal_tail:area"]
        cl0_htp = inputs["data:aerodynamics:horizontal_tail:low_speed:CL0"]
        cl_alpha_isolated_htp = inputs[
            "data:aerodynamics:horizontal_tail:low_speed:CL_alpha_isolated"
        ]
        cl_alpha_htp = inputs["data:aerodynamics:horizontal_tail:low_speed:CL_alpha"]
        cm0_wing = inputs["data:aerodynamics:wing:low_speed:CM0_clean"]
        cl_delta_elev = inputs["data:aerodynamics:elevator:low_speed:CL_delta"]

        # Calculate elevator max. additional lift
        elev_angle = inputs["data:mission:sizing:takeoff:elevator_angle"]
        cl_elev = cl_delta_elev * elev_angle

        # Define alpha for TO
        # Define angle of attack (aoa)
        alpha = 0.0
        # Interpolate cl/cm and define with ht reference surface
        cl_htp = (cl0_htp + (alpha * math.pi / 180) * cl_alpha_htp + cl_elev) * wing_area / ht_area

        cm_wing = cm0_wing

        outputs["cl_htp"] = cl_htp
        outputs["cm_wing"] = cm_wing
        outputs["cl_alpha_htp"] = cl_alpha_isolated_htp

    @staticmethod
    def _extrapolate(x, xp, yp) -> float:
        """
        Extrapolate linearly out of range x-value.
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
