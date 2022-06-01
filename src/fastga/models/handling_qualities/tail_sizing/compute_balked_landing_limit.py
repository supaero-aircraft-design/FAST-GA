"""
Estimation of the position of the CG that limits balked landing. Adaptation of the method
proposed by Gudmundsson.
"""
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
import scipy.interpolate as inter
from scipy.constants import g
import openmdao.api as om

# noinspection PyProtectedMember
from fastoad.module_management._bundle_loader import BundleLoader
import fastoad.api as oad
from fastoad.constants import EngineSetting

from stdatm import Atmosphere


class aircraft_equilibrium_limit(om.ExplicitComponent):
    """
    Compute the mass-lift equilibrium.
    """

    def setup(self):
        self.add_input("data:geometry:wing:MAC:length", np.nan, units="m")
        self.add_input("data:geometry:wing:MAC:at25percent:x", np.nan, units="m")
        self.add_input("data:geometry:wing:area", np.nan, units="m**2")
        self.add_input(
            "data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25", val=np.nan, units="m"
        )
        self.add_input("data:geometry:horizontal_tail:elevator_chord_ratio", val=np.nan)
        self.add_input("data:aerodynamics:wing:low_speed:CL_alpha", np.nan, units="rad**-1")
        self.add_input("data:aerodynamics:wing:low_speed:CL0_clean", np.nan)
        self.add_input("data:aerodynamics:wing:low_speed:CM0_clean", np.nan)
        self.add_input("data:aerodynamics:wing:low_speed:CL_max_clean", np.nan)
        self.add_input("data:aerodynamics:fuselage:cm_alpha", val=np.nan, units="rad**-1")
        self.add_input("data:aerodynamics:horizontal_tail:efficiency", val=np.nan)
        self.add_input(
            "data:aerodynamics:horizontal_tail:low_speed:CL_alpha", np.nan, units="rad**-1"
        )
        self.add_input(
            "data:aerodynamics:horizontal_tail:low_speed:CL_alpha_isolated", np.nan, units="rad**-1"
        )
        self.add_input(
            "data:aerodynamics:horizontal_tail:low_speed:clean:alpha_aircraft_min",
            val=np.nan,
            units="deg",
        )
        self.add_input(
            "data:aerodynamics:horizontal_tail:low_speed:clean:alpha_aircraft_max",
            val=np.nan,
            units="deg",
        )
        self.add_input("data:aerodynamics:elevator:low_speed:CL_delta", np.nan, units="rad**-1")
        self.add_input("data:aerodynamics:flaps:landing:CM", val=np.nan)
        self.add_input("data:aerodynamics:flaps:landing:CL", val=np.nan)
        self.add_input("data:aerodynamics:flaps:landing:CD", val=np.nan)
        self.add_input("data:aerodynamics:flaps:landing:CL_max", val=np.nan)
        self.add_input("data:mission:sizing:takeoff:elevator_angle", val=np.nan, units="rad")

    @staticmethod
    def found_cl_repartition(inputs, load_factor, mass, dynamic_pressure, x_cg):

        l0_wing = inputs["data:geometry:wing:MAC:length"]
        x_wing = inputs["data:geometry:wing:MAC:at25percent:x"]
        wing_area = inputs["data:geometry:wing:area"]
        x_htp = x_wing + inputs["data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25"]

        cl_max_clean = inputs["data:aerodynamics:wing:low_speed:CL_max_clean"]
        cl_alpha_htp = inputs["data:aerodynamics:horizontal_tail:low_speed:CL_alpha"]
        cl_delta_htp = inputs["data:aerodynamics:elevator:low_speed:CL_delta"]
        cm_alpha_fus = inputs["data:aerodynamics:fuselage:cm_alpha"]
        cm_flaps = inputs["data:aerodynamics:flaps:landing:CM"]
        cl_flaps = inputs["data:aerodynamics:flaps:landing:CL"]
        cl_max_flaps = inputs["data:aerodynamics:flaps:landing:CL_max"]
        tail_efficiency = float(inputs["data:aerodynamics:horizontal_tail:efficiency"])
        cl_alpha_wing = inputs["data:aerodynamics:wing:low_speed:CL_alpha"]
        cl0_wing = inputs["data:aerodynamics:wing:low_speed:CL0_clean"]
        cm0_wing = inputs["data:aerodynamics:wing:low_speed:CM0_clean"]
        stall_angle_min = inputs[
            "data:aerodynamics:horizontal_tail:low_speed:clean:alpha_aircraft_min"
        ]
        stall_angle_max = inputs[
            "data:aerodynamics:horizontal_tail:low_speed:clean:alpha_aircraft_max"
        ]

        max_elevator_deflection = inputs["data:mission:sizing:takeoff:elevator_angle"]

        cm_wing = cm0_wing + cm_flaps
        cl_max_landing = cl_max_clean + cl_max_flaps

        # Define matrix equilibrium (applying load and moment equilibrium)
        a11 = 1.0
        a12 = tail_efficiency
        b1 = mass * g * load_factor / (dynamic_pressure * wing_area)
        a21 = (x_wing - x_cg) - (cm_alpha_fus / cl_alpha_wing) * l0_wing
        a22 = tail_efficiency * (x_htp - x_cg)
        b2 = (cm_wing - (cm_alpha_fus / cl_alpha_wing) * cl0_wing) * l0_wing

        a = np.array([[a11, a12], [float(a21), float(a22)]])
        b = np.array([b1, b2])
        inv_a = np.linalg.inv(a)
        CL = np.dot(inv_a, b)

        # Now that we have the Cl on both lifting surfaces we need to find the corresponding
        # aircraft angle of attack and elevator deflection to see if the equilibrium is possible,
        # but first we must remove the effect of HLD on the wing

        Cl_corrected_1 = float(CL[0] - (cl_flaps + cl0_wing))
        Cl_corrected_2 = float(CL[1])
        CL_corrected = np.array([Cl_corrected_1, Cl_corrected_2])

        c = np.array([[float(cl_alpha_wing), 0.0], [float(cl_alpha_htp), float(cl_delta_htp)]])
        inv_c = np.linalg.inv(c)

        commands = np.dot(inv_c, CL_corrected)
        alpha_avion = commands[0]
        delta_e = commands[1]

        delta_alpha_stall = aircraft_equilibrium_limit._stall_angle_reduction(
            inputs, abs(delta_e) * 180.0 / math.pi
        )
        stall_angle_min_htp = stall_angle_min + delta_alpha_stall
        stall_angle_max_htp = stall_angle_max - delta_alpha_stall

        if abs(delta_e) > abs(max_elevator_deflection):
            equilibrium_found = False
        elif alpha_avion > stall_angle_max_htp:
            equilibrium_found = False
        elif alpha_avion < stall_angle_min_htp:
            equilibrium_found = False
        elif CL[0] > cl_max_landing:
            equilibrium_found = False
        else:
            equilibrium_found = True

        return alpha_avion, delta_e, equilibrium_found

    @staticmethod
    def _stall_angle_reduction(inputs, elevator_deflection):

        elevator_chord_ratio = inputs["data:geometry:horizontal_tail:elevator_chord_ratio"]
        cl_alpha_isolated_htp = inputs["data:aerodynamics:horizontal_tail:low_speed:CL_alpha"]
        cl_alpha_htp = inputs["data:aerodynamics:horizontal_tail:low_speed:CL_alpha_isolated"]

        elevator_chord_ratio_list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        elevator_deflection_list = [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0]

        stall_angle_reduction_list = [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.3, 0.5, 1.1, 1.6, 2.2, 2.7, 3.3, 3.9, 4.4, 5.0],
            [0.0, 0.6, 1.0, 2.1, 3.2, 4.4, 5.5, 6.6, 7.7, 8.9, 10.0],
            [0.0, 0.9, 1.5, 3.2, 4.9, 6.5, 8.2, 9.9, 11.6, 13.3, 15.0],
            [0.0, 1.2, 2.0, 4.2, 6.5, 8.7, 11.0, 13.2, 15.5, 17.7, 20.0],
            [0.0, 1.6, 2.5, 5.3, 8.1, 11.0, 13.7, 16.5, 19.4, 22.2, 25.0],
            [0.0, 1.9, 3.0, 6.4, 9.7, 13.1, 16.5, 19.9, 23.2, 26.6, 30.0],
        ]

        stall_angle_inter = inter.interp2d(
            elevator_chord_ratio_list, elevator_deflection_list, stall_angle_reduction_list
        )
        stall_angle_reduction = stall_angle_inter(elevator_chord_ratio, elevator_deflection)

        stall_angle_reduction_wrt_aircraft = (
            cl_alpha_isolated_htp / cl_alpha_htp * stall_angle_reduction
        )

        return stall_angle_reduction_wrt_aircraft


class ComputeBalkedLandingLimit(aircraft_equilibrium_limit):
    """
    Computes fwd limit position of cg in case of a balked landing
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._engine_wrapper = None

    def initialize(self):
        self.options.declare("propulsion_id", default="", types=str)

    def setup(self):
        super().setup()
        self._engine_wrapper = BundleLoader().instantiate_component(self.options["propulsion_id"])
        self._engine_wrapper.setup(self)

        self.add_input("data:weight:aircraft:MTOW", val=np.nan, units="kg")
        self.add_input("data:weight:aircraft:MLW", val=np.nan, units="kg")
        self.add_input("data:aerodynamics:wing:low_speed:induced_drag_coefficient", val=np.nan)
        self.add_input(
            "data:aerodynamics:horizontal_tail:low_speed:induced_drag_coefficient", val=np.nan
        )
        self.add_input("data:aerodynamics:aircraft:landing:CL_max", val=np.nan)
        self.add_input("data:aerodynamics:aircraft:low_speed:CD0", val=np.nan)

        self.add_output("data:handling_qualities:balked_landing_limit:x", val=4.0, units="m")
        self.add_output("data:handling_qualities:balked_landing_limit:MAC_position", val=np.nan)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        mlw = inputs["data:weight:aircraft:MLW"]

        cl_max_landing = inputs["data:aerodynamics:aircraft:landing:CL_max"]
        wing_area = inputs["data:geometry:wing:area"]
        fa_length = inputs["data:geometry:wing:MAC:at25percent:x"]
        l0_wing = inputs["data:geometry:wing:MAC:length"]

        rho = Atmosphere(0.0).density

        v_s0 = math.sqrt((mlw * 9.81) / (0.5 * rho * wing_area * cl_max_landing))
        v_ref = 1.3 * v_s0

        propulsion_model = self._engine_wrapper.get_model(inputs)

        x_cg = float(fa_length)
        increment = l0_wing / 100.0
        equilibrium_found = True
        climb_gradient_achieved = True

        while equilibrium_found and climb_gradient_achieved:
            climb_gradient, equilibrium_found = self.delta_climb_rate(
                x_cg, v_ref, mlw, propulsion_model, inputs
            )
            if climb_gradient < 0.033:
                climb_gradient_achieved = False
            x_cg -= increment

        outputs["data:handling_qualities:balked_landing_limit:x"] = x_cg

        x_cg_ratio = (x_cg - fa_length + 0.25 * l0_wing) / l0_wing

        outputs["data:handling_qualities:balked_landing_limit:MAC_position"] = x_cg_ratio

    def delta_climb_rate(self, x_cg, v_ref, mass, propulsion_model, inputs):

        coeff_k_wing = inputs["data:aerodynamics:wing:low_speed:induced_drag_coefficient"]
        coeff_k_htp = inputs["data:aerodynamics:horizontal_tail:low_speed:induced_drag_coefficient"]
        cl_alpha_wing = inputs["data:aerodynamics:wing:low_speed:CL_alpha"]
        cl_alpha_htp = inputs["data:aerodynamics:horizontal_tail:low_speed:CL_alpha"]
        cl_delta_htp = inputs["data:aerodynamics:elevator:low_speed:CL_delta"]
        cd_flaps = inputs["data:aerodynamics:flaps:landing:CD"]
        cl_flaps = inputs["data:aerodynamics:flaps:landing:CL"]
        cd_0 = inputs["data:aerodynamics:aircraft:low_speed:CD0"]

        rho = Atmosphere(0.0).density
        sos = Atmosphere(0.0).speed_of_sound

        dynamic_pressure = 1.0 / 2.0 * rho * v_ref ** 2.0

        alpha_ac, delta_e, equilibrium_found = self.found_cl_repartition(
            inputs, 1.0, mass, dynamic_pressure, x_cg
        )
        cl_AOA_wing = cl_alpha_wing * alpha_ac
        cl_AOA_htp = cl_alpha_htp * alpha_ac
        cl_elevator = cl_delta_htp * delta_e

        cd_min = cd_0 + cd_flaps

        cl = cl_AOA_wing + cl_AOA_htp + cl_elevator + cl_flaps
        cd = (
            cd_min
            + coeff_k_wing * (cl_AOA_wing + cl_flaps) ** 2.0
            + coeff_k_htp * (cl_AOA_htp + cl_elevator) ** 2.0
        )

        flight_point = oad.FlightPoint(
            mach=v_ref / sos, altitude=0.0, engine_setting=EngineSetting.TAKEOFF, thrust_rate=1.0
        )  # with engine_setting as EngineSetting
        propulsion_model.compute_flight_points(flight_point)
        thrust = float(flight_point.thrust)
        propeller_advance_ratio = v_ref / (2700.0 / 60.0 * 1.97)
        propeller_efficiency_reduction = math.sin(propeller_advance_ratio * math.pi / 2.0)

        climb_angle = math.asin(propeller_efficiency_reduction * thrust / (mass * 9.81) - cd / cl)
        climb_gradient = math.tan(climb_angle)

        return climb_gradient, equilibrium_found
