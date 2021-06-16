"""
Estimation of vertical tail area
"""
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
import openmdao.api as om

from scipy.optimize import fsolve
from fastoad.model_base.atmosphere import Atmosphere
from fastoad.module_management.service_registry import BundleLoader
from ...propulsion.fuel_propulsion.base import FuelEngineSet
from fastoad.model_base import FlightPoint
from fastoad.constants import EngineSetting


class UpdateVTArea(om.ExplicitComponent):
    """
    Computes needed vt area to:
      - have enough rotational moment/controllability during cruise
      - maintain a straight flight path during a crosswing landing
      - compensate 1-failed engine linear trajectory at limited altitude (5000ft)
      - compensate 1-failed engine linear trajectory at takeoff
      - compensate 1-failed engine linear trajectory at landing
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._engine_wrapper = None

    def initialize(self):
        self.options.declare("propulsion_id", default="", types=str)

    def setup(self):
        self._engine_wrapper = BundleLoader().instantiate_component(self.options["propulsion_id"])
        self._engine_wrapper.setup(self)

        self.add_input("data:TLAR:v_approach", val=np.nan, units="m/s")
        self.add_input("data:geometry:propulsion:count", val=np.nan)
        self.add_input("data:geometry:wing:area", val=np.nan, units="m**2")
        self.add_input("data:geometry:wing:span", val=np.nan, units="m")
        self.add_input("data:geometry:wing:aspect_ratio", val=np.nan)
        self.add_input("data:geometry:wing:sweep_25", val=np.nan, units="rad")
        self.add_input("data:geometry:wing:MAC:length", val=np.nan, units="m")
        self.add_input("data:geometry:wing:MAC:at25percent:x", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:maximum_width", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:maximum_height", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:front_length", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:rear_length", val=np.nan, units="m")
        self.add_input("data:geometry:cabin:length", val=np.nan, units="m")
        self.add_input("data:geometry:vertical_tail:MAC:at25percent:x:from_wingMAC25", val=np.nan, units="m")
        self.add_input("data:geometry:vertical_tail:rudder:max_deflection", val=np.nan, units="deg")
        self.add_input("data:geometry:propulsion:nacelle:y", val=np.nan, units="m")
        self.add_input("data:weight:aircraft:CG:aft:MAC_position", val=np.nan)
        self.add_input("data:weight:aircraft:MTOW", val=np.nan, units="kg")
        self.add_input("data:weight:aircraft:OWE", val=np.nan, units="kg")
        self.add_input("data:weight:aircraft:payload", val=np.nan, units="kg")
        self.add_input("data:aerodynamics:wing:low_speed:CL_max_clean", val=np.nan)
        self.add_input("data:aerodynamics:aircraft:landing:CL_max", val=np.nan)
        self.add_input("data:aerodynamics:aircraft:takeoff:CL_max", val=np.nan)
        self.add_input("data:aerodynamics:fuselage:cruise:CnBeta", val=np.nan)
        self.add_input("data:aerodynamics:vertical_tail:cruise:CL_alpha", val=np.nan, units="rad**-1")
        self.add_input("data:aerodynamics:vertical_tail:low_speed:CL_alpha", val=np.nan, units="rad**-1")
        self.add_input("data:aerodynamics:rudder:low_speed:Cy_delta_r", val=np.nan, units="rad**-1")

        self.add_input("data:mission:sizing:landing:target_sideslip", val=11.5, units="deg")

        self.add_input(
            "settings:handling_qualities:rudder:safety_margin",
            val=0.20,
            desc="Ratio of the total rudder deflection not used in the computation of the VT area to leave a safety "
                 "margin",
        )

        self.add_output("data:geometry:vertical_tail:area", val=2.5, units="m**2")

        self.declare_partials(
            "*",
            "*",
            method="fd",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        # Sizing constraints for the vertical tail.
        # Limiting cases: rotating torque objective (cn_beta_goal) during cruise, and  
        # compensation of engine failure induced torque at approach speed/altitude. 
        # Returns maximum area.

        propulsion_model = FuelEngineSet(
            self._engine_wrapper.get_model(inputs), inputs["data:geometry:propulsion:count"]
        )

        cruise_speed = inputs["data:TLAR:v_cruise"]

        engine_number = inputs["data:geometry:propulsion:count"]
        y_nacelle = inputs["data:geometry:propulsion:nacelle:y"]

        wing_area = inputs["data:geometry:wing:area"]
        span = inputs["data:geometry:wing:span"]
        l0_wing = inputs["data:geometry:wing:MAC:length"]

        wing_vtp_distance = inputs["data:geometry:vertical_tail:MAC:at25percent:x:from_wingMAC25"]
        rudder_max_deflection = inputs["data:geometry:vertical_tail:rudder:max_deflection"] * np.pi / 180.

        rudder_usage = 1. - inputs["settings:handling_qualities:rudder:safety_margin"]

        cg_mac_position = inputs["data:weight:aircraft:CG:aft:MAC_position"]

        mtow = inputs["data:weight:aircraft:MTOW"]
        owe = inputs["data:weight:aircraft:OWE"]
        payload = inputs["data:weight:aircraft:payload"]

        cl_max_clean = inputs["data:aerodynamics:wing:low_speed:CL_max_clean"]
        cl_max_takeoff = inputs["data:aerodynamics:aircraft:takeoff:CL_max"]
        cl_max_landing = inputs["data:aerodynamics:aircraft:landing:CL_max"]
        cn_beta_fuselage = inputs["data:aerodynamics:fuselage:cruise:CnBeta"]
        cl_alpha_vt_cruise = inputs["data:aerodynamics:vertical_tail:cruise:CL_alpha"]
        cy_delta_r = inputs["data:aerodynamics:rudder:low_speed:Cy_delta_r"]

        cruise_altitude = inputs["data:mission:sizing:main_route:cruise:altitude"]

        efficiency_vt = 0.95

        # CASE1: OBJECTIVE TORQUE @ CRUISE #############################################################################

        atm_cr = Atmosphere(cruise_altitude)
        speed_of_sound_cr = atm_cr.speed_of_sound
        cruise_mach = cruise_speed / speed_of_sound_cr
        # Matches suggested goal by Raymer, Fig 16.20
        # TODO: 2 contributions are missing in the equations suggested by Raymer the main one being the wing
        # TODO: contribution, which would require dihedral angle, ...
        # TODO : page 649 of pdf of Raymer's book
        cn_beta_goal = 0.0569 - 0.01694 * cruise_mach + 0.15904 * cruise_mach ** 2

        required_cn_beta_vtp = cn_beta_goal - cn_beta_fuselage
        distance_to_cg = wing_vtp_distance + 0.25 * l0_wing - cg_mac_position * l0_wing
        area_1 = required_cn_beta_vtp / (distance_to_cg / wing_area / span * cl_alpha_vt_cruise)

        # CASE 2 : CROSSWIND LANDING ###################################################################################
        # In this case we need to solve the lateral equilibrium equation. This method is adapted form what is presented
        # in : Al-Shamma, Omran, Rashid Ali, and Haitham S. Hasan. "An Educational Rudder Sizing Algorithm for
        # Utilization in Aircraft Design Software." International Journal of Applied Engineering Research 13.10 (2018):
        # 7889-7894.
        beta_crosswind = float(inputs["data:mission:sizing:landing:target_sideslip"]) * np.pi / 180.
        rudder_deflection = - float(rudder_usage * rudder_max_deflection)

        results = fsolve(self.lateral_equilibrium, np.array([float(area_1), 2. / 3. * beta_crosswind]), args=(
            inputs, beta_crosswind, rudder_deflection, efficiency_vt
        ), xtol=1e-4)
        area_2 = results[0]

        # CASE3: ENGINE FAILURE COMPENSATION DURING CLIMB ##############################################################

        failure_altitude_cl = 1524.0  # CS23 for Twin engine - at 5000ft
        atm_cl = Atmosphere(failure_altitude_cl)
        speed_of_sound_cl = atm_cl.speed_of_sound
        pressure_cl = atm_cl.pressure
        # TODO : compute the case for 4 engines on the wing, with identification of critic loss of engine
        if engine_number == 2.0:
            # STEP 1.0 for
            # Computation of the stall speed to get the minimum climb speed
            stall_speed_cl = np.sqrt((2. * mtow * 9.81) / (atm_cl.density * wing_area * cl_max_clean))
            speed_cl = 1.2 * stall_speed_cl  # Flights mechanics from GA - Serge Bonnet CS23, and CS 23.65
            mach_cl = speed_cl / speed_of_sound_cl
            # Calculation of engine power for given conditions
            flight_point_cl = FlightPoint(
                mach=mach_cl, altitude=failure_altitude_cl, engine_setting=EngineSetting.CLIMB,
                thrust_rate=1.0
            )  # forced to maximum thrust
            propulsion_model.compute_flight_points(flight_point_cl)
            # If you look at FuelEngineSet method to compute flight point you will see that it is multiplied buy engine
            # count so we must divide it here to get the thrust of 1 engine only
            thrust_cl = float(flight_point_cl.thrust) / 2.
            # Calculation of engine thrust and nacelle drag (failed one)
            max_power_oe_cl_hp = propulsion_model.compute_max_power(flight_point_cl) * 1.34102
            speed_cl_fps = speed_cl * 3.28084
            windmilling_prop_drag_cl = 33 * max_power_oe_cl_hp / speed_cl_fps
            # Roskam equation 4.68 in aerodynamics
            # Torque compensation
            rudder_side_force_coefficient = cy_delta_r * rudder_usage * rudder_max_deflection
            # TODO : Change aspect ratio of vertical tail
            area_3 = (
                y_nacelle * (thrust_cl + windmilling_prop_drag_cl)
            ) / (
                0.7 * pressure_cl * mach_cl ** 2 * efficiency_vt * rudder_side_force_coefficient * distance_to_cg
            )
        else:
            area_3 = 0.0

        # CASE4: ENGINE FAILURE COMPENSATION DURING TAKEOFF ############################################################

        failure_altitude_to = 0.0  # CS23 for Twin engine - at 0ft
        atm_to = Atmosphere(failure_altitude_to)
        speed_of_sound_to = atm_to.speed_of_sound
        pressure_to = atm_to.pressure
        if engine_number == 2.0:
            # STEP 1.0 for
            # Computation of the stall speed to get the minimum climb speed
            stall_speed_to = np.sqrt((2. * mtow * 9.81) / (atm_to.density * wing_area * cl_max_takeoff))
            vmc_to = 1.2 * stall_speed_to  # CS 23.149 (a)
            mc_mach_to = vmc_to / speed_of_sound_to
            # Calculation of engine power for given conditions
            flight_point_to = FlightPoint(
                mach=mc_mach_to, altitude=failure_altitude_to, engine_setting=EngineSetting.CLIMB,
                thrust_rate=1.0
            )  # forced to maximum thrust
            propulsion_model.compute_flight_points(flight_point_to)
            thrust_to = float(flight_point_to.thrust) / 2.
            # Calculation of engine thrust and nacelle drag (failed one)
            max_power_oe_to_hp = propulsion_model.compute_max_power(flight_point_to) * 1.34102
            mc_speed_to_fps = vmc_to * 3.28084
            windmilling_prop_drag_to = 33 * max_power_oe_to_hp / mc_speed_to_fps
            # Roskam equation 4.68 in aerodynamics
            # Torque compensation
            # Only take 0.85 percent ot leave a margin for the pilot
            rudder_side_force_coefficient = cy_delta_r * rudder_usage * rudder_max_deflection
            # We assume that the lift contribution added by the bank is located at the wing aerodynamics center
            bank_lever_arm = cg_mac_position * l0_wing - 0.25 * l0_wing
            # Vertical component of lift equals the weight, horizontal component used in bank
            bank_contribution_to = mtow * np.tan(5. * np.pi / 180.)
            area_4 = (
                y_nacelle * (thrust_to + windmilling_prop_drag_to) - bank_lever_arm * bank_contribution_to
            ) / (
                0.7 * pressure_to * mc_mach_to ** 2 * efficiency_vt * rudder_side_force_coefficient * distance_to_cg
            )
        else:
            area_4 = 0.0

        # CASE5: ENGINE FAILURE COMPENSATION DURING LANDING ############################################################

        failure_altitude_ldg = 0.0  # CS23 for Twin engine - at 0ft
        atm_ldg = Atmosphere(failure_altitude_ldg)
        speed_of_sound_ldg = atm_ldg.speed_of_sound
        pressure_ldg = atm_ldg.pressure
        if engine_number == 2.0:
            # STEP 1.0 for
            # Computation of the stall speed to get the minimum climb speed
            stall_speed_ldg = np.sqrt((2. * mtow * 9.81) / (atm_ldg.density * wing_area * cl_max_landing))
            vmc_ldg = 1.2 * stall_speed_ldg  # CS 23.149 (a)
            mc_mach_ldg = vmc_ldg / speed_of_sound_ldg
            # Calculation of engine power for given conditions
            flight_point_ldg = FlightPoint(
                mach=mc_mach_ldg, altitude=failure_altitude_ldg, engine_setting=EngineSetting.CLIMB,
                thrust_rate=1.0
            )  # forced to maximum thrust
            propulsion_model.compute_flight_points(flight_point_ldg)
            thrust_ldg = float(flight_point_ldg.thrust) / 2.
            # Calculation of engine thrust and nacelle drag (failed one)
            max_power_oe_ldg_hp = propulsion_model.compute_max_power(flight_point_ldg) * 1.34102
            mc_speed_ldg_fps = vmc_ldg * 3.28084
            windmilling_prop_drag_ldg = 33 * max_power_oe_ldg_hp / mc_speed_ldg_fps
            # Roskam equation 4.68 in aerodynamics
            # Torque compensation
            rudder_side_force_coefficient = cy_delta_r * rudder_usage * rudder_max_deflection
            # We assume that the lift contribution added by the bank is located at the wing aerodynamics center, may
            # need to put the contribution @ 0. Based on a pilot REX, there is usually no bank at landing ot avoid a
            # propeller strike on the runway
            bank_lever_arm = cg_mac_position * l0_wing - 0.25 * l0_wing
            bank_contribution_ldg = (owe + payload) * np.tan(5. * np.pi / 180.)
            area_5 = (
                y_nacelle * (thrust_ldg + windmilling_prop_drag_ldg) - bank_lever_arm * bank_contribution_ldg
            ) / (
                0.7 * pressure_ldg * mc_mach_ldg ** 2 * efficiency_vt * rudder_side_force_coefficient * distance_to_cg
            )
        else:
            area_5 = 0.0

        outputs["data:geometry:vertical_tail:area"] = max(area_1, area_2, area_3, area_4, area_5)

    @staticmethod
    def lateral_equilibrium(x, inputs, beta, rudder_angle, eta_v):
        """
        Computes the force on the y axis and the moment difference on the yaw axis

        @param inputs : dictionary containing the aircraft properties
        @param x : array containing the vertical tail plane area in m² and the crab angle in rad
        @param beta : the sideslip angle we want to compute the VTP area for, in rad
        @param rudder_angle : the rudder angle we allow ourself to maintain a straight flight path during the crosswind
        landing
        @param eta_v : defines the percentage of aircraft dynamic pressure seen by the vtp
        @return result_array : an array containing the moment and force difference on the appropriate axis
        """
        v_f = float(inputs["data:TLAR:v_approach"])

        ar_wing = float(inputs["data:geometry:wing:aspect_ratio"])
        sweep_wing = float(inputs["data:geometry:wing:sweep_25"])
        area_wing = float(inputs["data:geometry:wing:area"])
        span_wing = float(inputs["data:geometry:wing:span"])
        l0_wing = float(inputs["data:geometry:wing:MAC:length"])
        fa_length = float(inputs["data:geometry:wing:MAC:at25percent:x"])

        wing_vtp_distance = float(inputs["data:geometry:vertical_tail:MAC:at25percent:x:from_wingMAC25"])

        b_f = float(inputs["data:geometry:fuselage:maximum_width"])
        h_f = float(inputs["data:geometry:fuselage:maximum_height"])
        lcyl = float(inputs["data:geometry:cabin:length"])
        lav = float(inputs["data:geometry:fuselage:front_length"])
        lar = float(inputs["data:geometry:fuselage:rear_length"])

        cg_mac_position = float(inputs["data:weight:aircraft:CG:aft:MAC_position"])

        cl_alpha_vt_ls = float(inputs["data:aerodynamics:vertical_tail:low_speed:CL_alpha"])
        cy_delta_r_vtp = float(inputs["data:aerodynamics:rudder:low_speed:Cy_delta_r"])

        area_vtp = x[0]
        sigma = x[1]

        altitude = 0.0
        atm = Atmosphere(altitude)

        # TODO : Properly compute those variables later, using Roskam formulae
        k_f1 = 0.7
        k_f2 = 1.35
        side_drag_coefficient = 0.675

        # Computing the different speeds, w = wind, f = runway, t = total
        v_w = v_f * np.tan(beta)
        v_t = np.sqrt(v_f ** 2. + v_w ** 2.)

        # Sidewash influence on the vtp
        k_sigma = 0.724 + 0.2 + 0.009 * ar_wing + 3.06 / (1. + np.cos(sweep_wing)) * area_vtp / area_wing

        # Sideforce derivative computation
        cy_beta = - k_f1 * cl_alpha_vt_ls * k_sigma * area_vtp / area_wing
        cy_delta_r = - eta_v * cy_delta_r_vtp * area_vtp / area_wing

        # Side drag computation
        side_surface = area_vtp + h_f * (lar / 2. + lcyl + lav / 2.)
        side_drag = 0.5 * atm.density * v_w ** 2. * side_surface * side_drag_coefficient

        # Yawing moment derivative
        distance_to_cg = wing_vtp_distance + 0.25 * l0_wing - cg_mac_position * l0_wing
        cn_beta = - k_f2 / k_f1 * cy_beta * distance_to_cg / span_wing
        cn_delta_r = - cy_delta_r * distance_to_cg / span_wing

        # Side drag lever arm computation
        l_f = lav + lcyl + lar
        d_f = np.sqrt(h_f * b_f)
        x_f = (2. / 3. * lav * lav + (lav + 1. / 2. * lcyl) * lcyl + (lav + lcyl + 1. / 3. * lar) * lar) / l_f
        fa_vtp = fa_length + wing_vtp_distance
        x_ac_b = (l_f * d_f * x_f + area_vtp * fa_vtp) / (l_f * d_f + area_vtp)

        cg_position = fa_length + (cg_mac_position - 0.25) * l0_wing
        dc = x_ac_b - cg_position

        # Computation of the delta
        dynamic_pressure = 0.5 * atm.density * v_t ** 2.0

        delta_yaw = dynamic_pressure * area_wing * span_wing * (
                cn_beta * (beta - sigma) + cn_delta_r * rudder_angle
        ) + side_drag * dc * np.cos(sigma)
        delta_side_force = side_drag - dynamic_pressure * area_wing * (
                cy_beta * (beta - sigma) + cy_delta_r * rudder_angle
        )

        result_array = np.array([delta_yaw, delta_side_force])

        return result_array
