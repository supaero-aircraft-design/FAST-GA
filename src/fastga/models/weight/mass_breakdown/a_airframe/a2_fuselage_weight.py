"""
Estimation of fuselage weight
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
import math

from fastoad.model_base import Atmosphere

from fastoad.module_management._bundle_loader import BundleLoader

from fastga.models.propulsion.fuel_propulsion.base import FuelEngineSet


class ComputeFuselageWeight(om.ExplicitComponent):
    """
    Fuselage weight estimation

    Based on : Nicolai, Leland M., and Grant E. Carichner. Fundamentals of aircraft and airship design,
    Volume 1–Aircraft Design. American Institute of Aeronautics and Astronautics, 2010.

    Can also be found in : Gudmundsson, Snorri. General aviation aircraft design: Applied Methods and Procedures.
    Butterworth-Heinemann, 2013. Equation (6-25)
    """

    def setup(self):
        self.add_input("data:mission:sizing:cs23:sizing_factor_ultimate", val=np.nan)
        self.add_input("data:weight:aircraft:MTOW", val=np.nan, units="lb")
        self.add_input("data:weight:airframe:fuselage:k_factor", val=1.0)
        self.add_input("data:geometry:fuselage:maximum_width", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:maximum_height", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:length", val=np.nan, units="m")
        self.add_input("data:TLAR:v_max_sl", val=np.nan, units="kn")
        self.add_input("data:mission:sizing:main_route:cruise:altitude", val=np.nan, units="ft")

        self.add_output("data:weight:airframe:fuselage:mass", units="lb")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        sizing_factor_ultimate = inputs["data:mission:sizing:cs23:sizing_factor_ultimate"]
        mtow = inputs["data:weight:aircraft:MTOW"]
        maximum_width = inputs["data:geometry:fuselage:maximum_width"]
        maximum_height = inputs["data:geometry:fuselage:maximum_height"]
        fus_length = inputs["data:geometry:fuselage:length"]
        v_max_sl = inputs["data:TLAR:v_max_sl"]

        a2 = (
            200.0
            * (
                (mtow * sizing_factor_ultimate / (10.0 ** 5.0)) ** 0.286
                * (fus_length * 3.28084 / 10.0) ** 0.857
                * (maximum_width + maximum_height)
                * 3.28084
                / 10.0
                * (v_max_sl / 100.0) ** 0.338
            )
            ** 1.1
        )  # mass formula in lb

        outputs["data:weight:airframe:fuselage:mass"] = (
            a2 * inputs["data:weight:airframe:fuselage:k_factor"]
        )


class ComputeFuselageWeightAlternate(om.ExplicitComponent):
    """
    The geometry of the aircraft fuselage is inspired by the one proposed in the TASOPT 2.0 (2010). In the FASTGA
    model the cabin is delimited by two flat bulkheads, which separate the cabin from the non pressurized nose and
    tail cone.
    The mass models of the fuselage extra components (windows, doors,...) are extracted from the Torenbeek weight
    penalty method presented page 457 of his book "Synthesis of Subsonic Aircraft Design".
    Through all this model the hypothesis is made that the thickness of the fuselage skin is constant and far inferior
    to the fuselage radius.
    For the moment no k factor has been considered to correct the value of the output mass.
    TODO ? Apply this factor on all the output components or only on the final airframe mass ?
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._engine_wrapper = None

    def initialize(self):
        self.options.declare("propulsion_id", default="", types=str)

    def setup(self):
        self._engine_wrapper = BundleLoader().instantiate_component(self.options["propulsion_id"])
        self._engine_wrapper.setup(self)

        self.add_input("data:geometry:fuselage:maximum_width", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:maximum_height", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:front_length", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:rear_length", val=np.nan, units="m")
        self.add_input("data:geometry:cabin:length", val=np.nan, units="m")
        self.add_input("data:geometry:cabin:max_differential_pressure", val=np.nan, units="Pa")
        self.add_input("data:geometry:fuselage:wet_area", val=np.nan, units="m**2")
        self.add_input(
            "data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25", val=np.nan, units="m"
        )
        self.add_input("data:geometry:horizontal_tail:root:chord", val=np.nan, units="m")
        self.add_input("data:geometry:landing_gear:height", val=np.nan, units="m")
        self.add_input("data:geometry:propulsion:count", val=np.nan)
        self.add_input("data:mission:sizing:cs23:sizing_factor_ultimate", val=np.nan)
        self.add_input("data:flight_domain:diving_speed", val=np.nan, units="m/s")
        self.add_input("data:weight:aircraft:MTOW", val=np.nan, units="kg")
        self.add_input("settings:materials:fuselage:skin:density", val=np.nan, units="kg/m**3")
        self.add_input("settings:materials:fuselage:skin:young_modulus", val=np.nan, units="Pa")
        self.add_input("settings:materials:fuselage:skin:sigma_max", val=np.nan, units="Pa")
        self.add_input("settings:materials:fuselage:skin:sigma_02", val=np.nan, units="Pa")
        self.add_input("settings:materials:fuselage:stringer:density", val=np.nan, units="kg/m**3")
        self.add_input("settings:materials:fuselage:stringer:young_modulus", val=np.nan, units="Pa")
        self.add_input("settings:geometry:fuselage:min_skin_thickness", val=np.nan, units="m")

        self.add_output("data:geometry:fuselage:skin_thickness", units="m")
        self.add_output("data:weight:airframe:fuselage:shell_mass", units="kg")
        self.add_output("data:weight:airframe:fuselage:tail_cone_mass", units="kg")
        self.add_output("data:weight:airframe:fuselage:nose_mass", units="kg")
        self.add_output("data:weight:airframe:fuselage:pax_windows_mass", units="kg")
        self.add_output("data:weight:airframe:fuselage:cockpit_window_mass", units="kg")
        self.add_output("data:weight:airframe:fuselage:nlg_door_mass", units="kg")
        self.add_output("data:weight:airframe:fuselage:doors_mass", units="kg")
        self.add_output("data:weight:airframe:fuselage:wing_fuselage_connection_mass", units="kg")
        self.add_output("data:weight:airframe:fuselage:engine_support_mass", units="kg")
        self.add_output("data:weight:airframe:fuselage:floor_mass", units="kg")
        self.add_output("data:weight:airframe:fuselage:bulkheads_mass", units="kg")
        self.add_output("data:weight:airframe:fuselage:mass", units="kg")
        self.add_output("data:weight:airframe:fuselage:total_additional_mass", units="kg")

        self.add_output("data:loads:fuselage:inertia", units="m**4")
        self.add_output("data:loads:fuselage:sigmaMh", units="N/m**2")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        fuselage_max_width = inputs["data:geometry:fuselage:maximum_width"]
        fuselage_max_height = inputs["data:geometry:fuselage:maximum_height"]
        l_front = inputs["data:geometry:fuselage:front_length"]
        l_rear = inputs["data:geometry:fuselage:rear_length"]
        l_cabin = inputs["data:geometry:cabin:length"]
        delta_p_max = inputs["data:geometry:cabin:max_differential_pressure"]
        fuselage_wet_area = inputs["data:geometry:fuselage:wet_area"]
        htp_mac_25_from_wing_mac_25 = inputs[
            "data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25"
        ]
        htp_root_chord = inputs["data:geometry:horizontal_tail:root:chord"]
        lg_height = inputs["data:geometry:landing_gear:height"]
        engine_layout = inputs["data:geometry:propulsion:layout"]
        n_ult = inputs["data:mission:sizing:cs23:sizing_factor_ultimate"]
        v_d = inputs["data:flight_domain:diving_speed"]
        mtow = inputs["data:weight:aircraft:MTOW"]
        rho_skin = inputs["settings:materials:fuselage:skin:density"]
        e_skin = inputs["settings:materials:fuselage:skin:young_modulus"]
        sigma_max_skin = inputs["settings:materials:fuselage:skin:sigma_max"]
        sigma_02_skin = inputs["settings:materials:fuselage:skin:sigma_02"]
        rho_stringer = inputs["settings:materials:fuselage:stringer:density"]
        e_stringer = inputs["settings:materials:fuselage:stringer:young_modulus"]
        min_skin_thickness = inputs["settings:geometry:fuselage:min_skin_thickness"]

        fuselage_radius = fuselage_max_width / 2

        ratio_e = e_stringer / e_skin

        # Sizing of thickness based on hoop stress and norm constraints (limit load factor : 1.3, safety factor = 1.5)
        # Takes in account the minimum achievable industrial thickness
        thickness_limit = 1.33 * delta_p_max * fuselage_radius / sigma_02_skin
        thickness_ultimate = 1.5 * thickness_limit

        fuselage_skin_thickness = max(thickness_ultimate, min_skin_thickness)

        # Volume of fuselage skin components (TASOPT)
        volume_nose_skin = (
            2
            * np.pi
            * fuselage_radius ** 2
            * (1 / 3 + 2 / 3 * (l_front / fuselage_radius) ** (8 / 5)) ** (5 / 8)
            * fuselage_skin_thickness
        )
        volume_cabin_skin = 2 * np.pi * fuselage_radius * fuselage_skin_thickness * l_cabin
        # Taper parameter validated graphically and by computing the area function of TASOPT
        taper_cone = 0.2
        volume_cone_skin = (
            np.pi * fuselage_radius * fuselage_skin_thickness * l_rear * (1 + taper_cone ** 2)
        )

        # Mass of fuselage airframe skin
        volume_fuse_airframe = volume_nose_skin + volume_cabin_skin + volume_cone_skin
        mass_skin = rho_skin * volume_fuse_airframe

        # Mass of stringers (Torenbeek p459 formula D-6)
        # k_lambda is the factor which takes in account fuselage slenderness
        htp_root_from_wing_mac_25 = htp_mac_25_from_wing_mac_25 - 0.25 * htp_root_chord
        if (htp_root_from_wing_mac_25 / (fuselage_max_height + fuselage_max_width)) <= 2.61:
            k_lambda = (
                0.56
                * (htp_root_from_wing_mac_25 / (fuselage_max_height + fuselage_max_width)) ** 0.75
            )
        else:
            k_lambda = 1.15
        mass_stringer = 0.0117 * k_lambda * fuselage_wet_area ** 1.45 * v_d ** 0.39 * n_ult ** 0.316

        # Mass of frames (Torenbeek p459 formulas D-8 D-9)
        if mass_stringer + mass_skin > 286:
            mass_frames = 0.19 * (mass_stringer + mass_skin)
        else:
            mass_frames = 0.0911 * (mass_stringer + mass_skin) ** 1.13

        # Shell = skin + stringers / frames / additional local reinforcements everywhere (nose + cabin + rear cone)
        fuselage_shell_mass = mass_skin + mass_stringer + mass_frames
        cone_mass = fuselage_shell_mass * volume_cone_skin / volume_fuse_airframe
        nose_mass = fuselage_shell_mass * volume_nose_skin / volume_fuse_airframe
        shell_mass_area_density = fuselage_shell_mass / fuselage_wet_area

        # Equivalent thickness of shell, with the assumption that only the skin and the stringers contribute to bending.
        thick_shell = fuselage_skin_thickness * (
            1 + mass_stringer / mass_skin * rho_skin / rho_stringer * ratio_e
        )

        # Moment of inertia bending (about y and z axis)
        i_shell = np.pi * fuselage_radius ** 3 * thick_shell

        # Sigma Mh for added horizontal material.
        sigma_mh = sigma_max_skin - ratio_e * delta_p_max * fuselage_radius / thick_shell / 2

        pressurized = delta_p_max != 0

        # Mass of one cabin window
        window_width = 0.5
        window_height = 0.5
        if pressurized:
            mass_window_component = 23.9 * window_width * window_height * fuselage_max_width ** 0.5
        else:
            mass_window_component = 12.2 * window_height * window_width
        mass_removed = window_height * window_width * shell_mass_area_density
        mass_window = mass_window_component - mass_removed
        mass_windows = 4 * mass_window

        # Mass of one crew/passenger door
        delta_p_max = 0.4359
        door_width = 0.61  # defined by cs23 norm for commuters
        door_height = 0.8 * fuselage_max_height
        if pressurized:
            mass_door_component = 44.2 * delta_p_max ** 0.5 * door_height * door_width
            mass_surrounds = 29.8 * (door_width * door_height) ** 0.5
        else:
            mass_door_component = 9.765 * door_height * door_width
            mass_surrounds = 14.9 * (door_width * door_height) ** 0.5
        mass_removed = door_width * door_height * shell_mass_area_density
        mass_door = mass_door_component + mass_surrounds - mass_removed
        mass_doors = 2 * mass_door

        # Mass of the nose landing gears door
        door_height = 1.1 * lg_height
        door_width = 0.3
        if pressurized:
            mass_door_component = 22 * door_height * door_width
        else:
            mass_door_component = 16.1 * door_height * door_width
        mass_surrounds = 9.97 * (door_width * door_height) ** 0.5
        mass_removed = door_width * door_height * shell_mass_area_density
        mass_nlg_door = mass_door_component + mass_surrounds - mass_removed

        # Mass of cockpit window
        window_width = fuselage_max_width
        window_height = 0.5 * fuselage_max_height
        window_thickness = 0.0056
        window_mass_density = 1180
        mass_window_component = (
            window_height * window_width * window_thickness * window_mass_density
        )
        mass_surrounds = 2.98 * (window_width * window_height) ** 0.5
        mass_removed = window_width * window_height * shell_mass_area_density
        mass_cockpit_window = mass_window_component + mass_surrounds - mass_removed

        # Mass of the fuselage/wing connection
        if pressurized:
            mass_wing_fuselage = 20.4 + 0.907 * 0.001 * n_ult * mtow
        else:
            mass_wing_fuselage = 0.4 * 0.001 * (n_ult * mtow) ** 1.185

        # Mass of the floor
        kfl = 4.62
        floor_surface = l_cabin * fuselage_max_width
        mass_floor = kfl * floor_surface ** 1.045

        # Mass of the bulkheads
        if pressurized:
            mass_bulkhead = 9.1 + 12.48 * delta_p_max ** 0.8 * np.pi * (fuselage_max_width / 2) ** 2
        else:
            mass_bulkhead = 0
        mass_bulkheads = 2 * mass_bulkhead

        # Mass of the support structure for a nose or rear mounted engine
        propulsion_model = FuelEngineSet(
            self._engine_wrapper.get_model(inputs), inputs["data:geometry:propulsion:count"]
        )

        uninstalled_engine_weight = propulsion_model.compute_weight()

        mass_engine = 1.4 * uninstalled_engine_weight

        if engine_layout == 2 or engine_layout == 3:
            mass_support_engine = 0.025 * mass_engine
        else:
            mass_support_engine = 0

        fuselage_additional_mass = (
            mass_windows
            + mass_doors
            + mass_nlg_door
            + mass_cockpit_window
            + mass_wing_fuselage
            + mass_floor
            + mass_bulkheads
            + mass_support_engine
        )

        outputs["data:geometry:fuselage:skin_thickness"] = fuselage_skin_thickness
        outputs["data:weight:airframe:fuselage:shell_mass"] = fuselage_shell_mass
        outputs["data:weight:airframe:fuselage:tail_cone_mass"] = cone_mass
        outputs["data:weight:airframe:fuselage:nose_mass"] = nose_mass
        outputs["data:weight:airframe:fuselage:pax_windows_mass"] = mass_windows
        outputs["data:weight:airframe:fuselage:cockpit_window_mass"] = mass_cockpit_window
        outputs["data:weight:airframe:fuselage:nlg_door_mass"] = mass_nlg_door
        outputs["data:weight:airframe:fuselage:doors_mass"] = mass_doors
        outputs["data:weight:airframe:fuselage:wing_fuselage_connection_mass"] = mass_wing_fuselage
        outputs["data:weight:airframe:fuselage:engine_support_mass"] = mass_support_engine
        outputs["data:weight:airframe:fuselage:floor_mass"] = mass_floor
        outputs["data:weight:airframe:fuselage:bulkheads_mass"] = mass_bulkheads
        outputs["data:weight:airframe:fuselage:mass"] = (
            fuselage_shell_mass + fuselage_additional_mass
        )
        outputs["data:weight:airframe:fuselage:total_additional_mass"] = fuselage_additional_mass
        outputs["data:loads:fuselage:inertia"] = i_shell
        outputs["data:loads:fuselage:sigmaMh"] = sigma_mh


class ComputeFuselageWeightRaymer(om.ExplicitComponent):
    """
        Fuselage weight estimation

        Based on : Raymer, Daniel. Aircraft design: a conceptual approach. American Institute of Aeronautics and
        Astronautics, Inc., 2012.

        Can also be found in : Gudmundsson, Snorri. General aviation aircraft design: Applied Methods and Procedures.
        Butterworth-Heinemann, 2013. Equation (6-25)
        """

    def setup(self):

        self.add_input("data:geometry:fuselage:length", val=np.nan, units="ft")
        self.add_input("data:geometry:fuselage:front_length", val=np.nan, units="ft")
        self.add_input("data:geometry:fuselage:rear_length", val=np.nan, units="ft")
        self.add_input("data:geometry:fuselage:maximum_width", val=np.nan, units="ft")
        self.add_input("data:geometry:fuselage:maximum_height", val=np.nan, units="ft")
        self.add_input("data:geometry:fuselage:wet_area", val=np.nan, units="ft**2")
        self.add_input("data:mission:sizing:cs23:sizing_factor_ultimate", val=np.nan)
        self.add_input("data:weight:aircraft:MTOW", val=np.nan, units="lb")
        self.add_input("data:weight:airframe:fuselage:k_factor", val=1.0)
        self.add_input(
            "data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25", val=np.nan, units="ft"
        )
        self.add_input("data:mission:sizing:main_route:cruise:altitude", val=np.nan, units="ft")
        self.add_input("data:TLAR:v_cruise", val=np.nan, units="kn")

        self.add_output("data:weight:airframe:fuselage:mass_raymer", units="lb")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        fus_length = inputs["data:geometry:fuselage:length"]
        lav = inputs["data:geometry:fuselage:front_length"]
        lar = inputs["data:geometry:fuselage:rear_length"]
        maximum_width = inputs["data:geometry:fuselage:maximum_width"]
        maximum_height = inputs["data:geometry:fuselage:maximum_height"]
        wet_area_fus = inputs["data:geometry:fuselage:wet_area"]
        sizing_factor_ultimate = inputs["data:mission:sizing:cs23:sizing_factor_ultimate"]
        mtow = inputs["data:weight:aircraft:MTOW"]
        lp_ht = inputs["data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25"]
        cruise_alt = inputs["data:mission:sizing:main_route:cruise:altitude"]
        v_cruise = inputs["data:TLAR:v_cruise"] * 0.5144

        atm_cruise = Atmosphere(cruise_alt)
        rho_cruise = atm_cruise.density
        pressure_cruise = atm_cruise.pressure

        atm_sl = Atmosphere(0.0)
        pressure_sl = atm_sl.pressure

        dynamic_pressure = 1.0 / 2.0 * rho_cruise * v_cruise ** 2.0 * 0.020885434273039

        if cruise_alt > 10000.0:
            fus_dia = (maximum_height + maximum_width) / 2.0
            v_press = (fus_length - lar - lav) * math.pi * (fus_dia / 2.0) ** 2.0
            delta_p = (pressure_sl - pressure_cruise) * 0.000145038
        else:
            v_press = 0.0
            delta_p = 0.0

        a2 = 0.052 * (
            wet_area_fus ** 1.086
            * (sizing_factor_ultimate * mtow) ** 0.177
            * lp_ht ** (-0.051)
            * ((fus_length - lar - lav) / maximum_height) ** (-0.072)
            * dynamic_pressure ** 0.241
            + 11.9 * (v_press * delta_p) ** 0.271
        )

        outputs["data:weight:airframe:fuselage:mass_raymer"] = (
            a2 * inputs["data:weight:airframe:fuselage:k_factor"]
        )
