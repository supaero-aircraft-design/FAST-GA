"""
    Estimation of bending moments on fuselage
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
from openmdao.core.explicitcomponent import ExplicitComponent

DOMAIN_PTS_NB = 19  # number of (V,n) calculated for the flight domain


class ComputeFuselageShell(ExplicitComponent):
    # TODO: Document equations. Cite sources
    """
        Through all this model the hypothesis is made that the thickness of the fuselage skin is far inferior to the
        fuselage radius.
    """

    def setup(self):
        self.add_input("data:geometry:fuselage:maximum_width", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:maximum_height", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:front_length", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:rear_length", val=np.nan, units="m")
        self.add_input("data:geometry:cabin:length", val=np.nan, units="m")
        self.add_input("data:mission:sizing:cs23:sizing_factor_ultimate", val=np.nan)
        self.add_input("data:flight_domain:velocity", val=np.nan, units="m/s", shape=DOMAIN_PTS_NB)
        self.add_input("data:geometry:fuselage:wet_area", val=np.nan, units="m**2")
        self.add_input("data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25", val=np.nan, units="m")
        self.add_input("data:geometry:horizontal_tail:root:chord", val=np.nan, units="m")
        self.add_input("settings:materials:fuselage:skin:density", val=np.nan, units="kg/m**3")
        self.add_input("settings:materials:fuselage:skin:young_modulus", val=np.nan, units="Pa")
        self.add_input("settings:materials:fuselage:skin:sigma_max", val=np.nan, units="Pa")
        self.add_input("settings:materials:fuselage:skin:sigma_02", val=np.nan, units="Pa")
        self.add_input("settings:materials:fuselage:stringer:density", val=np.nan, units="kg/m**3")
        self.add_input("settings:materials:fuselage:stringer:young_modulus", val=np.nan, units="Pa")

        self.add_output("data:loads:fuselage:skin_thickness", units="m")
        self.add_output("data:loads:fuselage:shell_mass", units="kg")
        self.add_output("data:loads:fuselage:cone_mass", units="kg")
        self.add_output("data:loads:fuselage:nose_mass", units="kg")
        self.add_output("data:loads:fuselage:inertia", units="m**4")
        self.add_output("data:loads:fuselage:sigmaMh", units="N/m**2")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        fuselage_max_width = inputs["data:geometry:fuselage:maximum_width"]
        fuselage_max_height = inputs["data:geometry:fuselage:maximum_height"]
        l_front = inputs["data:geometry:fuselage:front_length"]
        l_rear = inputs["data:geometry:fuselage:rear_length"]
        l_cabin = inputs["data:geometry:cabin:length"]
        n_ult = inputs["data:mission:sizing:cs23:sizing_factor_ultimate"]
        v_d = inputs["data:flight_domain:velocity"][9]
        fuselage_wet_area = inputs["data:geometry:fuselage:wet_area"]
        htp_mac_25_from_wing_mac_25 = inputs["data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25"]
        htp_root_chord = inputs["data:geometry:horizontal_tail:root:chord"]
        rho_skin = inputs["settings:materials:fuselage:skin:density"]
        e_skin = inputs["settings:materials:fuselage:skin:young_modulus"]
        sigma_max_skin = inputs["settings:materials:fuselage:skin:sigma_max"]
        sigma_02_skin = inputs["settings:materials:fuselage:skin:sigma_02"]
        rho_stringer = inputs["settings:materials:fuselage:stringer:density"]
        e_stringer = inputs["settings:materials:fuselage:stringer:young_modulus"]

        fuselage_radius = fuselage_max_width / 2

        # Distance between frames and section of each frame [m]
        a = 0.75
        width_frame = 0.05 * fuselage_radius
        section_frame = np.pi * fuselage_radius ** 2 - np.pi * (fuselage_radius - width_frame) ** 2

        ratio_e = e_stringer / e_skin

        # Delta P max PIM [Pa]
        delta_p_max = 6.2 * 6894.78

        # Load cases (factor of safety = 1.5)

        thickness_limit = 1.33 * delta_p_max * fuselage_radius / sigma_02_skin
        thickness_ultimate = 1.995 * delta_p_max * fuselage_radius / sigma_max_skin

        fuselage_skin_thickness = max(thickness_limit, thickness_ultimate)

        # Volume of fuselage skin components
        volume_nose_skin = 2 * np.pi * fuselage_radius ** 2 * (1/3 + 2/3 * (l_front/fuselage_radius) ** (8/5)) ** (5/8)\
                           * fuselage_skin_thickness
        volume_cabin_skin_cyl = 2 * np.pi * fuselage_radius * fuselage_skin_thickness * l_cabin
        volume_cabin_skin_bulk = 2 * np.pi * fuselage_radius ** 2 * fuselage_skin_thickness
        volume_cabin_skin = volume_cabin_skin_cyl + volume_cabin_skin_bulk
        volume_cabin_inside = np.pi * fuselage_radius ** 2 * l_cabin + 2/3 * np.pi * fuselage_radius ** 3
        taper_cone = 0.2
        volume_cone_skin = np.pi * fuselage_radius * fuselage_skin_thickness * l_rear * (1 + taper_cone ** 2)

        # Mass of fuselage airframe skin
        volume_fuse_airframe = volume_nose_skin + volume_cabin_skin + volume_cone_skin
        mass_skin = rho_skin * volume_fuse_airframe

        # Mass of stringers (Torenbeek p459 formula D-6)
        # Factor which takes in account fuselage slenderness
        htp_root_from_wing_mac_25 = htp_mac_25_from_wing_mac_25 - 0.25 * htp_root_chord
        if (htp_root_from_wing_mac_25 / (fuselage_max_height + fuselage_max_width)) <= 2.61:
            k_lambda = 0.56 * (htp_root_from_wing_mac_25 / (fuselage_max_height + fuselage_max_width)) ** 0.75
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

        # Equivalent thickness of shell, with the assumption that only the skin and the stringers contribute to bending.

        thick_shell = fuselage_skin_thickness * (1 + mass_stringer / mass_skin * rho_skin / rho_stringer * ratio_e)

        # Moment of inertia bending (about y and z axis)
        i_shell = np.pi * fuselage_radius ** 3 * thick_shell

        # Sigma Mh for added horizontal material. TODO ne pas mettre le delta_p max mais celui associé au load case ?
        #  TODO et mettre le sigma max du stringer
        sigma_mh = sigma_max_skin - ratio_e * delta_p_max * fuselage_radius / thick_shell / 2

        outputs["data:loads:fuselage:skin_thickness"] = fuselage_skin_thickness
        outputs["data:loads:fuselage:shell_mass"] = fuselage_shell_mass
        outputs["data:loads:fuselage:cone_mass"] = cone_mass
        outputs["data:loads:fuselage:nose_mass"] = nose_mass
        outputs["data:loads:fuselage:inertia"] = i_shell
        outputs["data:loads:fuselage:sigmaMh"] = sigma_mh
