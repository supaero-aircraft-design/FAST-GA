"""Computes the mass of the fuselage shell, adapted from TASOPT by Lucas REMOND."""
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

import openmdao.api as om

import numpy as np


class ComputeShell(om.ExplicitComponent):
    def setup(self):
        self.add_input("data:geometry:fuselage:maximum_width", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:maximum_height", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:front_length", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:rear_length", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:wet_area", val=np.nan, units="m**2")
        self.add_input("data:geometry:cabin:max_differential_pressure", val=np.nan, units="Pa")
        self.add_input("data:geometry:cabin:length", val=np.nan, units="m")
        self.add_input(
            "data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25", val=np.nan, units="m"
        )
        self.add_input("data:mission:sizing:cs23:characteristic_speed:vd", val=np.nan, units="m/s")
        self.add_input("data:mission:sizing:cs23:sizing_factor:ultimate_aircraft", val=np.nan)

        self.add_input("settings:materials:fuselage:skin:density", val=np.nan, units="kg/m**3")
        self.add_input("settings:materials:fuselage:stringer:density", val=np.nan, units="kg/m**3")
        self.add_input("settings:materials:fuselage:stringer:young_modulus", val=np.nan, units="Pa")
        self.add_input("settings:materials:fuselage:skin:young_modulus", val=np.nan, units="Pa")

        self.add_input("settings:materials:fuselage:skin:sigma_02", val=np.nan, units="Pa")
        self.add_input("settings:materials:fuselage:skin:sigma_max", val=np.nan, units="Pa")
        self.add_input("settings:geometry:fuselage:min_skin_thickness", val=np.nan, units="m")
        self.add_input("settings:weight:airframe:fuselage:reinforcements:mass_fraction", val=np.nan)

        self.add_output("data:geometry:fuselage:skin_thickness", val=1e-3, units="m")
        self.add_output("data:loads:fuselage:inertia", units="m**4")
        self.add_output("data:loads:fuselage:sigmaMh", units="N/m**2")
        self.add_output("data:weight:airframe:fuselage:shell:mass", units="kg")
        self.add_output("data:weight:airframe:fuselage:shell:added_weight_ratio")
        self.add_output("data:weight:airframe:fuselage:shell:area_density", units="kg/m**2")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        fuselage_max_width = inputs["data:geometry:fuselage:maximum_width"]
        fuselage_max_height = inputs["data:geometry:fuselage:maximum_height"]
        lav = inputs["data:geometry:fuselage:front_length"]
        lar = inputs["data:geometry:fuselage:rear_length"]
        fuselage_wet_area = inputs["data:geometry:fuselage:wet_area"]
        delta_p_max = inputs["data:geometry:cabin:max_differential_pressure"]
        cabin_length = inputs["data:geometry:cabin:length"]
        lp_ht = inputs["data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25"]
        vd = inputs["data:mission:sizing:cs23:characteristic_speed:vd"]
        n_ult = inputs["data:mission:sizing:cs23:sizing_factor:ultimate_aircraft"]

        rho_skin = inputs["settings:materials:fuselage:skin:density"]
        rho_stringer = inputs["settings:materials:fuselage:stringer:density"]
        e_skin = inputs["settings:materials:fuselage:skin:young_modulus"]
        e_stringer = inputs["settings:materials:fuselage:stringer:young_modulus"]
        sigma_max_skin = inputs["settings:materials:fuselage:skin:sigma_max"]
        sigma_02_skin = inputs["settings:materials:fuselage:skin:sigma_02"]
        min_skin_thickness = inputs["settings:geometry:fuselage:min_skin_thickness"]
        f_add = inputs["settings:weight:airframe:fuselage:reinforcements:mass_fraction"]

        fuselage_radius = np.sqrt(fuselage_max_height * fuselage_max_width) / 2.0

        ratio_e = e_stringer / e_skin

        # Sizing skin mass based on pressure differential constraints or technological minimum
        thickness_limit = 1.33 * delta_p_max * fuselage_radius / sigma_02_skin
        thickness_ultimate = 1.5 * thickness_limit

        fuselage_skin_thickness = max(thickness_ultimate, min_skin_thickness)

        volume_nose_skin = (
            2
            * np.pi
            * fuselage_radius ** 2
            * (1.0 / 3.0 + 2.0 / 3.0 * (lav / fuselage_radius) ** (8 / 5)) ** (5 / 8)
            * fuselage_skin_thickness
        )

        volume_cabin_skin = 2 * np.pi * fuselage_radius * fuselage_skin_thickness * cabin_length

        # Taper parameter validated graphically and by computing the area function of TASOPT
        taper_cone = 0.2
        volume_cone_skin = (
            np.pi * fuselage_radius * fuselage_skin_thickness * lar * (1 + taper_cone ** 2)
        )

        # Mass of fuselage airframe skin
        volume_fuse_airframe = volume_nose_skin + volume_cabin_skin + volume_cone_skin
        mass_skin = rho_skin * volume_fuse_airframe

        # Mass of stringers (Torenbeek p459 formula D-6)
        # k_lambda is the factor which takes in account fuselage slenderness
        if (lp_ht / (fuselage_max_height + fuselage_max_width)) <= 2.61:
            k_lambda = 0.56 * (lp_ht / (fuselage_max_height + fuselage_max_width)) ** 0.75
        else:
            k_lambda = 1.15
        mass_stringer = 0.0117 * k_lambda * fuselage_wet_area ** 1.45 * vd ** 0.39 * n_ult ** 0.316

        # Mass of frames (Torenbeek p459 formulas D-8 D-9)
        if mass_stringer + mass_skin > 286.0:
            mass_frames = 0.19 * (mass_stringer + mass_skin)
        else:
            mass_frames = 0.0911 * (mass_stringer + mass_skin) ** 1.13

        mass_f_add = f_add * mass_skin

        fuselage_shell_mass = mass_skin + mass_stringer + mass_frames + mass_f_add

        # Equivalent thickness of shell, with the assumption that only the skin and the stringers
        # contribute to bending.
        thick_shell = fuselage_skin_thickness * (
            1 + mass_stringer / mass_skin * rho_skin / rho_stringer * ratio_e
        )

        # Sigma Mh for added horizontal material.
        sigma_mh = sigma_max_skin - ratio_e * delta_p_max * fuselage_radius / thick_shell / 2

        # Moment of inertia bending (about y and z axis)
        i_shell = np.pi * fuselage_radius ** 3 * thick_shell

        outputs["data:geometry:fuselage:skin_thickness"] = fuselage_skin_thickness
        outputs["data:loads:fuselage:sigmaMh"] = sigma_mh
        outputs["data:loads:fuselage:inertia"] = i_shell
        outputs["data:weight:airframe:fuselage:shell:mass"] = fuselage_shell_mass
        outputs["data:weight:airframe:fuselage:shell:added_weight_ratio"] = (
            fuselage_shell_mass / mass_skin
        )
        outputs["data:weight:airframe:fuselage:shell:area_density"] = (
            fuselage_shell_mass / fuselage_wet_area
        )
