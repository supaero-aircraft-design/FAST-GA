"""
Computes the mass of the fuselage windows (cockpits and cabin), adapted from TASOPT by Lucas
REMOND.
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

import openmdao.api as om

import numpy as np


class ComputeWindows(om.ExplicitComponent):
    def setup(self):

        self.add_input("data:geometry:fuselage:maximum_width", val=np.nan, units="m")
        self.add_input("data:geometry:cabin:windows:number", val=4.0)
        self.add_input("data:geometry:cabin:windows:height", val=0.5, units="m")
        self.add_input("data:geometry:cabin:windows:width", val=0.5, units="m")
        self.add_input("data:geometry:cabin:max_differential_pressure", val=np.nan, units="hPa")
        self.add_input(
            "data:geometry:cabin:pressurized",
            val=0.0,
            desc="Cabin pressurization; 0.0 for no pressurization, 1.0 for pressurization",
        )
        self.add_input("data:geometry:cockpit:windows:height", val=np.nan, units="m")
        self.add_input("data:geometry:cockpit:windows:width", val=np.nan, units="m")
        self.add_input(
            "data:weight:airframe:fuselage:shell:area_density", val=np.nan, units="kg/m**2"
        )
        self.add_input("data:mission:sizing:cs23:characteristic_speed:vd", val=np.nan, units="m/s")

        self.add_output("data:weight:airframe:fuselage:windows:mass", units="kg")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        fuselage_maximum_width = inputs["data:geometry:fuselage:maximum_width"]
        # Converting to kg/cm**2, can't be done by OpenMDAO
        delta_p_max = inputs["data:geometry:cabin:max_differential_pressure"] * 0.00102
        cabin_windows_height = inputs["data:geometry:cabin:windows:height"]
        cabin_windows_width = inputs["data:geometry:cabin:windows:width"]
        cabin_windows_number = inputs["data:geometry:cabin:windows:number"]
        cockpit_windows_height = inputs["data:geometry:cockpit:windows:height"]
        cockpit_windows_width = inputs["data:geometry:cockpit:windows:width"]
        shell_area_density = inputs["data:weight:airframe:fuselage:shell:area_density"]
        vd = inputs["data:mission:sizing:cs23:characteristic_speed:vd"]

        pressurized = inputs["data:geometry:cabin:pressurized"]

        # Cabin windows
        cabin_window_area = cabin_windows_width * cabin_windows_height
        cabin_window_removed_material = cabin_window_area * shell_area_density
        if pressurized:
            # We ensure that the windows weight remains non-negative
            cabin_window_mass = max(
                (23.9 * cabin_windows_width * cabin_windows_height * fuselage_maximum_width ** 0.5)
                - cabin_window_removed_material,
                0.0,
            )
        else:
            # We ensure that the windows weight remains non-negative
            cabin_window_mass = max(
                (12.2 * cabin_windows_width * cabin_windows_height - cabin_window_removed_material),
                0,
            )
        cabin_windows_mass = cabin_window_mass * cabin_windows_number

        # Cockpit windows
        cockpit_window_area = cockpit_windows_height * cockpit_windows_width
        if pressurized:
            cockpit_window_filling_mass = (
                4.31
                * cockpit_window_area
                * delta_p_max ** 0.25
                * (fuselage_maximum_width * vd) ** 0.5
            )
            cockpit_window_surround_mass = 0.0
        else:
            cockpit_window_filling_mass = (15.9 + 0.104 * vd) * cockpit_window_area
            cockpit_window_surround_mass = 2.98 * cockpit_window_area ** 0.5

        cockpit_window_mass = max(cockpit_window_filling_mass + cockpit_window_surround_mass, 0.0)

        windows_mass = cockpit_window_mass + cabin_windows_mass

        outputs["data:weight:airframe:fuselage:windows:mass"] = windows_mass
