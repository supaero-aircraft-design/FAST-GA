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


class ComputeFuselageAdditionalMass(ExplicitComponent):
    # TODO: Document equations. Cite sources
    """
        Except when specified, all the formulas are taken from Torenbeek's weight penalty model.
    """

    def setup(self):
        self.add_input("data:geometry:fuselage:maximum_width", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:maximum_height", val=np.nan, units="m")
        self.add_input("data:geometry:cabin:length", val=np.nan, units="m")
        self.add_input("data:geometry:cabin:pressurized", val=False)
        self.add_input("data:geometry:fuselage:wet_area", val=np.nan, units="m**2")
        self.add_input("data:geometry:landing_gear:height", val=np.nan, units="m")
        self.add_input("data:geometry:propulsion:layout", val=np.nan)
        self.add_input("data:loads:fuselage:shell_mass", val=np.nan, units="kg")
        self.add_input("data:mission:sizing:cs23:sizing_factor_ultimate", val=np.nan)
        self.add_input("data:weight:aircraft:MTOW", val=np.nan, units="kg")
        self.add_input("data:weight:propulsion:engine:mass", val=np.nan, units="kg")

        self.add_output("data:loads:fuselage:additional_mass", units="kg")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        fuselage_max_width = inputs["data:geometry:fuselage:maximum_width"]
        fuselage_max_height = inputs["data:geometry:fuselage:maximum_height"]
        l_cabin = inputs["data:geometry:cabin:length"]
        pressurized = inputs["data:geometry:cabin:pressurized"]
        fuselage_wet_area = inputs["data:geometry:fuselage:wet_area"]
        lg_height = inputs["data:geometry:landing_gear:height"]
        engine_layout = inputs["data:geometry:propulsion:layout"]
        mass_shell = inputs["data:loads:fuselage:shell_mass"]
        n_ult = inputs["data:mission:sizing:cs23:sizing_factor_ultimate"]
        mtow = inputs["data:weight:aircraft:MTOW"]
        mass_engine = inputs["data:weight:propulsion:engine:mass"]

        # Initialization of some variables
        mass_support_engine = 0
        mass_bulkhead = 0

        # Shell mass per unit area
        shell_mass_area_density = mass_shell / fuselage_wet_area

        # Mass of one cabin window
        window_width = 0.5
        window_height = 0.5
        if pressurized:
            mass_window_component = 23.9 * window_width * window_height * fuselage_max_width ** 0.5
        else:
            mass_window_component = 12.2 * window_height * window_width
        mass_removed = window_height * window_width * shell_mass_area_density
        mass_window = mass_window_component - mass_removed

        # Mass of one crew/passenger door
        delta_p_max = 0.4359
        door_width = 0.61   # defined by cs23 norm for commuters
        door_height = 0.8 * fuselage_max_height
        if pressurized:
            mass_door_component = 44.2 * delta_p_max ** 0.5 * door_height * door_width
            mass_surrounds = 29.8 * (door_width * door_height) ** 0.5
        else:
            mass_door_component = 9.765 * door_height * door_width
            mass_surrounds = 14.9 * (door_width * door_height) ** 0.5
        mass_removed = door_width * door_height * shell_mass_area_density
        mass_door = mass_door_component + mass_surrounds - mass_removed

        # Mass of the nose landing gears door
        door_height = 1.1 * lg_height
        door_width = 0.3
        if pressurized :
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
        mass_window_component = window_height * window_width * window_thickness * window_mass_density
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

        # Mass of the support structure for a nose or rear mounted engine
        if engine_layout == 2 or engine_layout == 3:
            mass_support_engine = 0.025 * mass_engine

        fuselage_additional_mass = 4 * mass_window + 2 * mass_door + mass_nlg_door + mass_cockpit_window \
                                   + mass_wing_fuselage + mass_floor + 2 * mass_bulkhead + mass_support_engine

        outputs["data:loads:fuselage:additional_mass"] = fuselage_additional_mass

