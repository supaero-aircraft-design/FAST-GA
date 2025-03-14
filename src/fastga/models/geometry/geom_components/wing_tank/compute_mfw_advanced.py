"""
Python module for maximum fuel weight calculation with detailed approach, part of the geometry
component.
"""
#  This file is part of FAST-OAD_CS23 : A framework for rapid Overall Aircraft Design
#  Copyright (C) 2025  ONERA & ISAE-SUPAERO
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
import fastoad.api as oad

from ...constants import SERVICE_MFW, SUBMODEL_MFW_ADVANCED

from .components import (
    ComputeWingTankSpans,
    ComputeWingTankYArray,
    ComputeWingTankChordArray,
    ComputeWingTankRelativeThicknessArray,
    ComputeWingTankThicknessArray,
    ComputeWingTankWidthArray,
    ComputeWingTankReducedWidthArray,
    ComputeWingTankCrossSectionArray,
    ComputeWingTanksCapacity,
    ComputeMFWFromWingTanksCapacity,
)


@oad.RegisterSubmodel(SERVICE_MFW, SUBMODEL_MFW_ADVANCED)
class ComputeMFWAdvanced(om.Group):
    """
    Max fuel weight estimation based on :cite:`jenkinson:2003` p.65. It discretizes the fuel tank in
    the wings along the span. Only works for linear chord and thickness profiles. The xml
    quantities "data:geometry:propulsion:tank:LE_chord_percentage",
    "data:geometry:propulsion:tank:TE_chord_percentage",
    "data:geometry:propulsion:tank:y_ratio_tank_beginning" and
    "data:geometry:propulsion:tank:y_ratio_tank_end" have to be determined as close to possible
    as the real aircraft quantities. The quantity "settings:geometry:fuel_tanks:depth" allows to
    calibrate the model for each aircraft. WARNING : If this class is updated, update_wing_area
    will have to be updated as well as it uses the same approach.
    """

    # pylint: disable=missing-function-docstring
    # Overriding OpenMDAO initialize
    def initialize(self):
        self.options.declare(
            "number_points_wing_mfw",
            default=50,
            types=int,
            desc="Number of points to use in the computation of the maximum fuel weight using the "
            "advanced model. Reducing that number can improve convergence.",
        )

    # pylint: disable=missing-function-docstring
    # Overriding OpenMDAO setup
    def setup(self):
        nb_point_wing = self.options["number_points_wing_mfw"]

        self.add_subsystem(name="tank_span", subsys=ComputeWingTankSpans(), promotes=["*"])
        self.add_subsystem(
            name="tank_y_array",
            subsys=ComputeWingTankYArray(number_points_wing_mfw=nb_point_wing),
            promotes=["*"],
        )
        self.add_subsystem(
            name="tank_chord_array",
            subsys=ComputeWingTankChordArray(number_points_wing_mfw=nb_point_wing),
            promotes=["*"],
        )
        self.add_subsystem(
            name="tank_relative_thickness_array",
            subsys=ComputeWingTankRelativeThicknessArray(number_points_wing_mfw=nb_point_wing),
            promotes=["*"],
        )
        self.add_subsystem(
            name="tank_thickness_array",
            subsys=ComputeWingTankThicknessArray(number_points_wing_mfw=nb_point_wing),
            promotes=["*"],
        )
        self.add_subsystem(
            name="tank_width_array",
            subsys=ComputeWingTankWidthArray(number_points_wing_mfw=nb_point_wing),
            promotes=["*"],
        )
        self.add_subsystem(
            name="tank_reduced_width_array",
            subsys=ComputeWingTankReducedWidthArray(number_points_wing_mfw=nb_point_wing),
            promotes=["*"],
        )
        self.add_subsystem(
            name="tank_cross_section_array",
            subsys=ComputeWingTankCrossSectionArray(number_points_wing_mfw=nb_point_wing),
            promotes=["*"],
        )
        self.add_subsystem(
            name="tanks_capacity",
            subsys=ComputeWingTanksCapacity(number_points_wing_mfw=nb_point_wing),
            promotes=["*"],
        )
        self.add_subsystem(name="mfw", subsys=ComputeMFWFromWingTanksCapacity(), promotes=["*"])
