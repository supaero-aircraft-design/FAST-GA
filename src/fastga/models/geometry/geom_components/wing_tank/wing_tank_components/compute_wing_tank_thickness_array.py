"""
Python module for tank thickness computation class(es), part of the advanced MFW computation method.
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

import numpy as np
import openmdao.api as om


class ComputeWingTankThicknessArray(om.ExplicitComponent):
    """
    Computes the thickness for each section of the tank. Based on wing thickness and a correction
    penalty.
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

        self.add_input(
            "data:geometry:propulsion:tank:chord_array",
            units="m",
            shape=nb_point_wing,
            val=np.nan,
        )
        self.add_input(
            "data:geometry:propulsion:tank:relative_thickness_array",
            shape=nb_point_wing,
            val=np.nan,
        )
        self.add_input("settings:geometry:fuel_tanks:depth", val=np.nan)

        self.add_output(
            "data:geometry:propulsion:tank:thickness_array",
            units="m",
            shape=nb_point_wing,
            val=np.full(nb_point_wing, 0.2),
        )

        self.declare_partials(
            of="data:geometry:propulsion:tank:thickness_array",
            wrt=[
                "data:geometry:propulsion:tank:chord_array",
                "data:geometry:propulsion:tank:relative_thickness_array",
            ],
            method="exact",
            rows=np.arange(nb_point_wing),
            cols=np.arange(nb_point_wing),
        )
        self.declare_partials(
            of="data:geometry:propulsion:tank:thickness_array",
            wrt="settings:geometry:fuel_tanks:depth",
            method="exact",
            rows=np.arange(nb_point_wing),
            cols=np.zeros(nb_point_wing),
        )

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute, not all arguments are used
    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs["data:geometry:propulsion:tank:thickness_array"] = (
            inputs["data:geometry:propulsion:tank:chord_array"]
            * inputs["data:geometry:propulsion:tank:relative_thickness_array"]
            * inputs["settings:geometry:fuel_tanks:depth"]
        )

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute_partials, not all arguments are used
    def compute_partials(self, inputs, partials, discrete_inputs=None):
        partials[
            "data:geometry:propulsion:tank:thickness_array",
            "data:geometry:propulsion:tank:chord_array",
        ] = (
            inputs["data:geometry:propulsion:tank:relative_thickness_array"]
            * inputs["settings:geometry:fuel_tanks:depth"]
        )
        partials[
            "data:geometry:propulsion:tank:thickness_array",
            "data:geometry:propulsion:tank:relative_thickness_array",
        ] = (
            inputs["data:geometry:propulsion:tank:chord_array"]
            * inputs["settings:geometry:fuel_tanks:depth"]
        )
        partials[
            "data:geometry:propulsion:tank:thickness_array",
            "settings:geometry:fuel_tanks:depth",
        ] = (
            inputs["data:geometry:propulsion:tank:relative_thickness_array"]
            * inputs["data:geometry:propulsion:tank:chord_array"]
        )
