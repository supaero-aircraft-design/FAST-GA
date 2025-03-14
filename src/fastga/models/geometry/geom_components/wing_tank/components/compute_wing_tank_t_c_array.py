"""
Python module for tank relative thickness computation class(es), part of the advanced MFW
computation method.
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


class ComputeWingTankRelativeThicknessArray(om.ExplicitComponent):
    """
    Computes the relative thickness for each section of the tank. Assumes a linear variation from
    chord to tip.
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

        self.add_input("data:geometry:wing:root:thickness_ratio", val=np.nan)
        self.add_input("data:geometry:wing:tip:thickness_ratio", val=np.nan)
        self.add_input("data:geometry:wing:root:y", val=np.nan, units="m")
        self.add_input("data:geometry:wing:tip:y", val=np.nan, units="m")
        self.add_input(
            "data:geometry:propulsion:tank:y_array",
            units="m",
            shape=nb_point_wing,
            val=np.nan,
        )

        self.add_output(
            "data:geometry:propulsion:tank:relative_thickness_array",
            shape=nb_point_wing,
            val=np.full(nb_point_wing, 0.15),
        )

        self.declare_partials(
            of="data:geometry:propulsion:tank:relative_thickness_array",
            wrt=[
                "data:geometry:wing:root:thickness_ratio",
                "data:geometry:wing:tip:thickness_ratio",
                "data:geometry:wing:root:y",
                "data:geometry:wing:tip:y",
            ],
            method="exact",
            rows=np.arange(nb_point_wing),
            cols=np.zeros(nb_point_wing),
        )
        self.declare_partials(
            of="data:geometry:propulsion:tank:relative_thickness_array",
            wrt="data:geometry:propulsion:tank:y_array",
            method="exact",
            rows=np.arange(nb_point_wing),
            cols=np.arange(nb_point_wing),
        )

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute, not all arguments are used
    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        root_tc = inputs["data:geometry:wing:root:thickness_ratio"]
        tip_tc = inputs["data:geometry:wing:tip:thickness_ratio"]
        root_y = inputs["data:geometry:wing:root:y"]
        tip_y = inputs["data:geometry:wing:tip:y"]

        y_array = inputs["data:geometry:propulsion:tank:y_array"]

        relative_thickness_array = np.where(
            y_array < root_y,
            np.full_like(y_array, root_tc),
            root_tc + y_array * (tip_tc - root_tc) / (tip_y - root_y),
        )

        outputs["data:geometry:propulsion:tank:relative_thickness_array"] = relative_thickness_array

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute_partials, not all arguments are used
    def compute_partials(self, inputs, partials, discrete_inputs=None):
        root_tc = inputs["data:geometry:wing:root:thickness_ratio"]
        tip_tc = inputs["data:geometry:wing:tip:thickness_ratio"]
        root_y = inputs["data:geometry:wing:root:y"]
        tip_y = inputs["data:geometry:wing:tip:y"]

        y_array = inputs["data:geometry:propulsion:tank:y_array"]

        partials[
            "data:geometry:propulsion:tank:relative_thickness_array",
            "data:geometry:wing:root:thickness_ratio",
        ] = np.where(
            y_array < root_y,
            np.full_like(y_array, 1e-6),
            1.0 - y_array / (tip_y - root_y),
        )
        partials[
            "data:geometry:propulsion:tank:relative_thickness_array",
            "data:geometry:wing:tip:thickness_ratio",
        ] = np.where(y_array < root_y, np.full_like(y_array, 1e-6), y_array / (tip_y - root_y))

        partials[
            "data:geometry:propulsion:tank:relative_thickness_array",
            "data:geometry:wing:root:y",
        ] = np.where(
            y_array < root_y,
            np.full_like(y_array, 1e-6),
            y_array * (tip_tc - root_tc) / (tip_y - root_y) ** 2.0,
        )
        partials[
            "data:geometry:propulsion:tank:relative_thickness_array",
            "data:geometry:wing:tip:y",
        ] = np.where(
            y_array < root_y,
            np.full_like(y_array, 1e-6),
            -y_array * (tip_tc - root_tc) / (tip_y - root_y) ** 2.0,
        )

        partials[
            "data:geometry:propulsion:tank:relative_thickness_array",
            "data:geometry:propulsion:tank:y_array",
        ] = np.where(
            y_array < root_y,
            np.full_like(y_array, 1e-6),
            np.full_like(y_array, (tip_tc - root_tc) / (tip_y - root_y)),
        )
