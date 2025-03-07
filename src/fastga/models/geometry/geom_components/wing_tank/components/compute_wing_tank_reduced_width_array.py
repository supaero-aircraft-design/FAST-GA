"""
Python module for actual tank width computation class(es), part of the advanced MFW computation
method.
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


class ComputeWingTankReducedWidthArray(om.ExplicitComponent):
    """
    Compute the wing tank width reduction due to engine nacelle and landing gear.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.in_engine = None
        self.in_landing_gear = None

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
            "data:geometry:propulsion:tank:width_array",
            units="m",
            shape=nb_point_wing,
            val=np.nan,
        )
        self.add_input(
            "data:geometry:propulsion:tank:y_array",
            units="m",
            shape=nb_point_wing,
            val=np.nan,
        )

        self.add_input("data:geometry:wing:span", val=np.nan, units="m")
        self.add_input("data:geometry:propulsion:nacelle:width", val=np.nan, units="m")
        self.add_input("data:geometry:landing_gear:type", val=np.nan)
        self.add_input("data:geometry:landing_gear:y", val=np.nan, units="m")
        self.add_input("data:geometry:propulsion:engine:layout", val=np.nan)
        self.add_input(
            "data:geometry:propulsion:engine:y_ratio",
            shape_by_conn=True,
        )

        self.add_output(
            "data:geometry:propulsion:tank:reduced_width_array",
            units="m",
            shape=nb_point_wing,
            val=np.full(nb_point_wing, 0.2),
        )

        self.declare_partials(
            of="data:geometry:propulsion:tank:reduced_width_array",
            wrt="data:geometry:propulsion:tank:width_array",
            method="exact",
            rows=np.arange(nb_point_wing),
            cols=np.arange(nb_point_wing),
        )
        # It actually does depend on them as the formula says but for the sake of what we will do
        # it should not be necessary
        self.declare_partials(
            of="data:geometry:propulsion:tank:reduced_width_array",
            wrt=[
                "data:geometry:propulsion:tank:y_array",
                "data:geometry:propulsion:nacelle:width",
                "data:geometry:landing_gear:type",
                "data:geometry:landing_gear:y",
                "data:geometry:propulsion:engine:layout",
                "data:geometry:propulsion:engine:y_ratio",
                "data:geometry:wing:span",
            ],
            method="exact",
            val=0.0,
        )

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute, not all arguments are used
    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        nb_point_wing = self.options["number_points_wing_mfw"]

        self.in_engine = np.full(nb_point_wing, False)
        self.in_landing_gear = np.full(nb_point_wing, False)

        lg_type = inputs["data:geometry:landing_gear:type"]
        y_lg = inputs["data:geometry:landing_gear:y"]

        nacelle_width = inputs["data:geometry:propulsion:nacelle:width"]
        engine_config = inputs["data:geometry:propulsion:engine:layout"]

        y_ratio = inputs["data:geometry:propulsion:engine:y_ratio"]

        width_array = inputs["data:geometry:propulsion:tank:width_array"]
        y_array = inputs["data:geometry:propulsion:tank:y_array"]

        span = inputs["data:geometry:wing:span"]

        if engine_config == 1.0:
            for y_eng in y_ratio * span / 2.0:
                self.in_engine = np.where(
                    np.abs(y_array - y_eng) < nacelle_width / 2.0,
                    np.full_like(self.in_engine, True),
                    self.in_engine,
                )

        if lg_type == 1.0:
            self.in_landing_gear = np.where(
                y_array < y_lg,
                np.full_like(self.in_landing_gear, True),
                self.in_landing_gear,
            )

        # For now 50% size reduction in the fuel tank capacity due to the engine
        reduced_width_array = np.where(self.in_engine, width_array * 0.5, width_array)

        # For now 80% size reduction in the fuel tank capacity due to the landing gear
        reduced_width_array = np.where(
            self.in_landing_gear, reduced_width_array * 0.2, reduced_width_array
        )

        outputs["data:geometry:propulsion:tank:reduced_width_array"] = reduced_width_array

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute_partials, not all arguments are used
    def compute_partials(self, inputs, partials, discrete_inputs=None):
        partials_width = np.ones_like(inputs["data:geometry:propulsion:tank:width_array"])
        partials_width = np.where(self.in_engine, partials_width * 0.5, partials_width)
        partials_width = np.where(self.in_landing_gear, partials_width * 0.2, partials_width)

        partials[
            "data:geometry:propulsion:tank:reduced_width_array",
            "data:geometry:propulsion:tank:width_array",
        ] = partials_width
