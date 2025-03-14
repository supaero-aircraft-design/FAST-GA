"""
Python module for landing gear geometry calculation, part of the geometry component.
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
import fastoad.api as oad

from ...constants import SERVICE_LANDING_GEAR_GEOMETRY, SUBMODEL_LANDING_GEAR_GEOMETRY_LEGACY


@oad.RegisterSubmodel(SERVICE_LANDING_GEAR_GEOMETRY, SUBMODEL_LANDING_GEAR_GEOMETRY_LEGACY)
class ComputeLGGeometry(om.ExplicitComponent):
    # TODO: Document equations. Cite sources
    """
    Landing gears geometry estimation. Position along the span is based on aircraft pictures
    analysis.
    """

    # pylint: disable=missing-function-docstring
    # Overriding OpenMDAO setup
    def setup(self):
        self.add_input("data:geometry:propeller:diameter", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:maximum_width", val=np.nan, units="m")

        self.add_output("data:geometry:landing_gear:height", units="m")
        self.add_output("data:geometry:landing_gear:y", units="m")

        self.declare_partials(
            "data:geometry:landing_gear:height", "data:geometry:propeller:diameter", val=0.41
        )
        self.declare_partials(
            "data:geometry:landing_gear:y", "data:geometry:propeller:diameter", val=0.492
        )
        self.declare_partials(
            "data:geometry:landing_gear:y", "data:geometry:fuselage:maximum_width", val=0.5
        )

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute, not all arguments are used
    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        prop_dia = inputs["data:geometry:propeller:diameter"]
        fuselage_max_width = inputs["data:geometry:fuselage:maximum_width"]
        lg_height = 0.41 * prop_dia
        y_lg = fuselage_max_width / 2.0 + lg_height * 1.2

        outputs["data:geometry:landing_gear:height"] = lg_height
        outputs["data:geometry:landing_gear:y"] = y_lg
