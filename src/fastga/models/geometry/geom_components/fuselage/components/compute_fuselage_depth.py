"""Estimation of fuselage average depth around the vertical tail."""
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

from ..constants import SERVICE_FUSELAGE_DEPTH, SUBMODEL_FUSELAGE_DEPTH_LEGACY

oad.RegisterSubmodel.active_models[SERVICE_FUSELAGE_DEPTH] = SUBMODEL_FUSELAGE_DEPTH_LEGACY


@oad.RegisterSubmodel(SERVICE_FUSELAGE_DEPTH, SUBMODEL_FUSELAGE_DEPTH_LEGACY)
class ComputeFuselageDepth(om.ExplicitComponent):
    """
    Fuselage average depth at the vertical tail location computation. Based on geometric
    consideration assuming the fuselage is cylindrical at the center section and the average
    diameter reduces linearly to the end of the aircraft.
    """

    # pylint: disable=missing-function-docstring
    # Overriding OpenMDAO setup
    def setup(self):
        self.add_input("data:geometry:fuselage:maximum_width", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:maximum_height", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:rear_length", val=np.nan, units="m")
        self.add_input("data:geometry:vertical_tail:root:chord", val=np.nan, units="m")

        self.add_output(
            "data:geometry:fuselage:average_depth",
            units="m",
            desc="Average fuselage depth at the vertical tail location",
        )

        self.declare_partials("*", "*", method="exact")

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute, not all arguments are used
    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        b_f = inputs["data:geometry:fuselage:maximum_width"]
        h_f = inputs["data:geometry:fuselage:maximum_height"]
        lar = inputs["data:geometry:fuselage:rear_length"]
        root_chord_vt = inputs["data:geometry:vertical_tail:root:chord"]

        # Using the simple geometric description
        avg_fus_depth = np.sqrt(b_f * h_f) * root_chord_vt / (2 * lar)

        outputs["data:geometry:fuselage:average_depth"] = avg_fus_depth

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute_partials, not all arguments are used
    def compute_partials(self, inputs, partials, discrete_inputs=None):
        b_f = inputs["data:geometry:fuselage:maximum_width"]
        h_f = inputs["data:geometry:fuselage:maximum_height"]
        lar = inputs["data:geometry:fuselage:rear_length"]
        root_chord_vt = inputs["data:geometry:vertical_tail:root:chord"]

        partials["data:geometry:fuselage:average_depth", "data:geometry:fuselage:maximum_width"] = (
            np.sqrt(h_f / b_f) * root_chord_vt / (2 * lar) / 2
        )
        partials[
            "data:geometry:fuselage:average_depth", "data:geometry:fuselage:maximum_height"
        ] = np.sqrt(b_f / h_f) * root_chord_vt / (2 * lar) / 2
        partials[
            "data:geometry:fuselage:average_depth", "data:geometry:vertical_tail:root:chord"
        ] = np.sqrt(b_f * h_f) / (2 * lar)
        partials["data:geometry:fuselage:average_depth", "data:geometry:fuselage:rear_length"] = (
            -np.sqrt(b_f * h_f) * root_chord_vt / (2 * lar**2)
        )
