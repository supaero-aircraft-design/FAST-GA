"""
Python module for fuselage volume calculation, part of the fuselage geometry.
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

from ..constants import SERVICE_FUSELAGE_VOLUME, SUBMODEL_FUSELAGE_VOLUME_LEGACY

oad.RegisterSubmodel.active_models[SERVICE_FUSELAGE_VOLUME] = SUBMODEL_FUSELAGE_VOLUME_LEGACY


@oad.RegisterSubmodel(SERVICE_FUSELAGE_VOLUME, SUBMODEL_FUSELAGE_VOLUME_LEGACY)
class ComputeFuselageVolume(om.ExplicitComponent):
    """
    Fuselage volume computation. Based on geometric consideration assuming the fuselage is
    cylindrical at the center section and conical at the front and back.
    """

    # pylint: disable=missing-function-docstring
    # Overriding OpenMDAO setup
    def setup(self):
        self.add_input("data:geometry:fuselage:maximum_width", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:maximum_height", val=np.nan, units="m")
        self.add_input("data:geometry:cabin:length", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:front_length", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:rear_length", val=np.nan, units="m")

        self.add_output(
            "data:geometry:fuselage:volume",
            units="m**3",
            desc="Volume of the fuselage",
        )

        self.declare_partials("*", "*", method="exact")

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute, not all arguments are used
    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        b_f = inputs["data:geometry:fuselage:maximum_width"]
        h_f = inputs["data:geometry:fuselage:maximum_height"]
        l_c = inputs["data:geometry:cabin:length"]
        lar = inputs["data:geometry:fuselage:rear_length"]
        lav = inputs["data:geometry:fuselage:front_length"]

        l_f = np.sqrt(b_f * h_f)
        # estimation of fuselage volume
        volume_fus = np.pi * l_f**2.0 / 4.0 * (0.7 * lav + 0.5 * lar + l_c)

        outputs["data:geometry:fuselage:volume"] = volume_fus

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute_partials, not all arguments are used
    def compute_partials(self, inputs, partials, discrete_inputs=None):
        b_f = inputs["data:geometry:fuselage:maximum_width"]
        h_f = inputs["data:geometry:fuselage:maximum_height"]
        l_c = inputs["data:geometry:cabin:length"]
        lar = inputs["data:geometry:fuselage:rear_length"]
        lav = inputs["data:geometry:fuselage:front_length"]

        partials["data:geometry:fuselage:volume", "data:geometry:fuselage:maximum_width"] = (
            np.pi * h_f / 4.0 * (0.7 * lav + 0.5 * lar + l_c)
        )
        partials["data:geometry:fuselage:volume", "data:geometry:fuselage:maximum_height"] = (
            np.pi * b_f / 4.0 * (0.7 * lav + 0.5 * lar + l_c)
        )
        partials["data:geometry:fuselage:volume", "data:geometry:cabin:length"] = (
            np.pi * b_f * h_f / 4.0
        )
        partials["data:geometry:fuselage:volume", "data:geometry:fuselage:rear_length"] = (
            0.5 * np.pi * b_f * h_f / 4.0
        )
        partials["data:geometry:fuselage:volume", "data:geometry:fuselage:front_length"] = (
            0.7 * np.pi * b_f * h_f / 4.0
        )
