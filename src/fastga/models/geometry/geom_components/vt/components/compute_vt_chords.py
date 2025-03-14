"""
Python module for vertical tail chords and span calculations, part of the vertical tail geometry.
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

from ..constants import SERVICE_VT_CHORD, SUBMODEL_VT_CHORD_LEGACY


@oad.RegisterSubmodel(SERVICE_VT_CHORD, SUBMODEL_VT_CHORD_LEGACY)
class ComputeVTChords(om.ExplicitComponent):
    """Vertical tail chords and span estimation."""

    # pylint: disable=missing-function-docstring
    # Overriding OpenMDAO setup
    def setup(self):
        self.add_input("data:geometry:vertical_tail:aspect_ratio", val=np.nan)
        self.add_input("data:geometry:vertical_tail:area", val=np.nan, units="m**2")
        self.add_input("data:geometry:vertical_tail:taper_ratio", val=np.nan)

        self.add_output("data:geometry:vertical_tail:span", units="m")
        self.add_output("data:geometry:vertical_tail:root:chord", units="m")
        self.add_output("data:geometry:vertical_tail:tip:chord", units="m")

        self.declare_partials(
            "data:geometry:vertical_tail:span",
            ["data:geometry:vertical_tail:aspect_ratio", "data:geometry:vertical_tail:area"],
            method="exact",
        )
        self.declare_partials("data:geometry:vertical_tail:root:chord", "*", method="exact")
        self.declare_partials("data:geometry:vertical_tail:tip:chord", "*", method="exact")

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute, not all arguments are used
    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        aspect_ratio_vt = inputs["data:geometry:vertical_tail:aspect_ratio"]
        s_v = inputs["data:geometry:vertical_tail:area"]
        taper_vt = inputs["data:geometry:vertical_tail:taper_ratio"]

        b_v = np.sqrt(max(aspect_ratio_vt * s_v, 0.1))
        # !!!: to avoid 0 division if s_v initialised to 0
        root_chord = s_v * 2.0 / (1.0 + taper_vt) / b_v
        tip_chord = root_chord * taper_vt

        outputs["data:geometry:vertical_tail:span"] = b_v
        outputs["data:geometry:vertical_tail:root:chord"] = root_chord
        outputs["data:geometry:vertical_tail:tip:chord"] = tip_chord

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute_partials, not all arguments are used
    def compute_partials(self, inputs, partials, discrete_inputs=None):
        s_v = inputs["data:geometry:vertical_tail:area"]
        taper_vt = inputs["data:geometry:vertical_tail:taper_ratio"]
        aspect_ratio_vt = inputs["data:geometry:vertical_tail:aspect_ratio"]

        if aspect_ratio_vt * s_v < 0.1:
            partials["data:geometry:vertical_tail:span", "data:geometry:vertical_tail:area"] = 0.0
            partials[
                "data:geometry:vertical_tail:span", "data:geometry:vertical_tail:aspect_ratio"
            ] = 0.0
        else:
            partials["data:geometry:vertical_tail:span", "data:geometry:vertical_tail:area"] = (
                np.sqrt(aspect_ratio_vt) / (2.0 * np.sqrt(s_v))
            )
            partials[
                "data:geometry:vertical_tail:span", "data:geometry:vertical_tail:aspect_ratio"
            ] = np.sqrt(s_v) / (2.0 * np.sqrt(aspect_ratio_vt))

        partials["data:geometry:vertical_tail:root:chord", "data:geometry:vertical_tail:area"] = (
            1.0 / (2.0 * np.sqrt(s_v * aspect_ratio_vt)) * 2.0 / (1.0 + taper_vt)
        )
        partials[
            "data:geometry:vertical_tail:root:chord", "data:geometry:vertical_tail:aspect_ratio"
        ] = -0.5 * np.sqrt(s_v / aspect_ratio_vt**3.0) * 2.0 / (1.0 + taper_vt)
        partials[
            "data:geometry:vertical_tail:root:chord", "data:geometry:vertical_tail:taper_ratio"
        ] = -np.sqrt(s_v / aspect_ratio_vt) * 2.0 / (1.0 + taper_vt) ** 2.0

        partials["data:geometry:vertical_tail:tip:chord", "data:geometry:vertical_tail:area"] = (
            1.0 / (2.0 * np.sqrt(s_v * aspect_ratio_vt)) * 2.0 * taper_vt / (1.0 + taper_vt)
        )
        partials[
            "data:geometry:vertical_tail:tip:chord", "data:geometry:vertical_tail:aspect_ratio"
        ] = -0.5 * np.sqrt(s_v / aspect_ratio_vt**3.0) * 2.0 * taper_vt / (1.0 + taper_vt)
        partials[
            "data:geometry:vertical_tail:tip:chord", "data:geometry:vertical_tail:taper_ratio"
        ] = np.sqrt(s_v / aspect_ratio_vt) * 2.0 / (1.0 + taper_vt) ** 2.0
