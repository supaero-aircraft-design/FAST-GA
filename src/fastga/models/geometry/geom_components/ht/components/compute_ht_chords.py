"""
Python module for horizontal tail chords and span calculations, part of the horizontal tail
geometry.
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

from ..constants import SERVICE_HT_CHORD, SUBMODEL_HT_CHORD_LEGACY


@oad.RegisterSubmodel(SERVICE_HT_CHORD, SUBMODEL_HT_CHORD_LEGACY)
class ComputeHTChord(om.ExplicitComponent):
    # TODO: Document equations. Cite sources
    """Horizontal tail chords and span estimation."""

    # pylint: disable=missing-function-docstring
    # Overriding OpenMDAO setup
    def setup(self):
        self.add_input("data:geometry:horizontal_tail:area", val=np.nan, units="m**2")
        self.add_input("data:geometry:horizontal_tail:taper_ratio", val=np.nan)
        self.add_input("data:geometry:horizontal_tail:aspect_ratio", val=np.nan)

        self.add_output("data:geometry:horizontal_tail:span", units="m")
        self.add_output("data:geometry:horizontal_tail:root:chord", units="m")
        self.add_output("data:geometry:horizontal_tail:tip:chord", units="m")

        self.declare_partials(
            of="data:geometry:horizontal_tail:span",
            wrt=[
                "data:geometry:horizontal_tail:area",
                "data:geometry:horizontal_tail:aspect_ratio",
            ],
            method="exact",
        )
        self.declare_partials(
            of="data:geometry:horizontal_tail:root:chord",
            wrt="*",
            method="exact",
        )
        self.declare_partials(
            of="data:geometry:horizontal_tail:tip:chord",
            wrt="*",
            method="exact",
        )

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute, not all arguments are used
    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        s_h = inputs["data:geometry:horizontal_tail:area"]
        taper_ht = inputs["data:geometry:horizontal_tail:taper_ratio"]
        aspect_ratio_ht = inputs["data:geometry:horizontal_tail:aspect_ratio"]

        b_h = np.sqrt(max(aspect_ratio_ht * s_h, 0.1))
        # !!!: to avoid 0 division if s_h initialised to 0
        root_chord = s_h * 2.0 / (1.0 + taper_ht) / b_h
        tip_chord = root_chord * taper_ht

        outputs["data:geometry:horizontal_tail:span"] = b_h
        outputs["data:geometry:horizontal_tail:root:chord"] = root_chord
        outputs["data:geometry:horizontal_tail:tip:chord"] = tip_chord

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute_partials, not all arguments are used
    def compute_partials(self, inputs, partials, discrete_inputs=None):
        s_h = inputs["data:geometry:horizontal_tail:area"]
        taper_ht = inputs["data:geometry:horizontal_tail:taper_ratio"]
        aspect_ratio_ht = inputs["data:geometry:horizontal_tail:aspect_ratio"]

        if aspect_ratio_ht * s_h < 0.1:
            partials["data:geometry:horizontal_tail:span", "data:geometry:horizontal_tail:area"] = (
                0.0
            )
            partials[
                "data:geometry:horizontal_tail:span", "data:geometry:horizontal_tail:aspect_ratio"
            ] = 0.0
        else:
            partials["data:geometry:horizontal_tail:span", "data:geometry:horizontal_tail:area"] = (
                np.sqrt(aspect_ratio_ht) / (2.0 * np.sqrt(s_h))
            )
            partials[
                "data:geometry:horizontal_tail:span", "data:geometry:horizontal_tail:aspect_ratio"
            ] = np.sqrt(s_h) / (2.0 * np.sqrt(aspect_ratio_ht))

        partials[
            "data:geometry:horizontal_tail:root:chord", "data:geometry:horizontal_tail:area"
        ] = 1.0 / (2.0 * np.sqrt(s_h * aspect_ratio_ht)) * 2.0 / (1.0 + taper_ht)
        partials[
            "data:geometry:horizontal_tail:root:chord", "data:geometry:horizontal_tail:aspect_ratio"
        ] = -0.5 * np.sqrt(s_h / aspect_ratio_ht**3.0) * 2.0 / (1.0 + taper_ht)
        partials[
            "data:geometry:horizontal_tail:root:chord", "data:geometry:horizontal_tail:taper_ratio"
        ] = -np.sqrt(s_h / aspect_ratio_ht) * 2.0 / (1.0 + taper_ht) ** 2.0

        partials[
            "data:geometry:horizontal_tail:tip:chord", "data:geometry:horizontal_tail:area"
        ] = 1.0 / (2.0 * np.sqrt(s_h * aspect_ratio_ht)) * 2.0 * taper_ht / (1.0 + taper_ht)
        partials[
            "data:geometry:horizontal_tail:tip:chord", "data:geometry:horizontal_tail:aspect_ratio"
        ] = -0.5 * np.sqrt(s_h / aspect_ratio_ht**3.0) * 2.0 * taper_ht / (1.0 + taper_ht)
        partials[
            "data:geometry:horizontal_tail:tip:chord", "data:geometry:horizontal_tail:taper_ratio"
        ] = np.sqrt(s_h / aspect_ratio_ht) * 2.0 / (1.0 + taper_ht) ** 2.0
