"""
    Estimation of horizontal tail chords and span
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
import math

from openmdao.core.explicitcomponent import ExplicitComponent


class ComputeHTChord(ExplicitComponent):
    # TODO: Document equations. Cite sources
    """ Horizontal tail chords and span estimation """

    def setup(self):
        self.add_input("data:geometry:horizontal_tail:area", val=np.nan, units="m**2")
        self.add_input("data:geometry:horizontal_tail:taper_ratio", val=np.nan)
        self.add_input("data:geometry:horizontal_tail:sweep_25", val=np.nan, units="deg")
        self.add_input("data:geometry:has_T_tail", val=np.nan)
        
        self.add_output("data:geometry:horizontal_tail:span", units="m")
        self.add_output("data:geometry:horizontal_tail:root:chord", units="m")
        self.add_output("data:geometry:horizontal_tail:tip:chord", units="m")
        self.add_output("data:geometry:horizontal_tail:aspect_ratio")
        
        self.declare_partials(
            "data:geometry:horizontal_tail:span",
            [
                "data:geometry:horizontal_tail:area",
                "data:geometry:horizontal_tail:sweep_25",
            ],
            method="fd",
        )

        self.declare_partials(
            "data:geometry:horizontal_tail:root:chord",
            [
                "data:geometry:horizontal_tail:area",
                "data:geometry:horizontal_tail:taper_ratio",
                "data:geometry:horizontal_tail:sweep_25",
            ],
            method="fd",
        )

        self.declare_partials(
            "data:geometry:horizontal_tail:tip:chord",
            [
                "data:geometry:horizontal_tail:area",
                "data:geometry:horizontal_tail:taper_ratio",
                "data:geometry:horizontal_tail:sweep_25",
            ],
            method="fd",
        )

        self.declare_partials(
            "data:geometry:horizontal_tail:aspect_ratio",
            [
                "data:geometry:horizontal_tail:sweep_25",
            ],
            method="fd",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        s_h = inputs["data:geometry:horizontal_tail:area"]
        taper_ht = inputs["data:geometry:horizontal_tail:taper_ratio"]
        sweep_25 = inputs["data:geometry:horizontal_tail:sweep_25"]
        tail_conf = inputs["data:geometry:has_T_tail"]

        if tail_conf == 1.0:
            aspect_ratio = 5.9 * math.cos(sweep_25 / 180.0 * math.pi)**2
        else:
            aspect_ratio = 5.5 * math.cos(sweep_25 / 180.0 * math.pi)**2
        b_h = np.sqrt(max(aspect_ratio * s_h, 0.1))  # !!!: to avoid 0 division if s_h initialised to 0
        root_chord = s_h * 2 / (1 + taper_ht) / b_h
        tip_chord = root_chord * taper_ht

        outputs["data:geometry:horizontal_tail:span"] = b_h
        outputs["data:geometry:horizontal_tail:root:chord"] = root_chord
        outputs["data:geometry:horizontal_tail:tip:chord"] = tip_chord
        outputs["data:geometry:horizontal_tail:aspect_ratio"] = aspect_ratio
