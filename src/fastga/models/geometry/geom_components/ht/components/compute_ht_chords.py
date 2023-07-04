"""
    Estimation of horizontal tail chords and span
"""
#  This file is part of FAST-OAD_CS23 : A framework for rapid Overall Aircraft Design
#  Copyright (C) 2022  ONERA & ISAE-SUPAERO
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

from openmdao.core.explicitcomponent import ExplicitComponent
from openmdao.core.group import Group
import fastoad.api as oad

from ..constants import SUBMODEL_HT_CHORD


@oad.RegisterSubmodel(SUBMODEL_HT_CHORD, "fastga.submodel.geometry.horizontal_tail.chord.legacy")
class ComputeHTChord(Group):
    # TODO: Document equations. Cite sources
    """Horizontal tail chords and span estimation"""

    def setup(self):

        self.add_subsystem("comp_HT_span", ComputeHTSpan(), promotes=["*"])
        self.add_subsystem("comp_HT_root_chord", ComputeHTRootChord(), promotes=["*"])
        self.add_subsystem("comp_HT_tip_chord", ComputeHTTipChord(), promotes=["*"])


class ComputeHTSpan(ExplicitComponent):
    '''Span estimation of horizontal tail'''

    def setup(self):

        self.add_input("data:geometry:horizontal_tail:area", val=np.nan, units="m**2")
        self.add_input("data:geometry:horizontal_tail:aspect_ratio", val=np.nan)

        self.add_output("data:geometry:horizontal_tail:span", units="m")

        self.declare_partials(
            of="data:geometry:horizontal_tail:span",
            wrt=[
                "data:geometry:horizontal_tail:area",
                "data:geometry:horizontal_tail:aspect_ratio",
            ],
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        s_h = inputs["data:geometry:horizontal_tail:area"]
        aspect_ratio = inputs["data:geometry:horizontal_tail:aspect_ratio"]

        # Give minimum value to avoid 0 division later if s_h initialised to 0

        b_h = np.sqrt(
            max(aspect_ratio * s_h, 0.1)
        )  

        outputs["data:geometry:horizontal_tail:span"] = b_h

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        
        s_h = inputs["data:geometry:horizontal_tail:area"]
        aspect_ratio = inputs["data:geometry:horizontal_tail:aspect_ratio"]
        
        if aspect_ratio * s_h < 0.1:
            partials["data:geometry:horizontal_tail:span", "data:geometry:horizontal_tail:area"] = 0
            partials[
                "data:geometry:horizontal_tail:span", "data:geometry:horizontal_tail:aspect_ratio"
            ] = 0
        else:
            partials[
                "data:geometry:horizontal_tail:span", "data:geometry:horizontal_tail:area"
            ] = np.sqrt(aspect_ratio) / (2.0 * np.sqrt(s_h))
            partials[
                "data:geometry:horizontal_tail:span", "data:geometry:horizontal_tail:aspect_ratio"
            ] = np.sqrt(s_h) / (2.0 * np.sqrt(aspect_ratio))

        
class ComputeHTRootChord(ExplicitComponent):
    '''Root chord estimation of horizontal tail'''

    def setup(self):

        self.add_input("data:geometry:horizontal_tail:area", val=np.nan, units="m**2")
        self.add_input("data:geometry:horizontal_tail:taper_ratio", val=np.nan)
        self.add_input("data:geometry:horizontal_tail:span", units="m")

        self.add_output("data:geometry:horizontal_tail:root:chord", units="m")

        self.declare_partials("*", "*", method="fd")
        
    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        
        s_h = inputs["data:geometry:horizontal_tail:area"]
        taper_ht = inputs["data:geometry:horizontal_tail:taper_ratio"]
        b_h = inputs["data:geometry:horizontal_tail:span"]

        root_chord = s_h * 2 / (1 + taper_ht) / b_h

        outputs["data:geometry:horizontal_tail:root:chord"] = root_chord

    
class ComputeHTTipChord(ExplicitComponent):
    '''Tip chord estimation of horizontal tail'''

    def setup(self):

        self.add_input("data:geometry:horizontal_tail:taper_ratio", val=np.nan)
        self.add_input("data:geometry:horizontal_tail:root:chord", units="m")

        self.add_output("data:geometry:horizontal_tail:tip:chord", units="m")
        
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        
        taper_ht = inputs["data:geometry:horizontal_tail:taper_ratio"]
        root_chord = inputs["data:geometry:horizontal_tail:root:chord"]

        tip_chord = root_chord * taper_ht

        outputs["data:geometry:horizontal_tail:tip:chord"] = tip_chord
        