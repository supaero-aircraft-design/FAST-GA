"""Estimation of wing ToC."""
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

from openmdao.core.group import Group
from openmdao.core.explicitcomponent import ExplicitComponent

import fastoad.api as oad

from ..constants import SUBMODEL_WING_THICKNESS_RATIO


# TODO: computes relative thickness and generates profiles --> decompose
@oad.RegisterSubmodel(
    SUBMODEL_WING_THICKNESS_RATIO, "fastga.submodel.geometry.wing.thickness_ratio.legacy"
)
class ComputeWingToc(Group):
    # TODO: Document hypothesis. Cite sources
    """Wing ToC estimation."""

    def setup(self):

        self.add_subsystem("comp_wing_toc_root", ComputeWingTocRoot(), promotes=["*"])
        self.add_subsystem("comp_wing_toc_kink", ComputeWingTocKink(), promotes=["*"])
        self.add_subsystem("comp_wing_toc_tip", ComputeWingTocTip(), promotes=["*"])


class ComputeWingTocRoot(ExplicitComponent):
    """Wing root ToC estimation."""

    def setup(self):

        self.add_input("data:geometry:wing:thickness_ratio", val=np.nan)

        self.add_output("data:geometry:wing:root:thickness_ratio")

        self.declare_partials("*", "*", val=1.24)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        el_aero = inputs["data:geometry:wing:thickness_ratio"]

        el_emp = 1.24 * el_aero

        outputs["data:geometry:wing:root:thickness_ratio"] = el_emp


class ComputeWingTocKink(ExplicitComponent):
    """Wing kink ToC estimation."""

    def setup(self):

        self.add_input("data:geometry:wing:thickness_ratio", val=np.nan)

        self.add_output("data:geometry:wing:kink:thickness_ratio")

        self.declare_partials("*", "*", val=0.94)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        el_aero = inputs["data:geometry:wing:thickness_ratio"]

        el_break = 0.94 * el_aero

        outputs["data:geometry:wing:kink:thickness_ratio"] = el_break


class ComputeWingTocTip(ExplicitComponent):
    """Wing tip ToC estimation."""

    def setup(self):

        self.add_input("data:geometry:wing:thickness_ratio", val=np.nan)

        self.add_output("data:geometry:wing:tip:thickness_ratio")

        self.declare_partials("*", "*", val=0.86)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        el_aero = inputs["data:geometry:wing:thickness_ratio"]

        el_ext = 0.86 * el_aero

        outputs["data:geometry:wing:tip:thickness_ratio"] = el_ext
