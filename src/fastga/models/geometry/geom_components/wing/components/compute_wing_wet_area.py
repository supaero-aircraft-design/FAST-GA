"""Estimation of wing wet area."""
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

from ..constants import SUBMODEL_WING_WET_AREA


@oad.RegisterSubmodel(SUBMODEL_WING_WET_AREA, "fastga.submodel.geometry.wing.wet_area.legacy")
class ComputeWingWetArea(Group):
    # TODO: Document equations. Cite sources
    """Wing wet and outer area estimation."""

    def setup(self):

        self.add_subsystem("comp_wing_outer_area", ComputeOuterArea(), promotes=["*"])
        self.add_subsystem("comp_wing_wet_area", ComputeWetArea(), promotes=["*"])


class ComputeOuterArea(ExplicitComponent):
    """Wing wet area estimation."""

    def setup(self):

        self.add_input("data:geometry:wing:root:virtual_chord", val=np.nan, units="m")
        self.add_input("data:geometry:wing:area", val=np.nan, units="m**2")
        self.add_input("data:geometry:fuselage:maximum_width", val=np.nan, units="m")

        self.add_output("data:geometry:wing:outer_area", units="m**2")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        wing_area = inputs["data:geometry:wing:area"]
        l1_wing = inputs["data:geometry:wing:root:virtual_chord"]
        y1_wing = inputs["data:geometry:fuselage:maximum_width"] / 2

        s_pf = wing_area - 2 * l1_wing * y1_wing

        outputs["data:geometry:wing:outer_area"] = s_pf


class ComputeWetArea(ExplicitComponent):
    """Wing outer area estimation based on Gudmunnson k_b (pag 707)."""

    def setup(self):

        self.add_input("data:geometry:wing:root:virtual_chord", val=np.nan, units="m")
        self.add_input("data:geometry:wing:area", val=np.nan, units="m**2")
        self.add_input("data:geometry:fuselage:maximum_width", val=np.nan, units="m")

        self.add_output("data:geometry:wing:wet_area", units="m**2")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        wing_area = inputs["data:geometry:wing:area"]
        l1_wing = inputs["data:geometry:wing:root:virtual_chord"]
        width_max = inputs["data:geometry:fuselage:maximum_width"]

        wet_area_wing = 2 * (wing_area - width_max * l1_wing) * 1.07

        outputs["data:geometry:wing:wet_area"] = wet_area_wing
