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
import openmdao.api as om

import fastoad.api as oad

from ..constants import SUBMODEL_WING_THICKNESS_RATIO


# TODO: computes relative thickness and generates profiles --> decompose
@oad.RegisterSubmodel(
    SUBMODEL_WING_THICKNESS_RATIO, "fastga.submodel.geometry.wing.thickness_ratio.legacy"
)
class ComputeWingToc(om.ExplicitComponent):
    # TODO: Document hypothesis. Cite sources
    """Wing ToC estimation."""

    def setup(self):

        self.add_input("data:geometry:wing:thickness_ratio", val=np.nan)

        self.add_output("data:geometry:wing:root:thickness_ratio")
        self.add_output("data:geometry:wing:kink:thickness_ratio")
        self.add_output("data:geometry:wing:tip:thickness_ratio")

        self.declare_partials("", "", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        el_aero = inputs["data:geometry:wing:thickness_ratio"]

        el_emp = 1.24 * el_aero
        el_break = 0.94 * el_aero
        el_ext = 0.86 * el_aero

        outputs["data:geometry:wing:root:thickness_ratio"] = el_emp
        outputs["data:geometry:wing:kink:thickness_ratio"] = el_break
        outputs["data:geometry:wing:tip:thickness_ratio"] = el_ext


@oad.RegisterSubmodel(
    SUBMODEL_WING_THICKNESS_RATIO, "fastga.submodel.geometry.wing.thickness_ratio.electric"
)
class ComputeWingTocElectric(om.ExplicitComponent):
    # TODO: Document hypothesis. Cite sources
    """Wing ToC estimation."""

    def setup(self):

        self.add_input("data:geometry:wing:thickness_ratio", val=np.nan)
        self.add_input("data:geometry:wing:outer_area", val=np.nan, units="m**2")
        self.add_input("data:geometry:propulsion:battery:volume", val=np.nan, units="m**3")
        self.add_input("data:geometry:wing:root:chord", val=np.nan, units="m")
        self.add_input("data:geometry:wing:tip:chord", val=np.nan, units="m")
        self.add_input("data:geometry:wing:kink:chord", val=np.nan, units="m")

        self.add_output("data:geometry:wing:root:thickness_ratio")
        self.add_output("data:geometry:wing:kink:thickness_ratio")
        self.add_output("data:geometry:wing:tip:thickness_ratio")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        wing_area = inputs["data:geometry:wing:outer_area"]
        battery_volume = inputs["data:geometry:propulsion:battery:volume"]
        wing_root_chord = inputs["data:geometry:wing:root:chord"]
        wing_tip_chord = inputs["data:geometry:wing:tip:chord"]
        wing_kink_chord = inputs["data:geometry:wing:tip:chord"]

        avg_wing_thickness = battery_volume/wing_area

        root_thickness = 1.24 * avg_wing_thickness
        tip_thickness = 0.86 * avg_wing_thickness
        break_thickness = 0.94 * avg_wing_thickness

        el_emp = root_thickness / wing_root_chord
        el_ext = tip_thickness / wing_tip_chord
        el_break = break_thickness / wing_kink_chord

        outputs["data:geometry:wing:root:thickness_ratio"] = el_emp
        outputs["data:geometry:wing:kink:thickness_ratio"] = el_break
        outputs["data:geometry:wing:tip:thickness_ratio"] = el_ext
