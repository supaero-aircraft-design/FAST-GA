"""Estimation of wing center of gravity."""
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
import math

from openmdao.core.explicitcomponent import ExplicitComponent

import fastoad.api as oad

from ..constants import SUBMODEL_WING_CG


@oad.RegisterSubmodel(SUBMODEL_WING_CG, "fastga.submodel.weight.cg.airframe.wing.legacy")
class ComputeWingCG(ExplicitComponent):
    """
    Wing center of gravity estimation

    Based on a statistical analysis. See :cite:`roskampart5:1985`.
    """

    def setup(self):

        self.add_input("data:geometry:wing:MAC:at25percent:x", val=np.nan, units="m")
        self.add_input("data:geometry:wing:MAC:leading_edge:x:local", val=np.nan, units="m")
        self.add_input("data:geometry:wing:MAC:length", val=np.nan, units="m")
        self.add_input("data:geometry:wing:sweep_25", val=np.nan, units="deg")
        self.add_input("data:geometry:wing:span", val=np.nan, units="m")
        self.add_input("data:geometry:flap:chord_ratio", val=0.2)
        self.add_input("data:geometry:wing:root:virtual_chord", val=np.nan, units="m")
        self.add_input("data:geometry:wing:tip:chord", val=np.nan, units="m")
        self.add_input("data:geometry:wing:root:y", val=np.nan, units="m")
        self.add_input("data:geometry:wing:tip:y", val=np.nan, units="m")

        self.add_output("data:weight:airframe:wing:CG:x", units="m")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        fa_length = inputs["data:geometry:wing:MAC:at25percent:x"]
        x0_wing = inputs["data:geometry:wing:MAC:leading_edge:x:local"]
        l0_wing = inputs["data:geometry:wing:MAC:length"]
        sweep_25 = inputs["data:geometry:wing:sweep_25"]
        span = inputs["data:geometry:wing:span"]
        l2_wing = inputs["data:geometry:wing:root:virtual_chord"]
        y2_wing = inputs["data:geometry:wing:root:y"]
        l4_wing = inputs["data:geometry:wing:tip:chord"]
        y4_wing = inputs["data:geometry:wing:tip:y"]

        if sweep_25 < 5.0:
            y_cg = 0.40 * span / 2.0

            if y_cg < y2_wing:
                chord_reduction = 0.0

            else:
                chord_reduction = (y_cg - y2_wing) / (y4_wing - y2_wing) * (l2_wing - l4_wing)

            chord_at_cg_pos = l2_wing - chord_reduction
            x_cg_wing_rel = 0.42 * chord_at_cg_pos + y_cg * math.tan(sweep_25 * math.pi / 180.0)

        else:
            y_cg = 0.35 * span / 2.0

            if y_cg < y2_wing:
                chord_reduction = 0.0

            else:
                chord_reduction = (y_cg - y2_wing) / (y4_wing - y2_wing) * (l2_wing - l4_wing)

            chord_at_cg_pos = l2_wing - chord_reduction
            # In the computation of the mfw, we assume that there is 30% of the chord between
            # front and rear spar, we will do the same assumption here. It was also assumed that
            # the front spar is at 30 % of the chord
            distance_between_spars = 0.30 * chord_at_cg_pos

            x_cg_wing_rel = (
                y_cg * math.tan(sweep_25 * math.pi / 180.0)
                + 0.30 * chord_at_cg_pos
                + 0.70 * distance_between_spars
            )

        x_cg_a1 = fa_length - 0.25 * l0_wing - x0_wing + x_cg_wing_rel

        outputs["data:weight:airframe:wing:CG:x"] = x_cg_a1
