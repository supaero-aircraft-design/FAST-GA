"""Estimation of wing Ys (sections span)."""
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

from ..constants import SUBMODEL_WING_SPAN


@oad.RegisterSubmodel(SUBMODEL_WING_SPAN, "fastga.submodel.geometry.wing.span.legacy")
class ComputeWingY(om.Group):
    # TODO: Document equations. Cite sources
    """Wing Ys estimation."""

    def setup(self):

        self.add_subsystem("comp_wing_span", ComputeWingSpan(), promotes=["*"])
        self.add_subsystem("comp_wing_root_y", ComputeWingRootY(), promotes=["*"])
        self.add_subsystem("comp_wing_tip_y", ComputeWingTipY(), promotes=["*"])
        self.add_subsystem("comp_wing_kink_y", ComputeWingKinkY(), promotes=["*"])


class ComputeWingSpan(om.ExplicitComponent):
    """Wing span estimation."""

    def setup(self):

        self.add_input("data:geometry:wing:aspect_ratio", val=np.nan)
        self.add_input("data:geometry:wing:area", val=np.nan, units="m**2")

        self.add_output("data:geometry:wing:span", units="m")

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        lambda_wing = inputs["data:geometry:wing:aspect_ratio"]
        wing_area = inputs["data:geometry:wing:area"]

        span = np.sqrt(lambda_wing * wing_area)

        outputs["data:geometry:wing:span"] = span

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        lambda_wing = inputs["data:geometry:wing:aspect_ratio"]
        wing_area = inputs["data:geometry:wing:area"]

        span = np.sqrt(lambda_wing * wing_area)

        partials["data:geometry:wing:span", "data:geometry:wing:aspect_ratio"] = (
            0.5 * wing_area / span
        )
        partials["data:geometry:wing:span", "data:geometry:wing:area"] = 0.5 * lambda_wing / span


class ComputeWingRootY(om.ExplicitComponent):
    """Wing root Ys estimation."""

    def setup(self):

        self.add_input("data:geometry:fuselage:maximum_width", val=np.nan, units="m")

        self.add_output("data:geometry:wing:root:y", units="m")

        self.declare_partials(of="*", wrt="*", val=0.5)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        width_max = inputs["data:geometry:fuselage:maximum_width"]

        y2_wing = width_max / 2.0

        outputs["data:geometry:wing:root:y"] = y2_wing


class ComputeWingTipY(om.ExplicitComponent):
    """Wing tip Ys estimation."""

    def setup(self):

        self.add_input("data:geometry:wing:span", val=np.nan, units="m")

        self.add_output("data:geometry:wing:tip:y", units="m")

        self.declare_partials(of="*", wrt="*", val=0.5)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        span = inputs["data:geometry:wing:span"]

        y4_wing = span / 2.0

        outputs["data:geometry:wing:tip:y"] = y4_wing


class ComputeWingKinkY(om.ExplicitComponent):
    """Wing kink Ys estimation."""

    def setup(self):

        self.add_input("data:geometry:wing:kink:span_ratio", val=0.5)
        self.add_input("data:geometry:wing:tip:y", val=np.nan, units="m")

        self.add_output("data:geometry:wing:kink:y", units="m")

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        wing_break = inputs["data:geometry:wing:kink:span_ratio"]
        y4_wing = inputs["data:geometry:wing:tip:y"]

        y3_wing = y4_wing * wing_break

        outputs["data:geometry:wing:kink:y"] = y3_wing

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        wing_break = inputs["data:geometry:wing:kink:span_ratio"]
        y4_wing = inputs["data:geometry:wing:tip:y"]

        partials["data:geometry:wing:kink:y", "data:geometry:wing:kink:span_ratio"] = y4_wing
        partials["data:geometry:wing:kink:y", "data:geometry:wing:tip:y"] = wing_break
