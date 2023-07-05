"""Estimation of wing sweeps."""
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

from ..constants import SUBMODEL_WING_SWEEP


@oad.RegisterSubmodel(SUBMODEL_WING_SWEEP, "fastga.submodel.geometry.wing.sweep.legacy")
class ComputeWingSweep(Group):
    # TODO: Document equations. Cite sources
    """Wing sweeps estimation."""

    def setup(self):

        self.add_subsystem("comp_wing_sweep_0", ComputeWingSweep0(), promotes=["*"])
        self.add_subsystem("comp_wing_sweep_50", ComputeWingSweep50(), promotes=["*"])
        self.add_subsystem("comp_wing_sweep_100_inner", ComputeWingSweep100Inner(), promotes=["*"])
        self.add_subsystem("comp_wing_sweep_100_outer", ComputeWingSweep100Outer(), promotes=["*"])


class ComputeWingSweep0(ExplicitComponent):
    """Estimation of wing sweep at l/c=0%"""

    def setup(self):

        self.add_input("data:geometry:wing:tip:leading_edge:x:local", val=np.nan, units="m")
        self.add_input("data:geometry:wing:root:y", val=np.nan, units="m")
        self.add_input("data:geometry:wing:tip:y", val=np.nan, units="m")

        self.add_output("data:geometry:wing:sweep_0", units="rad")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        x4_wing = inputs["data:geometry:wing:tip:leading_edge:x:local"]
        y2_wing = inputs["data:geometry:wing:root:y"]
        y4_wing = inputs["data:geometry:wing:tip:y"]

        sweep_0 = np.arctan2(x4_wing, (y4_wing - y2_wing))

        outputs["data:geometry:wing:sweep_0"] = sweep_0


class ComputeWingSweep50(ExplicitComponent):
    """Estimation of wing sweep at l/c=50%"""

    def setup(self):

        self.add_input("data:geometry:wing:aspect_ratio", val=np.nan)
        self.add_input("data:geometry:wing:taper_ratio", val=np.nan)
        self.add_input("data:geometry:wing:sweep_0", units="rad")

        self.add_output("data:geometry:wing:sweep_50", units="rad")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        wing_ar = inputs["data:geometry:wing:aspect_ratio"]
        taper_ratio_wing = inputs["data:geometry:wing:taper_ratio"]
        sweep_0 = inputs["data:geometry:wing:sweep_0"]

        outputs["data:geometry:wing:sweep_50"] = np.arctan(
            np.tan(sweep_0) - 2 / wing_ar * ((1 - taper_ratio_wing) / (1 + taper_ratio_wing))
        )


class ComputeWingSweep100Inner(ExplicitComponent):
    """Estimation of inner wing sweep at l/c=100%"""

    def setup(self):

        self.add_input("data:geometry:wing:tip:leading_edge:x:local", val=np.nan, units="m")
        self.add_input("data:geometry:wing:root:y", val=np.nan, units="m")
        self.add_input("data:geometry:wing:tip:y", val=np.nan, units="m")
        self.add_input("data:geometry:wing:root:chord", val=np.nan, units="m")
        self.add_input("data:geometry:wing:tip:chord", val=np.nan, units="m")

        self.add_output("data:geometry:wing:sweep_100_inner", units="rad")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        x4_wing = inputs["data:geometry:wing:tip:leading_edge:x:local"]
        y2_wing = inputs["data:geometry:wing:root:y"]
        y4_wing = inputs["data:geometry:wing:tip:y"]
        l2_wing = inputs["data:geometry:wing:root:chord"]
        l4_wing = inputs["data:geometry:wing:tip:chord"]

        outputs["data:geometry:wing:sweep_100_inner"] = np.arctan2(
            (x4_wing + l4_wing - l2_wing), (y4_wing - y2_wing)
        )


class ComputeWingSweep100Outer(ExplicitComponent):
    """Estimation of outer wing sweep at l/c=100%"""

    def setup(self):

        self.add_input("data:geometry:wing:tip:leading_edge:x:local", val=np.nan, units="m")
        self.add_input("data:geometry:wing:root:y", val=np.nan, units="m")
        self.add_input("data:geometry:wing:tip:y", val=np.nan, units="m")
        self.add_input("data:geometry:wing:root:chord", val=np.nan, units="m")
        self.add_input("data:geometry:wing:tip:chord", val=np.nan, units="m")

        self.add_output("data:geometry:wing:sweep_100_outer", units="rad")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        x4_wing = inputs["data:geometry:wing:tip:leading_edge:x:local"]
        y2_wing = inputs["data:geometry:wing:root:y"]
        y4_wing = inputs["data:geometry:wing:tip:y"]
        l2_wing = inputs["data:geometry:wing:root:chord"]
        l4_wing = inputs["data:geometry:wing:tip:chord"]

        outputs["data:geometry:wing:sweep_100_outer"] = np.arctan2(
            (x4_wing + l4_wing - l2_wing), (y4_wing - y2_wing)
        )
