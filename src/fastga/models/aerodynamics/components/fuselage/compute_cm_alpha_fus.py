"""Estimation of fuselage pitching moment."""
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

from fastga.models.aerodynamics.constants import SUBMODEL_CM_ALPHA_FUSELAGE
from fastga.models.aerodynamics.components.digitization.compute_k_fuselage import (
    ComputeFuselagePitchMomentFactor,
)


@oad.RegisterSubmodel(
    SUBMODEL_CM_ALPHA_FUSELAGE, "fastga.submodel.aerodynamics.fuselage.pitching_moment_alpha.legacy"
)
class ComputeCmAlphaFuselage(om.Group):  # pylint: disable=too-few-public-methods
    """
    Estimation of the fuselage pitching moment using the methodology described in section 16.3.8
    of Raymer

    Based on : Raymer, Daniel P. "Aircraft design: a conceptual approach (AIAA Education
    Series)." Reston, Virginia (2012).
    """

    # pylint: disable=missing-function-docstring
    # Overriding OpenMDAO setup
    def setup(self):
        self.add_subsystem(
            name="quarter_root_chord_position",
            subsys=_ComputeQuarterRootChordPositionRatio(),
            promotes=["*"],
        )
        self.add_subsystem(
            name="k_fuselage", subsys=ComputeFuselagePitchMomentFactor(), promotes=["*"]
        )
        self.add_subsystem(
            name="cm_alpha_fuselage", subsys=_ComputeCmAlphaFuselageNacelle(), promotes=["*"]
        )


class _ComputeQuarterRootChordPositionRatio(om.ExplicitComponent):
    """
    Computation of the position of the quarter wing root chord from the nose divided by the total
    fuselage length.
    """

    # pylint: disable=missing-function-docstring
    # Overriding OpenMDAO setup
    def setup(self):
        self.add_input("data:geometry:wing:MAC:leading_edge:x:local", val=np.nan, units="m")
        self.add_input("data:geometry:wing:MAC:length", val=np.nan, units="m")
        self.add_input("data:geometry:wing:MAC:at25percent:x", val=np.nan, units="m")
        self.add_input("data:geometry:wing:root:virtual_chord", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:length", val=np.nan, units="m")

        self.add_output("x0_ratio", val=0.2)

        self.declare_partials("x0_ratio", "*", method="exact")

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute, not all arguments are used
    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        x_wing = inputs["data:geometry:wing:MAC:at25percent:x"]
        x0_wing = inputs["data:geometry:wing:MAC:leading_edge:x:local"]
        l0_wing = inputs["data:geometry:wing:MAC:length"]
        l1_wing = inputs["data:geometry:wing:root:virtual_chord"]
        fus_length = inputs["data:geometry:fuselage:length"]

        outputs["x0_ratio"] = (x_wing - 0.25 * l0_wing - x0_wing + 0.25 * l1_wing) / fus_length

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute_partials, not all arguments are used
    def compute_partials(self, inputs, partials, discrete_inputs=None):
        x_wing = inputs["data:geometry:wing:MAC:at25percent:x"]
        x0_wing = inputs["data:geometry:wing:MAC:leading_edge:x:local"]
        l0_wing = inputs["data:geometry:wing:MAC:length"]
        l1_wing = inputs["data:geometry:wing:root:virtual_chord"]
        fus_length = inputs["data:geometry:fuselage:length"]

        partials["x0_ratio", "data:geometry:fuselage:length"] = -(
            (x_wing - 0.25 * l0_wing - x0_wing + 0.25 * l1_wing) / fus_length**2.0
        )
        partials["x0_ratio", "data:geometry:wing:MAC:at25percent:x"] = 1.0 / fus_length
        partials["x0_ratio", "data:geometry:wing:MAC:length"] = -0.25 / fus_length
        partials["x0_ratio", "data:geometry:wing:MAC:leading_edge:x:local"] = -1.0 / fus_length
        partials["x0_ratio", "data:geometry:wing:root:virtual_chord"] = 0.25 / fus_length


class _ComputeCmAlphaFuselageNacelle(om.ExplicitComponent):
    """
    Estimation of the fuselage pitching moment using the methodology described in section 16.3.8
    of Raymer :cite:`raymer:2012`.
    """

    # pylint: disable=missing-function-docstring
    # Overriding OpenMDAO setup
    def setup(self):
        self.add_input("data:geometry:wing:MAC:length", val=np.nan, units="m")
        self.add_input("data:geometry:wing:area", val=np.nan, units="m**2")
        self.add_input("data:geometry:fuselage:length", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:maximum_width", val=np.nan, units="m")
        self.add_input("fuselage_pitch_moment_factor", val=np.nan, units="rad**-1")

        self.add_output("data:aerodynamics:fuselage:cm_alpha", units="rad**-1")

        self.declare_partials("data:aerodynamics:fuselage:cm_alpha", "*", method="exact")

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute, not all arguments are used
    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        wing_area = inputs["data:geometry:wing:area"]
        l0_wing = inputs["data:geometry:wing:MAC:length"]
        fus_length = inputs["data:geometry:fuselage:length"]
        width_max = inputs["data:geometry:fuselage:maximum_width"]
        k_fus = inputs["fuselage_pitch_moment_factor"]

        outputs["data:aerodynamics:fuselage:cm_alpha"] = (
            -k_fus * width_max**2.0 * fus_length / (l0_wing * wing_area)
        )

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute_partials, not all arguments are used
    def compute_partials(self, inputs, partials, discrete_inputs=None):
        wing_area = inputs["data:geometry:wing:area"]
        l0_wing = inputs["data:geometry:wing:MAC:length"]
        fus_length = inputs["data:geometry:fuselage:length"]
        width_max = inputs["data:geometry:fuselage:maximum_width"]
        k_fus = inputs["fuselage_pitch_moment_factor"]

        partials["data:aerodynamics:fuselage:cm_alpha", "data:geometry:wing:area"] = (
            k_fus * width_max**2.0 * fus_length / (l0_wing * wing_area**2.0)
        )
        partials["data:aerodynamics:fuselage:cm_alpha", "data:geometry:wing:MAC:length"] = (
            k_fus * width_max**2.0 * fus_length / (l0_wing**2.0 * wing_area)
        )
        partials["data:aerodynamics:fuselage:cm_alpha", "data:geometry:fuselage:length"] = (
            -k_fus * width_max**2.0 / (l0_wing * wing_area)
        )
        partials["data:aerodynamics:fuselage:cm_alpha", "data:geometry:fuselage:maximum_width"] = (
            -2.0 * k_fus * width_max * fus_length / (l0_wing * wing_area)
        )
        partials["data:aerodynamics:fuselage:cm_alpha", "fuselage_pitch_moment_factor"] = (
            -(width_max**2.0) * fus_length / (l0_wing * wing_area)
        )
