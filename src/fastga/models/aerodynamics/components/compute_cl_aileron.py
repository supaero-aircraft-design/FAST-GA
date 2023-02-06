"""Estimation of rolling moment du to the ailerons."""
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
import fastoad.api as oad

from .figure_digitization import FigureDigitization
from ..constants import SUBMODEL_CL_AILERON


@oad.RegisterSubmodel(
    SUBMODEL_CL_AILERON, "fastga.submodel.aerodynamics.aileron.roll_moment.legacy"
)
class ComputeClDeltaAileron(FigureDigitization):
    """
    Roll moment due to aileron deflection estimated based on the methodology presented in
    Gudmundsson. This methodology is known to overestimate the coefficient so a correction factor
    will be added. This coefficient assumes a symmetrical deflection. The convention from
    :cite:`roskampart6:1985` are used, meaning that for lateral derivative, the reference length
    is the wing span.

    Based on the methodology presented in :cite:`gudmundsson:2013`.
    """

    def initialize(self):

        self.options.declare("low_speed_aero", default=False, types=bool)

    def setup(self):

        self.add_input("data:geometry:wing:aileron:chord_ratio", val=np.nan)
        self.add_input("data:geometry:wing:aileron:span_ratio", val=np.nan)

        self.add_input("data:geometry:wing:span", val=np.nan, units="m")
        self.add_input("data:geometry:wing:area", val=np.nan, units="m**2")
        self.add_input("data:geometry:wing:root:chord", val=np.nan, units="m")
        self.add_input("data:geometry:wing:taper_ratio", val=np.nan)

        self.add_input(
            "settings:aerodynamics:aileron:tip_effect:k_factor",
            val=0.9,
            desc="Correction coefficient to take into account tip effect when "
            "computing the roll authority of the ailerons",
        )

        if self.options["low_speed_aero"]:
            self.add_input("data:aerodynamics:low_speed:mach", val=np.nan)

            self.add_output("data:aerodynamics:aileron:low_speed:Cl_delta_a", units="rad**-1")
        else:
            self.add_input("data:aerodynamics:cruise:mach", val=np.nan)

            self.add_output("data:aerodynamics:aileron:cruise:Cl_delta_a", units="rad**-1")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        wing_span = inputs["data:geometry:wing:span"]
        wing_area = inputs["data:geometry:wing:area"]
        l2_wing = inputs["data:geometry:wing:root:chord"]
        wing_taper_ratio = inputs["data:geometry:wing:taper_ratio"]
        aileron_span_ratio = inputs["data:geometry:wing:aileron:span_ratio"]

        tip_effect_factor = inputs["settings:aerodynamics:aileron:tip_effect:k_factor"]

        if self.options["low_speed_aero"]:
            mach = inputs["data:aerodynamics:low_speed:mach"]
        else:
            mach = inputs["data:aerodynamics:cruise:mach"]

        aileron_chord_ratio = inputs["data:geometry:wing:aileron:chord_ratio"]

        aileron_outer_span = wing_span / 2.0
        aileron_inner_span = wing_span * (1.0 - aileron_span_ratio) / 2.0

        # Aileron are most mostly going to be used around delta_a = 0 degree, which is the reason
        # why the effectiveness is going to be computed around this deflection
        alpha_aileron = self.k_prime_single_slotted(0.0, float(aileron_chord_ratio))
        lift_increase_aileron = 2 * np.pi / np.sqrt(1 - mach ** 2) * alpha_aileron

        cl_delta_a = (
            lift_increase_aileron
            * l2_wing
            / (wing_area * wing_span)
            * (
                (aileron_outer_span ** 2.0 - aileron_inner_span ** 2.0)
                + (4.0 * (wing_taper_ratio - 1.0))
                / (3.0 * wing_span)
                * (aileron_outer_span ** 3.0 - aileron_inner_span ** 3.0)
            )
        ) * tip_effect_factor

        if self.options["low_speed_aero"]:
            outputs["data:aerodynamics:aileron:low_speed:Cl_delta_a"] = cl_delta_a
        else:
            outputs["data:aerodynamics:aileron:cruise:Cl_delta_a"] = cl_delta_a
