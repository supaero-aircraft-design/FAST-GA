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
from ..constants import SUBMODEL_CN_AILERON


@oad.RegisterSubmodel(SUBMODEL_CN_AILERON, "fastga.submodel.aerodynamics.aileron.yaw_moment.legacy")
class ComputeCnDeltaAileron(FigureDigitization):
    """
    Yaw moment due to aileron deflection (also called adverse aileron yaw). Depends on the wing
    lift, hence on the angle of attack, so the same remark as in ..compute_cy_yaw_rate.py holds.
    The convention from :cite:`roskampart6:1985` are used, meaning that for lateral derivative,
    the reference length is the wing span.

    Based on :cite:`roskampart6:1985` section 10.3.8.
    """

    def initialize(self):

        self.options.declare("low_speed_aero", default=False, types=bool)

    def setup(self):

        self.add_input("data:geometry:wing:aileron:span_ratio", val=np.nan)
        self.add_input("data:geometry:wing:taper_ratio", val=np.nan)
        self.add_input("data:geometry:wing:aspect_ratio", val=np.nan)

        if self.options["low_speed_aero"]:
            self.add_input(
                "settings:aerodynamics:reference_flight_conditions:low_speed:AOA",
                units="rad",
                val=5.0 * np.pi / 180.0,
            )
            self.add_input(
                "data:aerodynamics:aileron:low_speed:Cl_delta_a", val=np.nan, units="rad**-1"
            )
            self.add_input("data:aerodynamics:wing:low_speed:CL0_clean", val=np.nan)
            self.add_input("data:aerodynamics:wing:low_speed:CL_alpha", val=np.nan, units="rad**-1")

            self.add_output("data:aerodynamics:aileron:low_speed:Cn_delta_a", units="rad**-1")
        else:
            self.add_input(
                "settings:aerodynamics:reference_flight_conditions:cruise:AOA",
                units="rad",
                val=1.0 * np.pi / 180.0,
            )
            self.add_input(
                "data:aerodynamics:aileron:cruise:Cl_delta_a", val=np.nan, units="rad**-1"
            )
            self.add_input("data:aerodynamics:wing:cruise:CL0_clean", val=np.nan)
            self.add_input("data:aerodynamics:wing:cruise:CL_alpha", val=np.nan, units="rad**-1")

            self.add_output("data:aerodynamics:aileron:cruise:Cn_delta_a", units="rad**-1")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        wing_taper_ratio = inputs["data:geometry:wing:taper_ratio"]
        wing_aspect_ratio = inputs["data:geometry:wing:aspect_ratio"]
        aileron_span_ratio = inputs["data:geometry:wing:aileron:span_ratio"]

        if self.options["low_speed_aero"]:
            aoa_ref = inputs["settings:aerodynamics:reference_flight_conditions:low_speed:AOA"]
            cl_delta_a = inputs["data:aerodynamics:aileron:low_speed:Cl_delta_a"]
            cl_0_wing = inputs["data:aerodynamics:wing:low_speed:CL0_clean"]
            cl_alpha_wing = inputs["data:aerodynamics:wing:low_speed:CL_alpha"]
        else:
            aoa_ref = inputs["settings:aerodynamics:reference_flight_conditions:cruise:AOA"]
            cl_delta_a = inputs["data:aerodynamics:aileron:cruise:Cl_delta_a"]
            cl_0_wing = inputs["data:aerodynamics:wing:cruise:CL0_clean"]
            cl_alpha_wing = inputs["data:aerodynamics:wing:cruise:CL_alpha"]

        aileron_inner_span_ratio = 1.0 - aileron_span_ratio

        cl_w = cl_0_wing + cl_alpha_wing * aoa_ref

        correlation_constant = self.cn_delta_a_correlation_constant(
            wing_taper_ratio, wing_aspect_ratio, aileron_inner_span_ratio
        )

        cn_delta_a = correlation_constant * cl_w * cl_delta_a

        if self.options["low_speed_aero"]:
            outputs["data:aerodynamics:aileron:low_speed:Cn_delta_a"] = cn_delta_a
        else:
            outputs["data:aerodynamics:aileron:cruise:Cn_delta_a"] = cn_delta_a
