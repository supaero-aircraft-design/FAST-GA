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

from ..figure_digitization import FigureDigitization
from ...constants import SUBMODEL_CM_Q_WING


@oad.RegisterSubmodel(
    SUBMODEL_CM_Q_WING, "fastga.submodel.aerodynamics.wing.cm_pitch_velocity.legacy"
)
class ComputeCMPitchVelocityWing(FigureDigitization):
    """
    Class to compute the contribution of the wing to the pitch moment coefficient due to pitch
    rate. The vertical distance between the cg and the aerodynamic center of teh plane is taken
    equal to the vertical distance between the root chord and the fuselage centerline. The
    convention from :cite:`roskampart6:1985` are used, meaning that, for the derivative with
    respect to a pitch rate, this rate is made dimensionless by multiplying it by the MAC and
    dividing it by 2 times the airspeed.

    Based on :cite:`gudmundsson:2013`.
    """

    def initialize(self):

        self.options.declare("low_speed_aero", default=False, types=bool)

    def setup(self):

        self.add_input("data:geometry:wing:aspect_ratio", val=np.nan)
        self.add_input("data:geometry:wing:sweep_25", val=np.nan, units="rad")
        self.add_input("data:geometry:wing:root:z", units="m", val=np.nan)

        if self.options["low_speed_aero"]:
            self.add_input("data:aerodynamics:low_speed:mach", val=np.nan)
            self.add_input("data:aerodynamics:wing:low_speed:CL_alpha", val=np.nan, units="rad**-1")

            self.add_output("data:aerodynamics:wing:low_speed:Cm_q", units="rad**-1")

        else:
            self.add_input("data:aerodynamics:cruise:mach", val=np.nan)
            self.add_input("data:aerodynamics:wing:cruise:CL_alpha", val=np.nan, units="rad**-1")

            self.add_output("data:aerodynamics:wing:cruise:Cm_q", units="rad**-1")

        self.declare_partials(of="*", wrt="*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        wing_ar = inputs["data:geometry:wing:aspect_ratio"]
        wing_sweep_25 = inputs["data:geometry:wing:sweep_25"]  # In rad !!!
        # The convention used for the computation of z2_wing was positive when cg is above the
        # wing, here it correspond to the opposite, hence the minus sign
        h_ac_to_cg = -inputs["data:geometry:wing:root:z"]

        if self.options["low_speed_aero"]:
            mach = inputs["data:aerodynamics:low_speed:mach"]
            cl_alpha_wing = inputs["data:aerodynamics:wing:low_speed:CL_alpha"]
        else:
            mach = inputs["data:aerodynamics:cruise:mach"]
            cl_alpha_wing = inputs["data:aerodynamics:wing:cruise:CL_alpha"]

        cm_q_wing = (
            -0.7
            * cl_alpha_wing
            * np.cos(wing_sweep_25)
            * (
                (wing_ar * (0.5 * h_ac_to_cg + 2.0 * h_ac_to_cg ** 2.0))
                / (wing_ar + 2.0 * np.cos(wing_sweep_25))
                + 1.0
                / 24.0
                * (wing_ar ** 3.0 * np.tan(wing_sweep_25) ** 2.0)
                / (wing_ar + 60 * np.cos(wing_sweep_25))
                + 1.0 / 8.0
            )
        )

        if mach > 0.2:
            a1_coeff = (wing_ar ** 3.0 * np.tan(wing_sweep_25) ** 2.0) / (
                wing_ar * np.sqrt(1.0 - (mach * np.cos(wing_sweep_25)) ** 2.0)
                + 6.0 * np.cos(wing_sweep_25)
            ) + 3.0 / (np.sqrt(1.0 - (mach * np.cos(wing_sweep_25)) ** 2.0))
            a2_coeff = (wing_ar ** 3.0 * np.tan(wing_sweep_25) ** 2.0) / (
                wing_ar + 6.0 * np.cos(wing_sweep_25)
            ) + 3
            cm_q_wing *= a1_coeff / a2_coeff

        if self.options["low_speed_aero"]:
            outputs["data:aerodynamics:wing:low_speed:Cm_q"] = cm_q_wing
        else:
            outputs["data:aerodynamics:wing:cruise:Cm_q"] = cm_q_wing
