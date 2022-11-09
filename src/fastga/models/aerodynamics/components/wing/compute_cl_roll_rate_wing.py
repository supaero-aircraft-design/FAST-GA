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
from ...constants import SUBMODEL_CL_P_WING


@oad.RegisterSubmodel(
    SUBMODEL_CL_P_WING, "fastga.submodel.aerodynamics.wing.roll_moment_roll_rate.legacy"
)
class ComputeClRollRateWing(FigureDigitization):
    """
    Class to compute the contribution of the wing to the roll moment coefficient due to roll
    rate. (roll damping). Depends on the lift coefficient of the wing, hence on the reference
    angle of attack, so the same remark as in ..compute_cy_yaw_rate.py holds. The convention from
    :cite:`roskampart6:1985` are used, meaning that for lateral derivative, the reference length
    is the wing span. Another important point is that, for the derivative with respect to yaw and
    roll, the rotation speed are made dimensionless by multiplying them by the wing span and
    dividing them by 2 times the airspeed.

    Based on :cite:`roskampart6:1985` section 10.2.6
    """

    def initialize(self):

        self.options.declare("low_speed_aero", default=False, types=bool)

    def setup(self):

        self.add_input("data:geometry:wing:aspect_ratio", val=np.nan)
        self.add_input("data:geometry:wing:taper_ratio", val=np.nan)
        self.add_input("data:geometry:wing:sweep_25", val=np.nan, units="rad")
        self.add_input("data:geometry:wing:dihedral", val=np.nan, units="rad")
        self.add_input("data:geometry:wing:root:z", val=np.nan, units="m")
        self.add_input("data:geometry:wing:span", val=np.nan, units="m")

        self.add_input("data:aerodynamics:wing:airfoil:CL_alpha", val=np.nan, units="rad**-1")

        if self.options["low_speed_aero"]:
            self.add_input(
                "settings:aerodynamics:reference_flight_conditions:low_speed:AOA",
                units="rad",
                val=5.0 * np.pi / 180.0,
            )
            self.add_input("data:aerodynamics:low_speed:mach", val=np.nan)
            self.add_input("data:aerodynamics:wing:low_speed:CL0_clean", val=np.nan)
            self.add_input("data:aerodynamics:wing:low_speed:CL_alpha", val=np.nan, units="rad**-1")
            self.add_input("data:aerodynamics:wing:low_speed:CD0", val=np.nan)

            self.add_output("data:aerodynamics:wing:low_speed:Cl_p", units="rad**-1")

        else:
            self.add_input(
                "settings:aerodynamics:reference_flight_conditions:cruise:AOA",
                units="rad",
                val=1.0 * np.pi / 180.0,
            )
            self.add_input("data:aerodynamics:cruise:mach", val=np.nan)
            self.add_input("data:aerodynamics:wing:cruise:CL0_clean", val=np.nan)
            self.add_input("data:aerodynamics:wing:cruise:CL_alpha", val=np.nan, units="rad**-1")
            self.add_input("data:aerodynamics:wing:cruise:CD0", val=np.nan)

            self.add_output("data:aerodynamics:wing:cruise:Cl_p", units="rad**-1")

        self.declare_partials(of="*", wrt="*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        wing_ar = inputs["data:geometry:wing:aspect_ratio"]
        wing_span = inputs["data:geometry:wing:span"]
        wing_taper_ratio = inputs["data:geometry:wing:taper_ratio"]
        wing_sweep_25 = inputs["data:geometry:wing:sweep_25"]  # In rad !!!
        wing_dihedral = inputs["data:geometry:wing:dihedral"]  # In rad
        z2_wing = inputs["data:geometry:wing:root:z"]

        cl_alpha_airfoil = inputs["data:aerodynamics:wing:airfoil:CL_alpha"]

        if self.options["low_speed_aero"]:
            aoa_ref = inputs["settings:aerodynamics:reference_flight_conditions:low_speed:AOA"]
            mach = inputs["data:aerodynamics:low_speed:mach"]
            cl_0_wing = inputs["data:aerodynamics:wing:low_speed:CL0_clean"]
            cl_alpha_wing = inputs["data:aerodynamics:wing:low_speed:CL_alpha"]
            cd0_wing = inputs["data:aerodynamics:wing:low_speed:CD0"]
        else:
            aoa_ref = inputs["settings:aerodynamics:reference_flight_conditions:cruise:AOA"]
            mach = inputs["data:aerodynamics:cruise:mach"]
            cl_0_wing = inputs["data:aerodynamics:wing:cruise:CL0_clean"]
            cl_alpha_wing = inputs["data:aerodynamics:wing:cruise:CL_alpha"]
            cd0_wing = inputs["data:aerodynamics:wing:cruise:CD0"]

        k = cl_alpha_airfoil / (2.0 * np.pi)
        beta = np.sqrt(1.0 - mach ** 2.0)

        roll_damping_parameter = self.cl_p_roll_damping_parameter(
            wing_taper_ratio, wing_ar, mach, wing_sweep_25, k
        )

        cl_wf = cl_0_wing + cl_alpha_wing * aoa_ref

        dihedral_effect = (
            1.0
            - 4.0 * z2_wing / wing_span * np.sin(wing_dihedral)
            + 12.0 * (z2_wing / wing_span) ** 2.0 * np.sin(wing_dihedral) ** 2.0
        )

        roll_damping_due_to_induced_drag_parameter = self.cl_p_cdi_roll_damping(
            wing_sweep_25, wing_ar
        )
        roll_damping_due_to_induced_drag = (
            roll_damping_due_to_induced_drag_parameter * cl_wf ** 2.0 - 0.125 * cd0_wing
        )

        # The assumption we make on the lift curve makes the second term equal to 1.0
        cl_p_w = (
            roll_damping_parameter * k / beta * 1.0 * dihedral_effect
            + roll_damping_due_to_induced_drag
        )

        if self.options["low_speed_aero"]:
            outputs["data:aerodynamics:wing:low_speed:Cl_p"] = cl_p_w
        else:
            outputs["data:aerodynamics:wing:cruise:Cl_p"] = cl_p_w
