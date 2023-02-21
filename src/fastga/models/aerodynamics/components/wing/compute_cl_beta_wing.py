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
from ...constants import SUBMODEL_CL_BETA_WING


@oad.RegisterSubmodel(
    SUBMODEL_CL_BETA_WING, "fastga.submodel.aerodynamics.wing.roll_moment_beta.legacy"
)
class ComputeClBetaWing(FigureDigitization):
    """
    Class to compute the contribution of the wing to the roll moment coefficient due to sideslip.
    Depends on the lift coefficient of the wing, hence on the reference angle of attack,
    so the same remark as in ..compute_cy_yaw_rate.py holds. The convention from
    :cite:`roskampart6:1985` are used, meaning that for lateral derivative, the reference length
    is the wing span.

    Based on :cite:`roskampart6:1985` section 10.2.4.1.
    """

    def initialize(self):

        self.options.declare("low_speed_aero", default=False, types=bool)

    def setup(self):

        self.add_input("data:geometry:wing:aspect_ratio", val=np.nan)
        self.add_input("data:geometry:wing:taper_ratio", val=np.nan)
        self.add_input("data:geometry:wing:sweep_50", val=np.nan, units="rad")
        self.add_input("data:geometry:wing:sweep_25", val=np.nan, units="rad")
        self.add_input("data:geometry:wing:dihedral", val=np.nan, units="deg")
        self.add_input(
            "data:geometry:wing:twist",
            val=0.0,
            units="deg",
            desc="Negative twist means tip AOA is smaller than root",
        )
        self.add_input("data:geometry:wing:span", val=np.nan, units="m")
        self.add_input("data:geometry:wing:tip:leading_edge:x:absolute", val=np.nan, units="m")
        self.add_input("data:geometry:wing:tip:chord", val=np.nan, units="m")
        self.add_input("data:geometry:wing:root:z", val=np.nan, units="m")

        self.add_input("data:geometry:fuselage:master_cross_section", val=np.nan, units="m**2")

        if self.options["low_speed_aero"]:
            self.add_input(
                "settings:aerodynamics:reference_flight_conditions:low_speed:AOA",
                units="rad",
                val=5.0 * np.pi / 180.0,
            )
            self.add_input("data:aerodynamics:low_speed:mach", val=np.nan)
            self.add_input("data:aerodynamics:wing:low_speed:CL0_clean", val=np.nan)
            self.add_input("data:aerodynamics:wing:low_speed:CL_alpha", val=np.nan, units="rad**-1")

            self.add_output("data:aerodynamics:wing:low_speed:Cl_beta", units="rad**-1")

        else:
            self.add_input(
                "settings:aerodynamics:reference_flight_conditions:cruise:AOA",
                units="rad",
                val=1.0 * np.pi / 180.0,
            )
            self.add_input("data:aerodynamics:cruise:mach", val=np.nan)
            self.add_input("data:aerodynamics:wing:cruise:CL0_clean", val=np.nan)
            self.add_input("data:aerodynamics:wing:cruise:CL_alpha", val=np.nan, units="rad**-1")

            self.add_output("data:aerodynamics:wing:cruise:Cl_beta", units="rad**-1")

        self.declare_partials(of="*", wrt="*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        wing_ar = inputs["data:geometry:wing:aspect_ratio"]
        wing_taper_ratio = inputs["data:geometry:wing:taper_ratio"]
        wing_sweep_50 = inputs["data:geometry:wing:sweep_50"]  # In rad !!!
        wing_sweep_25 = inputs["data:geometry:wing:sweep_25"]  # In rad !!!
        wing_dihedral = inputs["data:geometry:wing:dihedral"]  # In deg, not specified in the
        # formula
        wing_twist = inputs["data:geometry:wing:twist"]  # In deg, not specified in the
        # formula
        wing_span = inputs["data:geometry:wing:span"]
        x4_wing_abs = inputs["data:geometry:wing:tip:leading_edge:x:absolute"]
        l4_wing = inputs["data:geometry:wing:tip:chord"]
        z2_wing = inputs["data:geometry:wing:root:z"]

        master_cross_section = inputs["data:geometry:fuselage:master_cross_section"]
        avg_fus_depth = np.sqrt(master_cross_section / 0.7854)

        if self.options["low_speed_aero"]:
            aoa_ref = inputs["settings:aerodynamics:reference_flight_conditions:low_speed:AOA"]
            mach = inputs["data:aerodynamics:low_speed:mach"]
            cl_0_wing = inputs["data:aerodynamics:wing:low_speed:CL0_clean"]
            cl_alpha_wing = inputs["data:aerodynamics:wing:low_speed:CL_alpha"]
        else:
            aoa_ref = inputs["settings:aerodynamics:reference_flight_conditions:cruise:AOA"]
            mach = inputs["data:aerodynamics:cruise:mach"]
            cl_0_wing = inputs["data:aerodynamics:wing:cruise:CL0_clean"]
            cl_alpha_wing = inputs["data:aerodynamics:wing:cruise:CL_alpha"]

        # Fuselage contribution neglected for now
        cl_wf = cl_0_wing + cl_alpha_wing * aoa_ref

        swept_wing_ar = wing_ar / np.cos(wing_sweep_50)
        swept_mach = mach * np.cos(wing_sweep_50)
        l_f = x4_wing_abs + 0.5 * l4_wing

        cl_beta_wf_sweep = self.cl_beta_sweep_contribution(
            wing_taper_ratio, wing_ar, wing_sweep_50 * 180.0 / np.pi
        )
        k_m_lambda = self.cl_beta_sweep_compressibility_correction(swept_wing_ar, swept_mach)
        k_f = self.cl_beta_fuselage_correction(swept_wing_ar, l_f / wing_span)

        cl_beta_wf_ar = self.cl_beta_ar_contribution(wing_taper_ratio, wing_ar)

        cl_beta_wf_dihedral = self.cl_beta_dihedral_contribution(
            wing_taper_ratio, wing_ar, wing_sweep_50 * 180.0 / np.pi
        )
        k_m_gamma = self.cl_beta_dihedral_compressibility_correction(swept_wing_ar, swept_mach)
        delta_cl_beta_wf_dihedral = 0.0005 * wing_ar * (avg_fus_depth / wing_span) ** 2.0

        delta_cl_beta_wf_zw = (
            0.042 * np.sqrt(wing_ar) * z2_wing / wing_span * avg_fus_depth / wing_span
        )
        k_epsilon = self.cl_beta_twist_correction(wing_taper_ratio, wing_ar)

        cl_beta_wf = 57.3 * (
            cl_wf * (cl_beta_wf_sweep * k_m_lambda * k_f + cl_beta_wf_ar)
            + wing_dihedral * (cl_beta_wf_dihedral * k_m_gamma + delta_cl_beta_wf_dihedral)
            + delta_cl_beta_wf_zw
            + wing_twist * np.tan(wing_sweep_25) * k_epsilon
        )

        if self.options["low_speed_aero"]:
            outputs["data:aerodynamics:wing:low_speed:Cl_beta"] = cl_beta_wf
        else:
            outputs["data:aerodynamics:wing:cruise:Cl_beta"] = cl_beta_wf
