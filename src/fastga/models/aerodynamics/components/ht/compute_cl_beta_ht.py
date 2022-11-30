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
from ...constants import SUBMODEL_CL_BETA_HT


@oad.RegisterSubmodel(
    SUBMODEL_CL_BETA_HT, "fastga.submodel.aerodynamics.horizontal_tail.roll_moment_beta.legacy"
)
class ComputeClBetaHorizontalTail(FigureDigitization):
    """
    Class to compute the contribution of the horizontal tail to the roll moment coefficient due
    to sideslip. Depends on the lift coefficient, hence on the reference angle of attack,
    so the same remark as in ..compute_cy_yaw_rate.py holds. The convention from
    :cite:`roskampart6:1985` are used, meaning that for lateral derivative, the reference length
    is the wing span.

    Based on :cite:`roskampart6:1985` section 10.2.4.1.
    """

    def initialize(self):

        self.options.declare("low_speed_aero", default=False, types=bool)

    def setup(self):

        self.add_input("data:geometry:horizontal_tail:aspect_ratio", val=np.nan)
        self.add_input("data:geometry:horizontal_tail:taper_ratio", val=np.nan)
        self.add_input("data:geometry:horizontal_tail:sweep_50", val=np.nan, units="rad")
        self.add_input("data:geometry:horizontal_tail:sweep_25", val=np.nan, units="rad")
        self.add_input("data:geometry:horizontal_tail:dihedral", val=0.0, units="deg")
        self.add_input("data:geometry:horizontal_tail:twist", val=0.0, units="deg")
        self.add_input("data:geometry:horizontal_tail:span", val=np.nan, units="m")
        self.add_input("data:geometry:horizontal_tail:area", val=np.nan, units="m**2")
        self.add_input("data:geometry:wing:area", val=np.nan, units="m**2")
        self.add_input("data:geometry:wing:span", val=np.nan, units="m")

        self.add_input("data:geometry:wing:MAC:at25percent:x", val=np.nan, units="m")
        self.add_input(
            "data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25", val=np.nan, units="m"
        )
        self.add_input("data:geometry:horizontal_tail:tip:chord", val=np.nan, units="m")

        self.add_input("data:geometry:horizontal_tail:z:from_wingMAC25", val=np.nan, units="m")
        self.add_input("data:geometry:wing:root:z", val=np.nan, units="m")

        self.add_input("data:geometry:fuselage:average_depth", val=np.nan, units="m")

        if self.options["low_speed_aero"]:
            self.add_input(
                "settings:aerodynamics:reference_flight_conditions:low_speed:AOA",
                units="rad",
                val=5.0 * np.pi / 180.0,
            )
            self.add_input("data:aerodynamics:low_speed:mach", val=np.nan)
            self.add_input("data:aerodynamics:horizontal_tail:low_speed:CL0", val=np.nan)
            self.add_input(
                "data:aerodynamics:horizontal_tail:low_speed:CL_alpha", val=np.nan, units="rad**-1"
            )

            self.add_output("data:aerodynamics:horizontal_tail:low_speed:Cl_beta", units="rad**-1")

        else:
            self.add_input(
                "settings:aerodynamics:reference_flight_conditions:cruise:AOA",
                units="rad",
                val=1.0 * np.pi / 180.0,
            )
            self.add_input("data:aerodynamics:cruise:mach", val=np.nan)
            self.add_input("data:aerodynamics:horizontal_tail:cruise:CL0", val=np.nan)
            self.add_input(
                "data:aerodynamics:horizontal_tail:cruise:CL_alpha", val=np.nan, units="rad**-1"
            )

            self.add_output("data:aerodynamics:horizontal_tail:cruise:Cl_beta", units="rad**-1")

        self.declare_partials(of="*", wrt="*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        ht_area = inputs["data:geometry:horizontal_tail:area"]
        wing_area = inputs["data:geometry:wing:area"]
        wing_span = inputs["data:geometry:wing:span"]
        ht_ar = inputs["data:geometry:horizontal_tail:aspect_ratio"]
        ht_taper_ratio = inputs["data:geometry:horizontal_tail:taper_ratio"]
        ht_sweep_50 = inputs["data:geometry:horizontal_tail:sweep_50"]  # In rad !!!
        ht_sweep_25 = inputs["data:geometry:horizontal_tail:sweep_25"]  # In rad !!!
        ht_dihedral = inputs["data:geometry:horizontal_tail:dihedral"]  # In deg, not specified
        # in the  formula
        ht_twist = inputs["data:geometry:horizontal_tail:twist"]  # In deg, not specified in the
        # formula
        ht_span = inputs["data:geometry:horizontal_tail:span"]

        fa_length = inputs["data:geometry:wing:MAC:at25percent:x"]
        lp_ht = inputs["data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25"]
        x4_ht = inputs["data:geometry:horizontal_tail:tip:chord"]

        if float(inputs["data:geometry:horizontal_tail:z:from_wingMAC25"]) == 0.0:
            z2_ht = 0.0  # Aligned with the fuselage centerline
        else:
            z2_ht = (
                inputs["data:geometry:wing:root:z"]
                - inputs["data:geometry:horizontal_tail:z:from_wingMAC25"]
            )

        # Represents the average depth at the VT location, used as an approximate
        avg_fus_depth = inputs["data:geometry:fuselage:average_depth"]

        if self.options["low_speed_aero"]:
            aoa_ref = inputs["settings:aerodynamics:reference_flight_conditions:low_speed:AOA"]
            mach = inputs["data:aerodynamics:low_speed:mach"]
            cl_0_ht = inputs["data:aerodynamics:horizontal_tail:low_speed:CL0"]
            cl_alpha_ht = inputs["data:aerodynamics:horizontal_tail:low_speed:CL_alpha"]
        else:
            aoa_ref = inputs["settings:aerodynamics:reference_flight_conditions:cruise:AOA"]
            mach = inputs["data:aerodynamics:cruise:mach"]
            cl_0_ht = inputs["data:aerodynamics:horizontal_tail:cruise:CL0"]
            cl_alpha_ht = inputs["data:aerodynamics:horizontal_tail:cruise:CL_alpha"]

        # Fuselage contribution neglected for now
        cl_hf = (cl_0_ht + cl_alpha_ht * aoa_ref) * wing_area / ht_area

        swept_ht_ar = ht_ar / np.cos(ht_sweep_50)
        swept_mach = mach * np.cos(ht_sweep_50)
        l_f = fa_length + lp_ht + 0.25 * x4_ht  # Neglects the effects of ht sweep

        cl_beta_hf_sweep = self.cl_beta_sweep_contribution(
            ht_taper_ratio, ht_ar, ht_sweep_50 * 180.0 / np.pi
        )
        k_m_lambda = self.cl_beta_sweep_compressibility_correction(swept_ht_ar, swept_mach)
        k_f = self.cl_beta_fuselage_correction(swept_ht_ar, l_f / ht_span)

        cl_beta_hf_ar = self.cl_beta_ar_contribution(ht_taper_ratio, ht_ar)

        cl_beta_hf_dihedral = self.cl_beta_dihedral_contribution(
            ht_taper_ratio, ht_ar, ht_sweep_50 * 180.0 / np.pi
        )
        k_m_gamma = self.cl_beta_dihedral_compressibility_correction(swept_ht_ar, swept_mach)
        delta_cl_beta_hf_dihedral = 0.0005 * ht_ar * (avg_fus_depth / ht_span) ** 2.0

        delta_cl_beta_hf_zh = 0.042 * np.sqrt(ht_ar) * z2_ht / ht_span * avg_fus_depth / ht_span
        k_epsilon = self.cl_beta_twist_correction(ht_taper_ratio, ht_ar)

        cl_beta_hf = (
            57.3
            * (
                cl_hf * (cl_beta_hf_sweep * k_m_lambda * k_f + cl_beta_hf_ar)
                + ht_dihedral * (cl_beta_hf_dihedral * k_m_gamma + delta_cl_beta_hf_dihedral)
                + delta_cl_beta_hf_zh
                + ht_twist * np.tan(ht_sweep_25) * k_epsilon
            )
            * (ht_area * ht_span)
            / (wing_area * wing_span)
        )

        if self.options["low_speed_aero"]:
            outputs["data:aerodynamics:horizontal_tail:low_speed:Cl_beta"] = cl_beta_hf
        else:
            outputs["data:aerodynamics:horizontal_tail:cruise:Cl_beta"] = cl_beta_hf
