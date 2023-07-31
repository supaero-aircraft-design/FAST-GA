"""
Estimation of tail weight.
"""
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

from stdatm import Atmosphere
import fastoad.api as oad

from .constants import SUBMODEL_VERTICAL_TAIL_MASS


@oad.RegisterSubmodel(
    SUBMODEL_VERTICAL_TAIL_MASS, "fastga.submodel.weight.mass.airframe.vertical_tail.torenbeek_gd"
)
class ComputeVerticalTailWeightTorenbeekGD(om.ExplicitComponent):
    """
    Weight estimation for tail weight

    Based on a statistical analysis. See :cite:`roskampart5:1985` traditionally used on
    commercial aircraft but found to work fairly well on high performance GA aircraft. Should
    only be used with aircraft having a diving speed above 250 KEAS.
    """

    def setup(self):

        self.add_input("data:TLAR:v_max_sl", val=np.nan, units="m/s")
        self.add_input("data:mission:sizing:cs23:sizing_factor:ultimate_aircraft", val=np.nan)
        self.add_input("data:weight:aircraft:MTOW", val=np.nan, units="lb")
        self.add_input("data:weight:airframe:vertical_tail:k_factor", val=1.0)

        self.add_input("data:geometry:has_T_tail", val=np.nan)
        self.add_input("data:geometry:vertical_tail:area", val=np.nan, units="ft**2")
        self.add_input(
            "data:geometry:vertical_tail:MAC:at25percent:x:from_wingMAC25", val=np.nan, units="ft"
        )
        # Rudder is assumed to take full span so rudder ratio is equal to chord ratio
        self.add_input("data:geometry:vertical_tail:rudder:chord_ratio", val=np.nan)
        self.add_input("data:geometry:vertical_tail:sweep_25", val=np.nan, units="rad")
        self.add_input("data:geometry:vertical_tail:aspect_ratio", val=np.nan)
        self.add_input("data:geometry:vertical_tail:taper_ratio", val=np.nan)

        self.add_output("data:weight:airframe:vertical_tail:mass", units="lb")

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        sizing_factor_ultimate = inputs["data:mission:sizing:cs23:sizing_factor:ultimate_aircraft"]
        mtow = inputs["data:weight:aircraft:MTOW"]

        has_t_tail = inputs["data:geometry:has_T_tail"]
        area_vt = inputs["data:geometry:vertical_tail:area"]
        v_h = inputs["data:TLAR:v_max_sl"]
        lp_vt = inputs["data:geometry:vertical_tail:MAC:at25percent:x:from_wingMAC25"]
        rudder_chord_ratio = inputs["data:geometry:vertical_tail:rudder:chord_ratio"]
        sweep_25_vt = inputs["data:geometry:vertical_tail:sweep_25"]
        ar_vt = inputs["data:geometry:vertical_tail:aspect_ratio"]
        taper_vt = inputs["data:geometry:vertical_tail:taper_ratio"]

        atm0 = Atmosphere(0)
        atm0.true_airspeed = v_h
        mach_h = atm0.mach

        a32 = (
            0.19
            * (
                (1 + has_t_tail) ** 0.5
                * (sizing_factor_ultimate * mtow) ** 0.363
                * area_vt ** 1.089
                * mach_h ** 0.601
                * lp_vt ** -0.726
                * (1 + rudder_chord_ratio) ** 0.217
                * ar_vt ** 0.337
                * (1 + taper_vt) ** 0.363
                * np.cos(sweep_25_vt) ** -0.484
            )
            ** 1.014
        )
        # Mass formula in lb

        outputs["data:weight:airframe:vertical_tail:mass"] = (
            a32 * inputs["data:weight:airframe:vertical_tail:k_factor"]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        sizing_factor_ultimate = inputs["data:mission:sizing:cs23:sizing_factor:ultimate_aircraft"]
        mtow = inputs["data:weight:aircraft:MTOW"]

        has_t_tail = inputs["data:geometry:has_T_tail"]
        area_vt = inputs["data:geometry:vertical_tail:area"]
        v_h = inputs["data:TLAR:v_max_sl"]
        lp_vt = inputs["data:geometry:vertical_tail:MAC:at25percent:x:from_wingMAC25"]
        rudder_chord_ratio = inputs["data:geometry:vertical_tail:rudder:chord_ratio"]
        sweep_25_vt = inputs["data:geometry:vertical_tail:sweep_25"]
        ar_vt = inputs["data:geometry:vertical_tail:aspect_ratio"]
        taper_vt = inputs["data:geometry:vertical_tail:taper_ratio"]

        k_factor = inputs["data:weight:airframe:vertical_tail:k_factor"]

        atm0 = Atmosphere(0)
        atm0.true_airspeed = v_h
        mach_h = atm0.mach

        a32 = (
            0.19
            * (
                (1 + has_t_tail) ** 0.5
                * (sizing_factor_ultimate * mtow) ** 0.363
                * area_vt ** 1.089
                * mach_h ** 0.601
                * lp_vt ** -0.726
                * (1 + rudder_chord_ratio) ** 0.217
                * ar_vt ** 0.337
                * (1 + taper_vt) ** 0.363
                * np.cos(sweep_25_vt) ** -0.484
            )
            ** 1.014
        )
        # Mass formula in lb

        partials[
            "data:weight:airframe:vertical_tail:mass",
            "data:mission:sizing:cs23:sizing_factor:ultimate_aircraft",
        ] = k_factor * (
            (
                0.06994
                * ar_vt ** 0.337
                * area_vt ** 1.089
                * mach_h ** 0.601
                * mtow
                * (has_t_tail + 1.0) ** 0.5
                * (rudder_chord_ratio + 1.0) ** 0.217
                * (taper_vt + 1.0) ** 0.363
                * (
                    (
                        ar_vt ** 0.337
                        * area_vt ** 1.089
                        * mach_h ** 0.601
                        * (has_t_tail + 1.0) ** 0.5
                        * (rudder_chord_ratio + 1.0) ** 0.217
                        * (taper_vt + 1.0) ** 0.363
                        * (mtow * sizing_factor_ultimate) ** 0.363
                    )
                    / (lp_vt ** 0.726 * np.cos(sweep_25_vt) ** 0.484)
                )
                ** 0.014
            )
            / (
                lp_vt ** 0.726
                * np.cos(sweep_25_vt) ** 0.484
                * (mtow * sizing_factor_ultimate) ** 0.637
            )
        )
        partials[
            "data:weight:airframe:vertical_tail:mass", "data:weight:aircraft:MTOW"
        ] = k_factor * (
            (
                0.06994
                * ar_vt ** 0.337
                * area_vt ** 1.089
                * mach_h ** 0.601
                * sizing_factor_ultimate
                * (has_t_tail + 1.0) ** 0.5
                * (rudder_chord_ratio + 1.0) ** 0.217
                * (taper_vt + 1.0) ** 0.363
                * (
                    (
                        ar_vt ** 0.337
                        * area_vt ** 1.089
                        * mach_h ** 0.601
                        * (has_t_tail + 1.0) ** 0.5
                        * (rudder_chord_ratio + 1.0) ** 0.217
                        * (taper_vt + 1.0) ** 0.363
                        * (mtow * sizing_factor_ultimate) ** 0.363
                    )
                    / (lp_vt ** 0.726 * np.cos(sweep_25_vt) ** 0.484)
                )
                ** 0.014
            )
            / (
                lp_vt ** 0.726
                * np.cos(sweep_25_vt) ** 0.484
                * (mtow * sizing_factor_ultimate) ** 0.637
            )
        )
        partials[
            "data:weight:airframe:vertical_tail:mass", "data:geometry:has_T_tail"
        ] = k_factor * (
            (
                0.09633
                * ar_vt ** 0.337
                * area_vt ** 1.089
                * mach_h ** 0.601
                * (rudder_chord_ratio + 1.0) ** 0.217
                * (taper_vt + 1.0) ** 0.363
                * (mtow * sizing_factor_ultimate) ** 0.363
                * (
                    (
                        ar_vt ** 0.337
                        * area_vt ** 1.089
                        * mach_h ** 0.601
                        * (has_t_tail + 1.0) ** 0.5
                        * (rudder_chord_ratio + 1.0) ** 0.217
                        * (taper_vt + 1.0) ** 0.363
                        * (mtow * sizing_factor_ultimate) ** 0.363
                    )
                    / (lp_vt ** 0.726 * np.cos(sweep_25_vt) ** 0.484)
                )
                ** 0.014
            )
            / (lp_vt ** 0.726 * np.cos(sweep_25_vt) ** 0.484 * (has_t_tail + 1.0) ** 0.5)
        )
        partials[
            "data:weight:airframe:vertical_tail:mass", "data:geometry:vertical_tail:area"
        ] = k_factor * (
            (
                0.2098
                * ar_vt ** 0.337
                * area_vt ** 0.089
                * mach_h ** 0.601
                * (has_t_tail + 1.0) ** 0.5
                * (rudder_chord_ratio + 1.0) ** 0.217
                * (taper_vt + 1.0) ** 0.363
                * (mtow * sizing_factor_ultimate) ** 0.363
                * (
                    (
                        ar_vt ** 0.337
                        * area_vt ** 1.089
                        * mach_h ** 0.601
                        * (has_t_tail + 1.0) ** 0.5
                        * (rudder_chord_ratio + 1.0) ** 0.217
                        * (taper_vt + 1.0) ** 0.363
                        * (mtow * sizing_factor_ultimate) ** 0.363
                    )
                    / (lp_vt ** 0.726 * np.cos(sweep_25_vt) ** 0.484)
                )
                ** 0.014
            )
            / (lp_vt ** 0.726 * np.cos(sweep_25_vt) ** 0.484)
        )
        partials[
            "data:weight:airframe:vertical_tail:mass",
            "data:geometry:vertical_tail:MAC:at25percent:x:from_wingMAC25",
        ] = k_factor * (
            -(
                0.1399
                * ar_vt ** 0.337
                * area_vt ** 1.089
                * mach_h ** 0.601
                * (has_t_tail + 1.0) ** 0.5
                * (rudder_chord_ratio + 1.0) ** 0.217
                * (taper_vt + 1.0) ** 0.363
                * (mtow * sizing_factor_ultimate) ** 0.363
                * (
                    (
                        ar_vt ** 0.337
                        * area_vt ** 1.089
                        * mach_h ** 0.601
                        * (has_t_tail + 1.0) ** 0.5
                        * (rudder_chord_ratio + 1.0) ** 0.217
                        * (taper_vt + 1.0) ** 0.363
                        * (mtow * sizing_factor_ultimate) ** 0.363
                    )
                    / (lp_vt ** 0.726 * np.cos(sweep_25_vt) ** 0.484)
                )
                ** 0.014
            )
            / (lp_vt ** 1.7260 * np.cos(sweep_25_vt) ** 0.484)
        )
        partials[
            "data:weight:airframe:vertical_tail:mass",
            "data:geometry:vertical_tail:rudder:chord_ratio",
        ] = k_factor * (
            (
                0.04181
                * ar_vt ** 0.337
                * area_vt ** 1.089
                * mach_h ** 0.601
                * (has_t_tail + 1.0) ** 0.5
                * (taper_vt + 1.0) ** 0.363
                * (mtow * sizing_factor_ultimate) ** 0.363
                * (
                    (
                        ar_vt ** 0.337
                        * area_vt ** 1.089
                        * mach_h ** 0.601
                        * (has_t_tail + 1.0) ** 0.5
                        * (rudder_chord_ratio + 1.0) ** 0.217
                        * (taper_vt + 1.0) ** 0.363
                        * (mtow * sizing_factor_ultimate) ** 0.363
                    )
                    / (lp_vt ** 0.726 * np.cos(sweep_25_vt) ** 0.484)
                )
                ** 0.014
            )
            / (lp_vt ** 0.726 * np.cos(sweep_25_vt) ** 0.484 * (rudder_chord_ratio + 1.0) ** 0.7830)
        )
        partials[
            "data:weight:airframe:vertical_tail:mass", "data:geometry:vertical_tail:sweep_25"
        ] = k_factor * (
            (
                0.09325
                * ar_vt ** 0.337
                * area_vt ** 1.089
                * mach_h ** 0.601
                * np.sin(sweep_25_vt)
                * (has_t_tail + 1.0) ** 0.5
                * (rudder_chord_ratio + 1.0) ** 0.217
                * (taper_vt + 1.0) ** 0.363
                * (mtow * sizing_factor_ultimate) ** 0.363
                * (
                    (
                        ar_vt ** 0.337
                        * area_vt ** 1.089
                        * mach_h ** 0.601
                        * (has_t_tail + 1.0) ** 0.5
                        * (rudder_chord_ratio + 1.0) ** 0.217
                        * (taper_vt + 1.0) ** 0.363
                        * (mtow * sizing_factor_ultimate) ** 0.363
                    )
                    / (lp_vt ** 0.726 * np.cos(sweep_25_vt) ** 0.484)
                )
                ** 0.014
            )
            / (lp_vt ** 0.726 * np.cos(sweep_25_vt) ** 1.4840)
        )
        partials[
            "data:weight:airframe:vertical_tail:mass", "data:geometry:vertical_tail:aspect_ratio"
        ] = k_factor * (
            (
                0.06493
                * area_vt ** 1.089
                * mach_h ** 0.601
                * (has_t_tail + 1.0) ** 0.5
                * (rudder_chord_ratio + 1.0) ** 0.217
                * (taper_vt + 1.0) ** 0.363
                * (mtow * sizing_factor_ultimate) ** 0.363
                * (
                    (
                        ar_vt ** 0.337
                        * area_vt ** 1.089
                        * mach_h ** 0.601
                        * (has_t_tail + 1.0) ** 0.5
                        * (rudder_chord_ratio + 1.0) ** 0.217
                        * (taper_vt + 1.0) ** 0.363
                        * (mtow * sizing_factor_ultimate) ** 0.363
                    )
                    / (lp_vt ** 0.726 * np.cos(sweep_25_vt) ** 0.484)
                )
                ** 0.014
            )
            / (ar_vt ** 0.6630 * lp_vt ** 0.726 * np.cos(sweep_25_vt) ** 0.484)
        )
        partials[
            "data:weight:airframe:vertical_tail:mass", "data:geometry:vertical_tail:taper_ratio"
        ] = k_factor * (
            (
                0.06994
                * ar_vt ** 0.337
                * area_vt ** 1.089
                * mach_h ** 0.601
                * (has_t_tail + 1.0) ** 0.5
                * (rudder_chord_ratio + 1.0) ** 0.217
                * (mtow * sizing_factor_ultimate) ** 0.363
                * (
                    (
                        ar_vt ** 0.337
                        * area_vt ** 1.089
                        * mach_h ** 0.601
                        * (has_t_tail + 1.0) ** 0.5
                        * (rudder_chord_ratio + 1.0) ** 0.217
                        * (taper_vt + 1.0) ** 0.363
                        * (mtow * sizing_factor_ultimate) ** 0.363
                    )
                    / (lp_vt ** 0.726 * np.cos(sweep_25_vt) ** 0.484)
                )
                ** 0.014
            )
            / (lp_vt ** 0.726 * np.cos(sweep_25_vt) ** 0.484 * (taper_vt + 1.0) ** 0.637)
        )
        d_a32_d_mach = (
            0.1158
            * ar_vt ** 0.337
            * area_vt ** 1.089
            * (has_t_tail + 1.0) ** 0.5
            * (rudder_chord_ratio + 1.0) ** 0.217
            * (taper_vt + 1.0) ** 0.363
            * (mtow * sizing_factor_ultimate) ** 0.363
            * (
                (
                    ar_vt ** 0.337
                    * area_vt ** 1.089
                    * mach_h ** 0.601
                    * (has_t_tail + 1.0) ** 0.5
                    * (rudder_chord_ratio + 1.0) ** 0.217
                    * (taper_vt + 1.0) ** 0.363
                    * (mtow * sizing_factor_ultimate) ** 0.363
                )
                / (lp_vt ** 0.726 * np.cos(sweep_25_vt) ** 0.484)
            )
            ** 0.014
        ) / (lp_vt ** 0.726 * mach_h ** (399 / 1000) * np.cos(sweep_25_vt) ** 0.484)
        d_mach_d_vh = 1 / atm0.speed_of_sound
        partials["data:weight:airframe:vertical_tail:mass", "data:TLAR:v_max_sl"] = k_factor * (
            d_a32_d_mach * d_mach_d_vh
        )
        partials[
            "data:weight:airframe:vertical_tail:mass", "data:weight:airframe:vertical_tail:k_factor"
        ] = a32
