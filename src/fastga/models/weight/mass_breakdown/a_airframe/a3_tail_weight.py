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

import fastoad.api as oad
import numpy as np
import openmdao.api as om
from stdatm import Atmosphere, AtmosphereWithPartials

from .constants import SUBMODEL_TAIL_MASS

oad.RegisterSubmodel.active_models[SUBMODEL_TAIL_MASS] = (
    "fastga.submodel.weight.mass.airframe.tail.legacy"
)


@oad.RegisterSubmodel(SUBMODEL_TAIL_MASS, "fastga.submodel.weight.mass.airframe.tail.legacy")
class ComputeTailWeight(om.ExplicitComponent):
    """
    Weight estimation for tail weight

    Based on a statistical analysis. See :cite:`raymer:2012` but can also be found in
    :cite:`gudmundsson:2013`
    """

    def setup(self):
        self.add_input("data:mission:sizing:cs23:sizing_factor:ultimate_aircraft", val=np.nan)
        self.add_input("data:weight:aircraft:MTOW", val=np.nan, units="lb")
        self.add_input("data:weight:airframe:horizontal_tail:k_factor", val=1.0)
        self.add_input("data:weight:airframe:vertical_tail:k_factor", val=1.0)
        self.add_input("data:TLAR:v_cruise", val=np.nan, units="m/s")
        self.add_input("data:mission:sizing:main_route:cruise:altitude", val=np.nan, units="ft")

        self.add_input("data:geometry:horizontal_tail:area", val=np.nan, units="ft**2")
        self.add_input("data:geometry:horizontal_tail:thickness_ratio", val=np.nan)
        self.add_input("data:geometry:horizontal_tail:sweep_25", val=np.nan, units="rad")
        self.add_input("data:geometry:horizontal_tail:aspect_ratio", val=np.nan)
        self.add_input("data:geometry:horizontal_tail:taper_ratio", val=np.nan)

        self.add_input("data:geometry:has_T_tail", val=np.nan)
        self.add_input("data:geometry:vertical_tail:area", val=np.nan, units="ft**2")
        self.add_input("data:geometry:vertical_tail:thickness_ratio", val=np.nan)
        self.add_input("data:geometry:vertical_tail:sweep_25", val=np.nan, units="rad")
        self.add_input("data:geometry:vertical_tail:aspect_ratio", val=np.nan)
        self.add_input("data:geometry:vertical_tail:taper_ratio", val=np.nan)

        self.add_output("data:weight:airframe:horizontal_tail:mass", units="lb")
        self.add_output("data:weight:airframe:vertical_tail:mass", units="lb")

        self.declare_partials(
            of="data:weight:airframe:horizontal_tail:mass",
            wrt=[
                "data:mission:sizing:cs23:sizing_factor:ultimate_aircraft",
                "data:weight:aircraft:MTOW",
                "data:weight:airframe:horizontal_tail:k_factor",
                "data:TLAR:v_cruise",
                "data:geometry:horizontal_tail:area",
                "data:geometry:horizontal_tail:thickness_ratio",
                "data:geometry:horizontal_tail:sweep_25",
                "data:geometry:horizontal_tail:aspect_ratio",
                "data:geometry:horizontal_tail:taper_ratio",
                "data:mission:sizing:main_route:cruise:altitude",
            ],
            method="exact",
        )

        self.declare_partials(
            of="data:weight:airframe:vertical_tail:mass",
            wrt=[
                "data:mission:sizing:cs23:sizing_factor:ultimate_aircraft",
                "data:weight:aircraft:MTOW",
                "data:weight:airframe:vertical_tail:k_factor",
                "data:TLAR:v_cruise",
                "data:geometry:has_T_tail",
                "data:geometry:vertical_tail:area",
                "data:geometry:vertical_tail:thickness_ratio",
                "data:geometry:vertical_tail:sweep_25",
                "data:geometry:vertical_tail:aspect_ratio",
                "data:geometry:vertical_tail:taper_ratio",
                "data:mission:sizing:main_route:cruise:altitude",
            ],
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        sizing_factor_ultimate = inputs["data:mission:sizing:cs23:sizing_factor:ultimate_aircraft"]
        mtow = inputs["data:weight:aircraft:MTOW"]
        v_cruise_ktas = inputs["data:TLAR:v_cruise"]
        cruise_alt = inputs["data:mission:sizing:main_route:cruise:altitude"]

        area_ht = inputs["data:geometry:horizontal_tail:area"]
        t_c_ht = inputs["data:geometry:horizontal_tail:thickness_ratio"]
        sweep_25_ht = inputs["data:geometry:horizontal_tail:sweep_25"]
        ar_ht = inputs["data:geometry:horizontal_tail:aspect_ratio"]
        taper_ht = inputs["data:geometry:horizontal_tail:taper_ratio"]

        rho_cruise = Atmosphere(cruise_alt, altitude_in_feet=True).density
        dynamic_pressure = 1.0 / 2.0 * rho_cruise * v_cruise_ktas**2.0 * 0.0208854
        # In lb/ft2

        a31 = 0.016 * (
            (sizing_factor_ultimate * mtow) ** 0.414
            * dynamic_pressure**0.168
            * area_ht**0.896
            * (100.0 * t_c_ht / np.cos(sweep_25_ht)) ** -0.12
            * (ar_ht / np.cos(sweep_25_ht) ** 2.0) ** 0.043
            * taper_ht**-0.02
        )
        # Mass formula in lb

        outputs["data:weight:airframe:horizontal_tail:mass"] = (
            a31 * inputs["data:weight:airframe:horizontal_tail:k_factor"]
        )

        has_t_tail = inputs["data:geometry:has_T_tail"]
        area_vt = inputs["data:geometry:vertical_tail:area"]
        t_c_vt = inputs["data:geometry:vertical_tail:thickness_ratio"]
        sweep_25_vt = inputs["data:geometry:vertical_tail:sweep_25"]
        ar_vt = inputs["data:geometry:vertical_tail:aspect_ratio"]
        taper_vt = inputs["data:geometry:vertical_tail:taper_ratio"]

        a32 = (
            0.073
            * (1.0 + 0.2 * has_t_tail)
            * (
                (sizing_factor_ultimate * mtow) ** 0.376
                * dynamic_pressure**0.122
                * area_vt**0.873
                * (100.0 * t_c_vt / np.cos(sweep_25_vt)) ** -0.49
                * (ar_vt / np.cos(sweep_25_vt) ** 2.0) ** 0.357
                * taper_vt**0.039
            )
        )
        # Mass formula in lb

        outputs["data:weight:airframe:vertical_tail:mass"] = (
            a32 * inputs["data:weight:airframe:vertical_tail:k_factor"]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        sizing_factor_ultimate = inputs["data:mission:sizing:cs23:sizing_factor:ultimate_aircraft"]
        mtow = inputs["data:weight:aircraft:MTOW"]
        v_cruise_ktas = inputs["data:TLAR:v_cruise"]
        cruise_alt = inputs["data:mission:sizing:main_route:cruise:altitude"]
        atm_cruise = AtmosphereWithPartials(cruise_alt)

        area_ht = inputs["data:geometry:horizontal_tail:area"]
        t_c_ht = inputs["data:geometry:horizontal_tail:thickness_ratio"]
        sweep_25_ht = inputs["data:geometry:horizontal_tail:sweep_25"]
        ar_ht = inputs["data:geometry:horizontal_tail:aspect_ratio"]
        taper_ht = inputs["data:geometry:horizontal_tail:taper_ratio"]

        k_factor_ht = inputs["data:weight:airframe:horizontal_tail:k_factor"]
        k_factor_vt = inputs["data:weight:airframe:vertical_tail:k_factor"]

        has_t_tail = inputs["data:geometry:has_T_tail"]
        area_vt = inputs["data:geometry:vertical_tail:area"]
        t_c_vt = inputs["data:geometry:vertical_tail:thickness_ratio"]
        sweep_25_vt = inputs["data:geometry:vertical_tail:sweep_25"]
        ar_vt = inputs["data:geometry:vertical_tail:aspect_ratio"]
        taper_vt = inputs["data:geometry:vertical_tail:taper_ratio"]

        rho_cruise = atm_cruise.density
        dynamic_pressure = 1.0 / 2.0 * rho_cruise * v_cruise_ktas**2.0 * 0.0208854

        partials[
            "data:weight:airframe:horizontal_tail:mass",
            "data:mission:sizing:cs23:sizing_factor:ultimate_aircraft",
        ] = k_factor_ht * (
            0.016
            * (
                0.414
                * (sizing_factor_ultimate * mtow) ** -0.586
                * mtow
                * dynamic_pressure**0.168
                * area_ht**0.896
                * (100.0 * t_c_ht / np.cos(sweep_25_ht)) ** -0.12
                * (ar_ht / np.cos(sweep_25_ht) ** 2.0) ** 0.043
                * taper_ht ** -0.02
            )
        )
        partials["data:weight:airframe:horizontal_tail:mass", "data:weight:aircraft:MTOW"] = (
            k_factor_ht
            * (
                0.016
                * (
                    0.414
                    * (sizing_factor_ultimate * mtow) ** -0.586
                    * sizing_factor_ultimate
                    * dynamic_pressure**0.168
                    * area_ht**0.896
                    * (100.0 * t_c_ht / np.cos(sweep_25_ht)) ** -0.12
                    * (ar_ht / np.cos(sweep_25_ht) ** 2.0) ** 0.043
                    * taper_ht ** -0.02
                )
            )
        )
        partials["data:weight:airframe:horizontal_tail:mass", "data:TLAR:v_cruise"] = (
            k_factor_ht
            * 0.016
            * (
                (sizing_factor_ultimate * mtow) ** 0.414
                * 2
                * 0.168
                * (0.5 * rho_cruise * 0.0208854) ** 0.168
                * v_cruise_ktas ** (-0.664)
                * area_ht**0.896
                * (100.0 * t_c_ht / np.cos(sweep_25_ht)) ** -0.12
                * (ar_ht / np.cos(sweep_25_ht) ** 2.0) ** 0.043
                * taper_ht ** -0.02
            )
        )
        partials[
            "data:weight:airframe:horizontal_tail:mass", "data:geometry:horizontal_tail:area"
        ] = k_factor_ht * (
            0.016
            * (
                (sizing_factor_ultimate * mtow) ** 0.414
                * dynamic_pressure**0.168
                * 0.896
                * area_ht ** (-0.104)
                * (100.0 * t_c_ht / np.cos(sweep_25_ht)) ** -0.12
                * (ar_ht / np.cos(sweep_25_ht) ** 2.0) ** 0.043
                * taper_ht ** -0.02
            )
        )
        partials[
            "data:weight:airframe:horizontal_tail:mass",
            "data:geometry:horizontal_tail:thickness_ratio",
        ] = k_factor_ht * (
            0.016
            * (
                (sizing_factor_ultimate * mtow) ** 0.414
                * dynamic_pressure**0.168
                * area_ht**0.896
                * (100.0 / np.cos(sweep_25_ht)) ** -0.12
                * (-0.12 * t_c_ht**-1.12)
                * (ar_ht / np.cos(sweep_25_ht) ** 2.0) ** 0.043
                * taper_ht ** -0.02
            )
        )
        partials[
            "data:weight:airframe:horizontal_tail:mass", "data:geometry:horizontal_tail:sweep_25"
        ] = k_factor_ht * (
            0.016
            * (
                (sizing_factor_ultimate * mtow) ** 0.414
                * dynamic_pressure**0.168
                * area_ht**0.896
                * (100.0 * t_c_ht) ** -0.12
                * ar_ht**0.043
                * taper_ht ** -0.02
                * (
                    -0.12 * np.cos(sweep_25_ht) ** 0.034 * np.tan(sweep_25_ht)
                    + 0.086 * np.tan(sweep_25_ht) * np.cos(sweep_25_ht)
                )
            )
        )
        partials[
            "data:weight:airframe:horizontal_tail:mass",
            "data:geometry:horizontal_tail:aspect_ratio",
        ] = k_factor_ht * (
            0.016
            * (
                (sizing_factor_ultimate * mtow) ** 0.414
                * dynamic_pressure**0.168
                * area_ht**0.896
                * (100.0 * t_c_ht / np.cos(sweep_25_ht)) ** -0.12
                * (1 / np.cos(sweep_25_ht) ** 2.0) ** 0.043
                * 0.043
                * ar_ht ** (-0.957)
                * taper_ht ** -0.02
            )
        )
        partials[
            "data:weight:airframe:horizontal_tail:mass", "data:geometry:horizontal_tail:taper_ratio"
        ] = k_factor_ht * (
            -0.016
            * (
                (sizing_factor_ultimate * mtow) ** 0.414
                * dynamic_pressure**0.168
                * area_ht**0.896
                * (100.0 * t_c_ht / np.cos(sweep_25_ht)) ** -0.12
                * (ar_ht / np.cos(sweep_25_ht) ** 2.0) ** 0.043
                * 0.02
                * taper_ht ** -1.02
            )
        )
        partials[
            "data:weight:airframe:horizontal_tail:mass",
            "data:weight:airframe:horizontal_tail:k_factor",
        ] = 0.016 * (
            (sizing_factor_ultimate * mtow) ** 0.414
            * dynamic_pressure**0.168
            * area_ht**0.896
            * (100.0 * t_c_ht / np.cos(sweep_25_ht)) ** -0.12
            * (ar_ht / np.cos(sweep_25_ht) ** 2.0) ** 0.043
            * taper_ht ** -0.02
        )

        partials[
            "data:weight:airframe:horizontal_tail:mass",
            "data:mission:sizing:main_route:cruise:altitude",
        ] = k_factor_ht * (
            0.016
            * (
                (sizing_factor_ultimate * mtow) ** 0.414
                * (0.5 * v_cruise_ktas**2.0 * 0.0208854) ** 0.168
                * 0.168
                * rho_cruise ** (-0.832)
                * atm_cruise.partial_density_altitude
                * area_ht**0.896
                * (100.0 * t_c_ht / np.cos(sweep_25_ht)) ** -0.12
                * (ar_ht / np.cos(sweep_25_ht) ** 2.0) ** 0.043
                * taper_ht ** -0.02
            )
        )

        partials[
            "data:weight:airframe:vertical_tail:mass",
            "data:mission:sizing:cs23:sizing_factor:ultimate_aircraft",
        ] = k_factor_vt * (
            0.073
            * (1.0 + 0.2 * has_t_tail)
            * (
                0.376
                * (sizing_factor_ultimate * mtow) ** (-0.624)
                * mtow
                * dynamic_pressure**0.122
                * area_vt**0.873
                * (100.0 * t_c_vt / np.cos(sweep_25_vt)) ** -0.49
                * (ar_vt / np.cos(sweep_25_vt) ** 2.0) ** 0.357
                * taper_vt**0.039
            )
        )
        partials["data:weight:airframe:vertical_tail:mass", "data:weight:aircraft:MTOW"] = (
            k_factor_vt
            * (
                0.073
                * (1.0 + 0.2 * has_t_tail)
                * (
                    0.376
                    * (sizing_factor_ultimate * mtow) ** (-0.624)
                    * sizing_factor_ultimate
                    * dynamic_pressure**0.122
                    * area_vt**0.873
                    * (100.0 * t_c_vt / np.cos(sweep_25_vt)) ** -0.49
                    * (ar_vt / np.cos(sweep_25_vt) ** 2.0) ** 0.357
                    * taper_vt**0.039
                )
            )
        )
        partials["data:weight:airframe:vertical_tail:mass", "data:TLAR:v_cruise"] = k_factor_vt * (
            0.073
            * (1.0 + 0.2 * has_t_tail)
            * (
                (sizing_factor_ultimate * mtow) ** 0.376
                * 2
                * 0.122
                * (0.5 * rho_cruise * 0.0208854) ** 0.122
                * v_cruise_ktas ** (-0.756)
                * area_vt**0.873
                * (100.0 * t_c_vt / np.cos(sweep_25_vt)) ** -0.49
                * (ar_vt / np.cos(sweep_25_vt) ** 2.0) ** 0.357
                * taper_vt**0.039
            )
        )
        partials["data:weight:airframe:vertical_tail:mass", "data:geometry:has_T_tail"] = (
            k_factor_vt
            * (
                0.073
                * 0.2
                * (
                    (sizing_factor_ultimate * mtow) ** 0.376
                    * dynamic_pressure**0.122
                    * area_vt**0.873
                    * (100.0 * t_c_vt / np.cos(sweep_25_vt)) ** -0.49
                    * (ar_vt / np.cos(sweep_25_vt) ** 2.0) ** 0.357
                    * taper_vt**0.039
                )
            )
        )
        partials["data:weight:airframe:vertical_tail:mass", "data:geometry:vertical_tail:area"] = (
            k_factor_vt
            * (
                0.073
                * (1.0 + 0.2 * has_t_tail)
                * (
                    (sizing_factor_ultimate * mtow) ** 0.376
                    * dynamic_pressure**0.122
                    * 0.873
                    * area_vt ** (-0.127)
                    * (100.0 * t_c_vt / np.cos(sweep_25_vt)) ** -0.49
                    * (ar_vt / np.cos(sweep_25_vt) ** 2.0) ** 0.357
                    * taper_vt**0.039
                )
            )
        )
        partials[
            "data:weight:airframe:vertical_tail:mass", "data:geometry:vertical_tail:thickness_ratio"
        ] = k_factor_vt * (
            -0.073
            * (1.0 + 0.2 * has_t_tail)
            * (
                (sizing_factor_ultimate * mtow) ** 0.376
                * dynamic_pressure**0.122
                * area_vt**0.873
                * (100.0 / np.cos(sweep_25_vt)) ** -0.49
                * 0.49
                * t_c_vt ** (-1.49)
                * (ar_vt / np.cos(sweep_25_vt) ** 2.0) ** 0.357
                * taper_vt**0.039
            )
        )
        partials[
            "data:weight:airframe:vertical_tail:mass", "data:geometry:vertical_tail:sweep_25"
        ] = k_factor_vt * (
            0.073
            * (1.0 + 0.2 * has_t_tail)
            * (
                (sizing_factor_ultimate * mtow) ** 0.376
                * dynamic_pressure**0.122
                * area_vt**0.873
                * (100.0 * t_c_vt) ** -0.49
                * ar_vt**0.357
                * 0.224
                * np.tan(sweep_25_vt)
                * np.cos(sweep_25_vt) ** (-0.224)
                * taper_vt**0.039
            )
        )
        partials[
            "data:weight:airframe:vertical_tail:mass", "data:geometry:vertical_tail:aspect_ratio"
        ] = k_factor_vt * (
            0.073
            * (1.0 + 0.2 * has_t_tail)
            * (
                (sizing_factor_ultimate * mtow) ** 0.376
                * dynamic_pressure**0.122
                * area_vt**0.873
                * (100.0 * t_c_vt / np.cos(sweep_25_vt)) ** -0.49
                * (1 / np.cos(sweep_25_vt) ** 2.0) ** 0.357
                * 0.357
                * ar_vt ** (-0.643)
                * taper_vt**0.039
            )
        )
        partials[
            "data:weight:airframe:vertical_tail:mass", "data:geometry:vertical_tail:taper_ratio"
        ] = k_factor_vt * (
            0.073
            * (1.0 + 0.2 * has_t_tail)
            * (
                (sizing_factor_ultimate * mtow) ** 0.376
                * dynamic_pressure**0.122
                * area_vt**0.873
                * (100.0 * t_c_vt / np.cos(sweep_25_vt)) ** -0.49
                * (ar_vt / np.cos(sweep_25_vt) ** 2.0) ** 0.357
                * 0.039
                * taper_vt ** (-0.961)
            )
        )
        partials[
            "data:weight:airframe:vertical_tail:mass", "data:weight:airframe:vertical_tail:k_factor"
        ] = (
            0.073
            * (1.0 + 0.2 * has_t_tail)
            * (
                (sizing_factor_ultimate * mtow) ** 0.376
                * dynamic_pressure**0.122
                * area_vt**0.873
                * (100.0 * t_c_vt / np.cos(sweep_25_vt)) ** -0.49
                * (ar_vt / np.cos(sweep_25_vt) ** 2.0) ** 0.357
                * taper_vt**0.039
            )
        )

        partials[
            "data:weight:airframe:vertical_tail:mass",
            "data:mission:sizing:main_route:cruise:altitude",
        ] = k_factor_vt * (
            0.073
            * (1.0 + 0.2 * has_t_tail)
            * (
                (sizing_factor_ultimate * mtow) ** 0.376
                * (0.5 * v_cruise_ktas**2.0 * 0.0208854) ** 0.122
                * 0.122
                * rho_cruise ** (-0.878)
                * atm_cruise.partial_density_altitude
                * area_vt**0.873
                * (100.0 * t_c_vt / np.cos(sweep_25_vt)) ** -0.49
                * (ar_vt / np.cos(sweep_25_vt) ** 2.0) ** 0.357
                * taper_vt**0.039
            )
        )


@oad.RegisterSubmodel(SUBMODEL_TAIL_MASS, "fastga.submodel.weight.mass.airframe.tail.gd")
class ComputeTailWeightGD(om.ExplicitComponent):
    """
    Weight estimation for tail weight

    Based on a statistical analysis. See :cite:`roskampart5:1985` traditionally used on
    commercial aircraft but found to work fairly well on high performance GA aircraft.
    """

    def setup(self):
        self.add_input("data:TLAR:v_max_sl", val=np.nan, units="m/s")
        self.add_input("data:mission:sizing:cs23:sizing_factor:ultimate_aircraft", val=np.nan)
        self.add_input("data:weight:aircraft:MTOW", val=np.nan, units="lb")
        self.add_input("data:weight:airframe:horizontal_tail:k_factor", val=1.0)
        self.add_input("data:weight:airframe:vertical_tail:k_factor", val=1.0)

        self.add_input("data:geometry:horizontal_tail:area", val=np.nan, units="ft**2")
        self.add_input("data:geometry:horizontal_tail:thickness_ratio", val=np.nan)
        self.add_input("data:geometry:horizontal_tail:root:chord", val=np.nan, units="ft")
        self.add_input("data:geometry:horizontal_tail:MAC:length", val=np.nan, units="ft")
        self.add_input("data:geometry:horizontal_tail:span", val=np.nan, units="ft")
        self.add_input(
            "data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25", val=np.nan, units="ft"
        )

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

        self.add_output("data:weight:airframe:horizontal_tail:mass", units="lb")
        self.add_output("data:weight:airframe:vertical_tail:mass", units="lb")

        self.declare_partials(
            of="data:weight:airframe:horizontal_tail:mass",
            wrt=[
                "data:weight:aircraft:MTOW",
                "data:mission:sizing:cs23:sizing_factor:ultimate_aircraft",
                "data:geometry:horizontal_tail:area",
                "data:geometry:horizontal_tail:thickness_ratio",
                "data:geometry:horizontal_tail:root:chord",
                "data:geometry:horizontal_tail:span",
                "data:geometry:horizontal_tail:MAC:length",
                "data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25",
                "data:weight:airframe:horizontal_tail:k_factor",
            ],
            method="exact",
        )
        self.declare_partials(
            of="data:weight:airframe:vertical_tail:mass",
            wrt=[
                "data:geometry:has_T_tail",
                "data:weight:aircraft:MTOW",
                "data:mission:sizing:cs23:sizing_factor:ultimate_aircraft",
                "data:TLAR:v_max_sl",
                "data:geometry:vertical_tail:area",
                "data:geometry:vertical_tail:MAC:at25percent:x:from_wingMAC25",
                "data:geometry:vertical_tail:rudder:chord_ratio",
                "data:geometry:vertical_tail:aspect_ratio",
                "data:geometry:vertical_tail:taper_ratio",
                "data:geometry:vertical_tail:sweep_25",
                "data:weight:airframe:vertical_tail:k_factor",
            ],
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        sizing_factor_ultimate = inputs["data:mission:sizing:cs23:sizing_factor:ultimate_aircraft"]
        mtow = inputs["data:weight:aircraft:MTOW"]

        area_ht = inputs["data:geometry:horizontal_tail:area"]
        span_ht = inputs["data:geometry:horizontal_tail:span"]
        t_c_ht = inputs["data:geometry:horizontal_tail:thickness_ratio"]
        root_chord_ht = inputs["data:geometry:horizontal_tail:root:chord"]
        root_thickness = t_c_ht * root_chord_ht
        mac_ht = inputs["data:geometry:horizontal_tail:MAC:length"]
        lp_ht = inputs["data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25"]

        a31 = (
            0.0034
            * (
                (sizing_factor_ultimate * mtow) ** 0.813
                * area_ht**0.584
                * (span_ht / root_thickness) ** 0.033
                * (mac_ht / lp_ht) ** 0.28
            )
            ** 0.915
        )
        # Mass formula in lb

        outputs["data:weight:airframe:horizontal_tail:mass"] = (
            a31 * inputs["data:weight:airframe:horizontal_tail:k_factor"]
        )

        has_t_tail = inputs["data:geometry:has_T_tail"]
        area_vt = inputs["data:geometry:vertical_tail:area"]
        v_h = inputs["data:TLAR:v_max_sl"]
        atm0 = Atmosphere(0)
        atm0.true_airspeed = v_h
        mach_h = atm0.mach
        lp_vt = inputs["data:geometry:vertical_tail:MAC:at25percent:x:from_wingMAC25"]
        rudder_chord_ratio = inputs["data:geometry:vertical_tail:rudder:chord_ratio"]
        sweep_25_vt = inputs["data:geometry:vertical_tail:sweep_25"]
        ar_vt = inputs["data:geometry:vertical_tail:aspect_ratio"]
        taper_vt = inputs["data:geometry:vertical_tail:taper_ratio"]

        a32 = (
            0.19
            * (
                (1 + has_t_tail) ** 0.5
                * (sizing_factor_ultimate * mtow) ** 0.363
                * area_vt**1.089
                * mach_h**0.601
                * lp_vt**-0.726
                * (1 + rudder_chord_ratio) ** 0.217
                * ar_vt**0.337
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

        area_ht = inputs["data:geometry:horizontal_tail:area"]
        span_ht = inputs["data:geometry:horizontal_tail:span"]
        t_c_ht = inputs["data:geometry:horizontal_tail:thickness_ratio"]
        root_chord_ht = inputs["data:geometry:horizontal_tail:root:chord"]
        mac_ht = inputs["data:geometry:horizontal_tail:MAC:length"]
        lp_ht = inputs["data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25"]
        k_factor_ht = inputs["data:weight:airframe:horizontal_tail:k_factor"]

        root_thickness = t_c_ht * root_chord_ht

        has_t_tail = inputs["data:geometry:has_T_tail"]
        area_vt = inputs["data:geometry:vertical_tail:area"]
        lp_vt = inputs["data:geometry:vertical_tail:MAC:at25percent:x:from_wingMAC25"]
        rudder_chord_ratio = inputs["data:geometry:vertical_tail:rudder:chord_ratio"]
        sweep_25_vt = inputs["data:geometry:vertical_tail:sweep_25"]
        ar_vt = inputs["data:geometry:vertical_tail:aspect_ratio"]
        taper_vt = inputs["data:geometry:vertical_tail:taper_ratio"]
        v_h = inputs["data:TLAR:v_max_sl"]

        k_factor_vt = inputs["data:weight:airframe:vertical_tail:k_factor"]

        atm0 = Atmosphere(0)
        atm0.true_airspeed = v_h
        mach_h = atm0.mach

        partials[
            "data:weight:airframe:horizontal_tail:mass",
            "data:mission:sizing:cs23:sizing_factor:ultimate_aircraft",
        ] = k_factor_ht * (
            0.0034
            * 0.915
            * (area_ht**0.584 * (span_ht / root_thickness) ** 0.033 * (mac_ht / lp_ht) ** 0.28)
            ** 0.915
            * (sizing_factor_ultimate * mtow) ** (-0.069105)
            * 0.813
            * (sizing_factor_ultimate * mtow) ** (-0.187)
            * mtow
        )
        partials["data:weight:airframe:horizontal_tail:mass", "data:weight:aircraft:MTOW"] = (
            k_factor_ht
            * (
                0.0034
                * 0.915
                * (area_ht**0.584 * (span_ht / root_thickness) ** 0.033 * (mac_ht / lp_ht) ** 0.28)
                ** 0.915
                * (sizing_factor_ultimate * mtow) ** (-0.069105)
                * 0.813
                * (sizing_factor_ultimate * mtow) ** (-0.187)
                * sizing_factor_ultimate
            )
        )
        partials[
            "data:weight:airframe:horizontal_tail:mass", "data:geometry:horizontal_tail:area"
        ] = k_factor_ht * (
            0.0034
            * 0.915
            * (
                (sizing_factor_ultimate * mtow) ** 0.813
                * (span_ht / root_thickness) ** 0.033
                * (mac_ht / lp_ht) ** 0.28
            )
            ** 0.915
            * area_ht ** (-0.04964)
            * 0.584
            * area_ht ** (-0.416)
        )
        partials[
            "data:weight:airframe:horizontal_tail:mass", "data:geometry:horizontal_tail:span"
        ] = k_factor_ht * (
            0.0034
            * 0.915
            * (
                (sizing_factor_ultimate * mtow) ** 0.813
                * area_ht**0.584
                * 1/root_thickness**0.033
                * (mac_ht / lp_ht) ** 0.28
            )
            ** 0.915
            * span_ht ** (-0.002805)
            * 0.033
            * span_ht ** (-0.967)
        )
        partials[
            "data:weight:airframe:horizontal_tail:mass",
            "data:geometry:horizontal_tail:thickness_ratio",
        ] = k_factor_ht * (
            0.0034
            * 0.915
            * (
                (sizing_factor_ultimate * mtow) ** 0.813
                * area_ht**0.584
                * (span_ht / root_chord_ht) ** 0.033
                * (mac_ht / lp_ht) ** 0.28
            )
            ** 0.915
            * t_c_ht**0.002805
            * (-0.033 * t_c_ht ** (-1.033))
        )
        partials[
            "data:weight:airframe:horizontal_tail:mass", "data:geometry:horizontal_tail:root:chord"
        ] = k_factor_ht * (
            0.0034
            * 0.915
            * (
                (sizing_factor_ultimate * mtow) ** 0.813
                * area_ht**0.584
                * (span_ht / t_c_ht) ** 0.033
                * (mac_ht / lp_ht) ** 0.28
            )
            ** 0.915
            * root_chord_ht**0.002805
            * (-0.033 * root_chord_ht**-1.033)
        )
        partials[
            "data:weight:airframe:horizontal_tail:mass", "data:geometry:horizontal_tail:MAC:length"
        ] = k_factor_ht * (
            0.0034
            * 0.915
            * (
                (sizing_factor_ultimate * mtow) ** 0.813
                * area_ht**0.584
                * (span_ht / root_thickness) ** 0.033
                * lp_ht**-0.28
            )
            ** 0.915
            * mac_ht ** (-0.0238)
            * 0.28
            * mac_ht ** (-0.72)
        )
        partials[
            "data:weight:airframe:horizontal_tail:mass",
            "data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25",
        ] = k_factor_ht * (
            0.0034
            * 0.915
            * (
                (sizing_factor_ultimate * mtow) ** 0.813
                * area_ht**0.584
                * (span_ht / root_thickness) ** 0.033
                * mac_ht**0.28
            )
            ** 0.915
            * lp_ht**0.0238
            * (-0.28 * lp_ht**-1.28)
        )
        partials[
            "data:weight:airframe:horizontal_tail:mass",
            "data:weight:airframe:horizontal_tail:k_factor",
        ] = (
            0.0034
            * (
                (sizing_factor_ultimate * mtow) ** 0.813
                * area_ht**0.584
                * (span_ht / root_thickness) ** 0.033
                * (mac_ht / lp_ht) ** 0.28
            )
            ** 0.915
        )

        partials[
            "data:weight:airframe:vertical_tail:mass",
            "data:mission:sizing:cs23:sizing_factor:ultimate_aircraft",
        ] = k_factor_vt * (
            0.19
            * 1.014
            * (
                (1 + has_t_tail) ** 0.5
                * area_vt**1.089
                * mach_h**0.601
                * lp_vt**-0.726
                * (1 + rudder_chord_ratio) ** 0.217
                * ar_vt**0.337
                * (1 + taper_vt) ** 0.363
                * np.cos(sweep_25_vt) ** -0.484
            )
            ** 1.014
            * (sizing_factor_ultimate * mtow) ** 0.005082
            * 0.363
            * (sizing_factor_ultimate * mtow) ** (-0.637)
            * mtow
        )
        partials["data:weight:airframe:vertical_tail:mass", "data:weight:aircraft:MTOW"] = (
            k_factor_vt
            * (
                0.19
                * 1.014
                * (
                    (1 + has_t_tail) ** 0.5
                    * area_vt**1.089
                    * mach_h**0.601
                    * lp_vt**-0.726
                    * (1 + rudder_chord_ratio) ** 0.217
                    * ar_vt**0.337
                    * (1 + taper_vt) ** 0.363
                    * np.cos(sweep_25_vt) ** -0.484
                )
                ** 1.014
                * (sizing_factor_ultimate * mtow) ** 0.005082
                * 0.363
                * (sizing_factor_ultimate * mtow) ** (-0.637)
                * sizing_factor_ultimate
            )
        )
        partials["data:weight:airframe:vertical_tail:mass", "data:geometry:has_T_tail"] = (
            k_factor_vt
            * (
                0.19
                * 1.014
                * (
                    (sizing_factor_ultimate * mtow) ** 0.363
                    * area_vt**1.089
                    * mach_h**0.601
                    * lp_vt**-0.726
                    * (1 + rudder_chord_ratio) ** 0.217
                    * ar_vt**0.337
                    * (1 + taper_vt) ** 0.363
                    * np.cos(sweep_25_vt) ** -0.484
                )
                ** 1.014
            )
            * (1 + has_t_tail) ** 0.007
            * 0.5
            * (1 + has_t_tail) ** -0.5
        )
        partials["data:weight:airframe:vertical_tail:mass", "data:geometry:vertical_tail:area"] = (
            k_factor_vt
            * (
                0.19
                * 1.014
                * (
                    (1 + has_t_tail) ** 0.5
                    * (sizing_factor_ultimate * mtow) ** 0.363
                    * mach_h**0.601
                    * lp_vt**-0.726
                    * (1 + rudder_chord_ratio) ** 0.217
                    * ar_vt**0.337
                    * (1 + taper_vt) ** 0.363
                    * np.cos(sweep_25_vt) ** -0.484
                )
                ** 1.014
                * area_vt**0.015246
                * 1.089
                * area_vt**0.089
            )
        )
        partials[
            "data:weight:airframe:vertical_tail:mass",
            "data:geometry:vertical_tail:MAC:at25percent:x:from_wingMAC25",
        ] = k_factor_vt * (
            0.19
            * 1.014
            * (
                (1 + has_t_tail) ** 0.5
                * (sizing_factor_ultimate * mtow) ** 0.363
                * area_vt**1.089
                * mach_h**0.601
                * (1 + rudder_chord_ratio) ** 0.217
                * ar_vt**0.337
                * (1 + taper_vt) ** 0.363
                * np.cos(sweep_25_vt) ** -0.484
            )
            ** 1.014
            * lp_vt ** (-0.010164)
            * (-0.726 * lp_vt**-1.726)
        )
        partials[
            "data:weight:airframe:vertical_tail:mass",
            "data:geometry:vertical_tail:rudder:chord_ratio",
        ] = k_factor_vt * (
            0.19
            * 1.014
            * (
                (1 + has_t_tail) ** 0.5
                * (sizing_factor_ultimate * mtow) ** 0.363
                * area_vt**1.089
                * mach_h**0.601
                * lp_vt**-0.726
                * ar_vt**0.337
                * (1 + taper_vt) ** 0.363
                * np.cos(sweep_25_vt) ** -0.484
            )
            ** 1.014
            * (1 + rudder_chord_ratio) ** 0.003038
            * 0.217
            * (1 + rudder_chord_ratio) ** (-0.783)
        )
        partials[
            "data:weight:airframe:vertical_tail:mass", "data:geometry:vertical_tail:sweep_25"
        ] = k_factor_vt * (
            0.19
            * 1.014
            * (
                (1 + has_t_tail) ** 0.5
                * (sizing_factor_ultimate * mtow) ** 0.363
                * area_vt**1.089
                * mach_h**0.601
                * lp_vt**-0.726
                * (1 + rudder_chord_ratio) ** 0.217
                * ar_vt**0.337
                * (1 + taper_vt) ** 0.363
            )
            ** 1.014
            * np.cos(sweep_25_vt) ** (-0.006776)
            * 0.484
            * np.cos(sweep_25_vt) ** -1.484
            * np.sin(sweep_25_vt)
        )
        partials[
            "data:weight:airframe:vertical_tail:mass", "data:geometry:vertical_tail:aspect_ratio"
        ] = k_factor_vt * (
            0.19
            * 1.014
            * (
                (1 + has_t_tail) ** 0.5
                * (sizing_factor_ultimate * mtow) ** 0.363
                * area_vt**1.089
                * mach_h**0.601
                * lp_vt**-0.726
                * (1 + rudder_chord_ratio) ** 0.217
                * (1 + taper_vt) ** 0.363
                * np.cos(sweep_25_vt) ** -0.484
            )
            ** 1.014
            * ar_vt**0.004718
            * 0.337
            * ar_vt ** (-0.663)
        )
        partials[
            "data:weight:airframe:vertical_tail:mass", "data:geometry:vertical_tail:taper_ratio"
        ] = k_factor_vt * (
            0.19
            * 1.014
            * (
                (1 + has_t_tail) ** 0.5
                * (sizing_factor_ultimate * mtow) ** 0.363
                * area_vt**1.089
                * mach_h**0.601
                * lp_vt**-0.726
                * (1 + rudder_chord_ratio) ** 0.217
                * ar_vt**0.337
                * np.cos(sweep_25_vt) ** -0.484
            )
            ** 1.014
            * (1 + taper_vt) ** 0.005082
            * 0.363
            * (1 + taper_vt) ** (-0.637)
        )

        partials["data:weight:airframe:vertical_tail:mass", "data:TLAR:v_max_sl"] = k_factor_vt * (
            0.19
            * 1.014
            * (
                (1 + has_t_tail) ** 0.5
                * (sizing_factor_ultimate * mtow) ** 0.363
                * area_vt**1.089
                * lp_vt**-0.726
                * (1 + rudder_chord_ratio) ** 0.217
                * ar_vt**0.337
                * (1 + taper_vt) ** 0.363
                * np.cos(sweep_25_vt) ** -0.484
            )
            ** 1.014
            * mach_h**0.008414
            * 0.601
            * mach_h ** (-0.399)
            / atm0.speed_of_sound
        )
        partials[
            "data:weight:airframe:vertical_tail:mass", "data:weight:airframe:vertical_tail:k_factor"
        ] = (
            0.19
            * (
                (1 + has_t_tail) ** 0.5
                * (sizing_factor_ultimate * mtow) ** 0.363
                * area_vt**1.089
                * mach_h**0.601
                * lp_vt**-0.726
                * (1 + rudder_chord_ratio) ** 0.217
                * ar_vt**0.337
                * (1 + taper_vt) ** 0.363
                * np.cos(sweep_25_vt) ** -0.484
            )
            ** 1.014
        )


@oad.RegisterSubmodel(SUBMODEL_TAIL_MASS, "fastga.submodel.weight.mass.airframe.tail.torenbeek_gd")
class ComputeTailWeightTorenbeekGD(om.ExplicitComponent):
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
        self.add_input("data:weight:airframe:horizontal_tail:k_factor", val=1.0)
        self.add_input("data:weight:airframe:vertical_tail:k_factor", val=1.0)

        self.add_input("data:geometry:horizontal_tail:area", val=np.nan, units="ft**2")
        self.add_input("data:geometry:horizontal_tail:sweep_25", val=np.nan, units="rad")

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

        self.add_input("data:mission:sizing:cs23:characteristic_speed:vd", units="kn")

        self.add_output("data:weight:airframe:horizontal_tail:mass", units="lb")
        self.add_output("data:weight:airframe:vertical_tail:mass", units="lb")

        self.declare_partials(
            of="data:weight:airframe:horizontal_tail:mass",
            wrt=[
                "data:geometry:horizontal_tail:area",
                "data:weight:airframe:horizontal_tail:k_factor",
                "data:mission:sizing:cs23:characteristic_speed:vd",
                "data:geometry:horizontal_tail:sweep_25",
            ],
            method="exact",
        )
        self.declare_partials(
            of="data:weight:airframe:vertical_tail:mass",
            wrt=[
                "data:geometry:has_T_tail",
                "data:weight:aircraft:MTOW",
                "data:mission:sizing:cs23:sizing_factor:ultimate_aircraft",
                "data:TLAR:v_max_sl",
                "data:geometry:vertical_tail:area",
                "data:geometry:vertical_tail:MAC:at25percent:x:from_wingMAC25",
                "data:geometry:vertical_tail:rudder:chord_ratio",
                "data:geometry:vertical_tail:aspect_ratio",
                "data:geometry:vertical_tail:taper_ratio",
                "data:geometry:vertical_tail:sweep_25",
                "data:weight:airframe:vertical_tail:k_factor",
            ],
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        sizing_factor_ultimate = inputs["data:mission:sizing:cs23:sizing_factor:ultimate_aircraft"]
        mtow = inputs["data:weight:aircraft:MTOW"]

        area_ht = inputs["data:geometry:horizontal_tail:area"]
        vd = inputs["data:mission:sizing:cs23:characteristic_speed:vd"]
        sweep_25 = inputs["data:geometry:horizontal_tail:sweep_25"]

        a31 = area_ht * (3.81 * area_ht**0.2 * vd / (1000.0 * np.cos(sweep_25)) - 0.287)
        # Mass formula in lb

        outputs["data:weight:airframe:horizontal_tail:mass"] = (
            a31 * inputs["data:weight:airframe:horizontal_tail:k_factor"]
        )

        has_t_tail = inputs["data:geometry:has_T_tail"]
        area_vt = inputs["data:geometry:vertical_tail:area"]
        v_h = inputs["data:TLAR:v_max_sl"]
        atm0 = Atmosphere(0)
        atm0.true_airspeed = v_h
        mach_h = atm0.mach
        lp_vt = inputs["data:geometry:vertical_tail:MAC:at25percent:x:from_wingMAC25"]
        rudder_chord_ratio = inputs["data:geometry:vertical_tail:rudder:chord_ratio"]
        sweep_25_vt = inputs["data:geometry:vertical_tail:sweep_25"]
        ar_vt = inputs["data:geometry:vertical_tail:aspect_ratio"]
        taper_vt = inputs["data:geometry:vertical_tail:taper_ratio"]

        a32 = (
            0.19
            * (
                (1 + has_t_tail) ** 0.5
                * (sizing_factor_ultimate * mtow) ** 0.363
                * area_vt**1.089
                * mach_h**0.601
                * lp_vt**-0.726
                * (1 + rudder_chord_ratio) ** 0.217
                * ar_vt**0.337
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

        area_ht = inputs["data:geometry:horizontal_tail:area"]
        vd = inputs["data:mission:sizing:cs23:characteristic_speed:vd"]
        sweep_25 = inputs["data:geometry:horizontal_tail:sweep_25"]
        k_factor_ht = inputs["data:weight:airframe:horizontal_tail:k_factor"]

        has_t_tail = inputs["data:geometry:has_T_tail"]
        area_vt = inputs["data:geometry:vertical_tail:area"]
        v_h = inputs["data:TLAR:v_max_sl"]
        lp_vt = inputs["data:geometry:vertical_tail:MAC:at25percent:x:from_wingMAC25"]
        rudder_chord_ratio = inputs["data:geometry:vertical_tail:rudder:chord_ratio"]
        sweep_25_vt = inputs["data:geometry:vertical_tail:sweep_25"]
        ar_vt = inputs["data:geometry:vertical_tail:aspect_ratio"]
        taper_vt = inputs["data:geometry:vertical_tail:taper_ratio"]

        k_factor_vt = inputs["data:weight:airframe:vertical_tail:k_factor"]

        atm0 = AtmosphereWithPartials(0)
        mach_h = v_h / atm0.speed_of_sound

        partials[
            "data:weight:airframe:horizontal_tail:mass", "data:geometry:horizontal_tail:area"
        ] = k_factor_ht * ((0.004572 * area_ht**0.2 * vd) / np.cos(sweep_25) - 0.287)
        partials[
            "data:weight:airframe:horizontal_tail:mass",
            "data:mission:sizing:cs23:characteristic_speed:vd",
        ] = k_factor_ht * ((0.00381 * area_ht**1.2) / np.cos(sweep_25))
        partials[
            "data:weight:airframe:horizontal_tail:mass", "data:geometry:horizontal_tail:sweep_25"
        ] = k_factor_ht * (
            (0.00381 * area_ht**1.2 * vd * np.sin(sweep_25)) / np.cos(sweep_25) ** 2.0
        )
        partials[
            "data:weight:airframe:horizontal_tail:mass",
            "data:weight:airframe:horizontal_tail:k_factor",
        ] = area_ht * (3.81 * area_ht**0.2 * vd / (1000.0 * np.cos(sweep_25)) - 0.287)

        partials[
            "data:weight:airframe:vertical_tail:mass",
            "data:mission:sizing:cs23:sizing_factor:ultimate_aircraft",
        ] = k_factor_vt * (
            0.19
            * 1.014
            * (
                (1 + has_t_tail) ** 0.5
                * area_vt**1.089
                * mach_h**0.601
                * lp_vt**-0.726
                * (1 + rudder_chord_ratio) ** 0.217
                * ar_vt**0.337
                * (1 + taper_vt) ** 0.363
                * np.cos(sweep_25_vt) ** -0.484
            )
            ** 1.014
            * (sizing_factor_ultimate * mtow) ** 0.005082
            * 0.363
            * (sizing_factor_ultimate * mtow) ** (-0.637)
            * mtow
        )
        partials["data:weight:airframe:vertical_tail:mass", "data:weight:aircraft:MTOW"] = (
            k_factor_vt
            * (
                0.19
                * 1.014
                * (
                    (1 + has_t_tail) ** 0.5
                    * area_vt**1.089
                    * mach_h**0.601
                    * lp_vt**-0.726
                    * (1 + rudder_chord_ratio) ** 0.217
                    * ar_vt**0.337
                    * (1 + taper_vt) ** 0.363
                    * np.cos(sweep_25_vt) ** -0.484
                )
                ** 1.014
                * (sizing_factor_ultimate * mtow) ** 0.005082
                * 0.363
                * (sizing_factor_ultimate * mtow) ** (-0.637)
                * sizing_factor_ultimate
            )
        )
        partials["data:weight:airframe:vertical_tail:mass", "data:geometry:has_T_tail"] = (
            k_factor_vt
            * (
                0.19
                * 1.014
                * (
                    (sizing_factor_ultimate * mtow) ** 0.363
                    * area_vt**1.089
                    * mach_h**0.601
                    * lp_vt**-0.726
                    * (1 + rudder_chord_ratio) ** 0.217
                    * ar_vt**0.337
                    * (1 + taper_vt) ** 0.363
                    * np.cos(sweep_25_vt) ** -0.484
                )
                ** 1.014
            )
            * (1 + has_t_tail) ** 0.007
            * 0.5
            * (1 + has_t_tail) ** -0.5
        )
        partials["data:weight:airframe:vertical_tail:mass", "data:geometry:vertical_tail:area"] = (
            k_factor_vt
            * (
                0.19
                * 1.014
                * (
                    (1 + has_t_tail) ** 0.5
                    * (sizing_factor_ultimate * mtow) ** 0.363
                    * mach_h**0.601
                    * lp_vt**-0.726
                    * (1 + rudder_chord_ratio) ** 0.217
                    * ar_vt**0.337
                    * (1 + taper_vt) ** 0.363
                    * np.cos(sweep_25_vt) ** -0.484
                )
                ** 1.014
                * area_vt**0.015246
                * 1.089
                * area_vt**0.089
            )
        )
        partials[
            "data:weight:airframe:vertical_tail:mass",
            "data:geometry:vertical_tail:MAC:at25percent:x:from_wingMAC25",
        ] = k_factor_vt * (
            0.19
            * 1.014
            * (
                (1 + has_t_tail) ** 0.5
                * (sizing_factor_ultimate * mtow) ** 0.363
                * area_vt**1.089
                * mach_h**0.601
                * (1 + rudder_chord_ratio) ** 0.217
                * ar_vt**0.337
                * (1 + taper_vt) ** 0.363
                * np.cos(sweep_25_vt) ** -0.484
            )
            ** 1.014
            * lp_vt ** (-0.010164)
            * (-0.726 * lp_vt**-1.726)
        )
        partials[
            "data:weight:airframe:vertical_tail:mass",
            "data:geometry:vertical_tail:rudder:chord_ratio",
        ] = k_factor_vt * (
            0.19
            * 1.014
            * (
                (1 + has_t_tail) ** 0.5
                * (sizing_factor_ultimate * mtow) ** 0.363
                * area_vt**1.089
                * mach_h**0.601
                * lp_vt**-0.726
                * ar_vt**0.337
                * (1 + taper_vt) ** 0.363
                * np.cos(sweep_25_vt) ** -0.484
            )
            ** 1.014
            * (1 + rudder_chord_ratio) ** 0.003038
            * 0.217
            * (1 + rudder_chord_ratio) ** (-0.783)
        )
        partials[
            "data:weight:airframe:vertical_tail:mass", "data:geometry:vertical_tail:sweep_25"
        ] = k_factor_vt * (
            0.19
            * 1.014
            * (
                (1 + has_t_tail) ** 0.5
                * (sizing_factor_ultimate * mtow) ** 0.363
                * area_vt**1.089
                * mach_h**0.601
                * lp_vt**-0.726
                * (1 + rudder_chord_ratio) ** 0.217
                * ar_vt**0.337
                * (1 + taper_vt) ** 0.363
            )
            ** 1.014
            * np.cos(sweep_25_vt) ** (-0.006776)
            * 0.484
            * np.cos(sweep_25_vt) ** -1.484
            * np.sin(sweep_25_vt)
        )
        partials[
            "data:weight:airframe:vertical_tail:mass", "data:geometry:vertical_tail:aspect_ratio"
        ] = k_factor_vt * (
            0.19
            * 1.014
            * (
                (1 + has_t_tail) ** 0.5
                * (sizing_factor_ultimate * mtow) ** 0.363
                * area_vt**1.089
                * mach_h**0.601
                * lp_vt**-0.726
                * (1 + rudder_chord_ratio) ** 0.217
                * (1 + taper_vt) ** 0.363
                * np.cos(sweep_25_vt) ** -0.484
            )
            ** 1.014
            * ar_vt ** (0.337 * 0.014)
            * 0.337
            * ar_vt ** (-0.663)
        )
        partials[
            "data:weight:airframe:vertical_tail:mass", "data:geometry:vertical_tail:taper_ratio"
        ] = k_factor_vt * (
            0.19
            * 1.014
            * (
                (1 + has_t_tail) ** 0.5
                * (sizing_factor_ultimate * mtow) ** 0.363
                * area_vt**1.089
                * mach_h**0.601
                * lp_vt**-0.726
                * (1 + rudder_chord_ratio) ** 0.217
                * ar_vt**0.337
                * np.cos(sweep_25_vt) ** -0.484
            )
            ** 1.014
            * (1 + taper_vt) ** 0.005082
            * 0.363
            * (1 + taper_vt) ** (-0.637)
        )

        partials["data:weight:airframe:vertical_tail:mass", "data:TLAR:v_max_sl"] = k_factor_vt * (
            0.19
            * 1.014
            * (
                (1 + has_t_tail) ** 0.5
                * (sizing_factor_ultimate * mtow) ** 0.363
                * area_vt**1.089
                * lp_vt**-0.726
                * (1 + rudder_chord_ratio) ** 0.217
                * ar_vt**0.337
                * (1 + taper_vt) ** 0.363
                * np.cos(sweep_25_vt) ** -0.484
            )
            ** 1.014
            * mach_h**0.008414
            * 0.601
            * mach_h ** (-0.399)
            / atm0.speed_of_sound
        )
        partials[
            "data:weight:airframe:vertical_tail:mass", "data:weight:airframe:vertical_tail:k_factor"
        ] = (
            0.19
            * (
                (1 + has_t_tail) ** 0.5
                * (sizing_factor_ultimate * mtow) ** 0.363
                * area_vt**1.089
                * mach_h**0.601
                * lp_vt**-0.726
                * (1 + rudder_chord_ratio) ** 0.217
                * ar_vt**0.337
                * (1 + taper_vt) ** 0.363
                * np.cos(sweep_25_vt) ** -0.484
            )
            ** 1.014
        )
