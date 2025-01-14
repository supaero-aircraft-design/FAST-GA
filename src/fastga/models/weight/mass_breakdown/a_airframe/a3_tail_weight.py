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
from stdatm import Atmosphere

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
        self.add_input("data:geometry:horizontal_tail:sweep_25", val=np.nan, units="deg")
        self.add_input("data:geometry:horizontal_tail:aspect_ratio", val=np.nan)
        self.add_input("data:geometry:horizontal_tail:taper_ratio", val=np.nan)

        self.add_input("data:geometry:has_T_tail", val=np.nan)
        self.add_input("data:geometry:vertical_tail:area", val=np.nan, units="ft**2")
        self.add_input("data:geometry:vertical_tail:thickness_ratio", val=np.nan)
        self.add_input("data:geometry:vertical_tail:sweep_25", val=np.nan, units="deg")
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
            ],
            method="exact",
        )
        # TODO: Try to make it exact
        self.declare_partials(
            of="*", wrt="data:mission:sizing:main_route:cruise:altitude", method="fd", step=1.0e2
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
            * (100.0 * t_c_ht / np.cos(sweep_25_ht * np.pi / 180.0)) ** -0.12
            * (ar_ht / (np.cos(sweep_25_ht * np.pi / 180.0)) ** 2.0) ** 0.043
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
                * (100.0 * t_c_vt / np.cos(sweep_25_vt * np.pi / 180.0)) ** -0.49
                * (ar_vt / (np.cos(sweep_25_vt * np.pi / 180.0)) ** 2.0) ** 0.357
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

        rho_cruise = Atmosphere(cruise_alt, altitude_in_feet=True).density
        dynamic_pressure = 1.0 / 2.0 * rho_cruise * v_cruise_ktas**2.0 * 0.0208854
        # In lb/ft2

        a31 = 0.016 * (
            (sizing_factor_ultimate * mtow) ** 0.414
            * dynamic_pressure**0.168
            * area_ht**0.896
            * (100.0 * t_c_ht / np.cos(sweep_25_ht * np.pi / 180.0)) ** -0.12
            * (ar_ht / (np.cos(sweep_25_ht * np.pi / 180.0)) ** 2.0) ** 0.043
            * taper_ht**-0.02
        )
        # Mass formula in lb

        a32 = (
            0.073
            * (1.0 + 0.2 * has_t_tail)
            * (
                (sizing_factor_ultimate * mtow) ** 0.376
                * dynamic_pressure**0.122
                * area_vt**0.873
                * (100.0 * t_c_vt / np.cos(sweep_25_vt * np.pi / 180.0)) ** -0.49
                * (ar_vt / (np.cos(sweep_25_vt * np.pi / 180.0)) ** 2.0) ** 0.357
                * taper_vt**0.039
            )
        )
        # Mass formula in lb

        partials[
            "data:weight:airframe:horizontal_tail:mass",
            "data:mission:sizing:cs23:sizing_factor:ultimate_aircraft",
        ] = k_factor_ht * (
            (
                0.006624
                * area_ht**0.896
                * mtow
                * (ar_ht / np.cos(np.pi / 180.0 * sweep_25_ht) ** 2) ** 0.043
                * (0.010443 * rho_cruise * v_cruise_ktas**2) ** 0.168
            )
            / (
                taper_ht**0.02
                * (mtow * sizing_factor_ultimate) ** 0.586
                * ((100.0 * t_c_ht) / np.cos(np.pi / 180.0 * sweep_25_ht)) ** 0.12
            )
        )
        partials["data:weight:airframe:horizontal_tail:mass", "data:weight:aircraft:MTOW"] = (
            k_factor_ht
            * (
                (
                    0.006624
                    * area_ht**0.896
                    * sizing_factor_ultimate
                    * (ar_ht / np.cos(np.pi / 180.0 * sweep_25_ht) ** 2) ** 0.043
                    * (0.010443 * rho_cruise * v_cruise_ktas**2) ** 0.168
                )
                / (
                    taper_ht**0.02
                    * (mtow * sizing_factor_ultimate) ** 0.586
                    * ((100.0 * t_c_ht) / np.cos(np.pi / 180.0 * sweep_25_ht)) ** 0.12
                )
            )
        )
        partials["data:weight:airframe:horizontal_tail:mass", "data:TLAR:v_cruise"] = (
            k_factor_ht
            * (
                (
                    5.614e-5
                    * area_ht**0.896
                    * rho_cruise
                    * v_cruise_ktas
                    * (mtow * sizing_factor_ultimate) ** 0.414
                    * (ar_ht / np.cos(np.pi / 180.0 * sweep_25_ht) ** 2) ** 0.043
                )
                / (
                    taper_ht**0.02
                    * ((100.0 * t_c_ht) / np.cos(np.pi / 180.0 * sweep_25_ht)) ** 0.12
                    * (0.010443 * rho_cruise * v_cruise_ktas**2) ** 0.832
                )
            )
        )
        partials[
            "data:weight:airframe:horizontal_tail:mass", "data:geometry:horizontal_tail:area"
        ] = k_factor_ht * (
            (
                0.014336
                * (mtow * sizing_factor_ultimate) ** 0.414
                * (ar_ht / np.cos(np.pi / 180.0 * sweep_25_ht) ** 2) ** 0.043
                * (0.010443 * rho_cruise * v_cruise_ktas**2) ** 0.168
            )
            / (
                area_ht ** (13 / 125)
                * taper_ht**0.02
                * ((100.0 * t_c_ht) / np.cos(np.pi / 180.0 * sweep_25_ht)) ** 0.12
            )
        )
        partials[
            "data:weight:airframe:horizontal_tail:mass",
            "data:geometry:horizontal_tail:thickness_ratio",
        ] = k_factor_ht * (
            -(
                0.192
                * area_ht**0.896
                * (mtow * sizing_factor_ultimate) ** 0.414
                * (ar_ht / np.cos(np.pi / 180.0 * sweep_25_ht) ** 2) ** 0.043
                * (0.010443 * rho_cruise * v_cruise_ktas**2) ** 0.168
            )
            / (
                taper_ht**0.02
                * np.cos(np.pi / 180.0 * sweep_25_ht)
                * ((100.0 * t_c_ht) / np.cos(np.pi / 180.0 * sweep_25_ht)) ** 1.12
            )
        )
        partials[
            "data:weight:airframe:horizontal_tail:mass", "data:geometry:horizontal_tail:sweep_25"
        ] = k_factor_ht * (
            (
                2.4016e-5
                * ar_ht
                * area_ht**0.896
                * np.sin(np.pi / 180.0 * sweep_25_ht)
                * (mtow * sizing_factor_ultimate) ** 0.414
                * (0.010443 * rho_cruise * v_cruise_ktas**2) ** 0.168
            )
            / (
                taper_ht**0.02
                * np.cos(np.pi / 180.0 * sweep_25_ht) ** 3
                * (ar_ht / np.cos(np.pi / 180.0 * sweep_25_ht) ** 2) ** 0.957
                * ((100.0 * t_c_ht) / np.cos(np.pi / 180.0 * sweep_25_ht)) ** 0.12
            )
            - (
                0.003351
                * area_ht**0.896
                * t_c_ht
                * np.sin(np.pi / 180.0 * sweep_25_ht)
                * (mtow * sizing_factor_ultimate) ** 0.414
                * (ar_ht / np.cos(np.pi / 180.0 * sweep_25_ht) ** 2) ** 0.043
                * (0.010443 * rho_cruise * v_cruise_ktas**2) ** 0.168
            )
            / (
                taper_ht**0.02
                * np.cos(np.pi / 180.0 * sweep_25_ht) ** 2
                * ((100.0 * t_c_ht) / np.cos(np.pi / 180.0 * sweep_25_ht)) ** 1.12
            )
        )
        partials[
            "data:weight:airframe:horizontal_tail:mass",
            "data:geometry:horizontal_tail:aspect_ratio",
        ] = k_factor_ht * (
            (
                6.88e-4
                * area_ht**0.896
                * (mtow * sizing_factor_ultimate) ** 0.414
                * (0.010443 * rho_cruise * v_cruise_ktas**2) ** 0.168
            )
            / (
                taper_ht**0.02
                * np.cos(np.pi / 180.0 * sweep_25_ht) ** 2
                * (ar_ht / np.cos(np.pi / 180.0 * sweep_25_ht) ** 2) ** 0.957
                * ((100.0 * t_c_ht) / np.cos(np.pi / 180.0 * sweep_25_ht)) ** 0.12
            )
        )
        partials[
            "data:weight:airframe:horizontal_tail:mass", "data:geometry:horizontal_tail:taper_ratio"
        ] = k_factor_ht * (
            -(
                3.2e-4
                * area_ht**0.896
                * (mtow * sizing_factor_ultimate) ** 0.414
                * (ar_ht / np.cos(np.pi / 180.0 * sweep_25_ht) ** 2) ** 0.043
                * (0.010443 * rho_cruise * v_cruise_ktas**2) ** 0.168
            )
            / (taper_ht**1.02 * ((100.0 * t_c_ht) / np.cos(np.pi / 180.0 * sweep_25_ht)) ** 0.12)
        )
        partials[
            "data:weight:airframe:horizontal_tail:mass",
            "data:weight:airframe:horizontal_tail:k_factor",
        ] = a31

        partials[
            "data:weight:airframe:vertical_tail:mass",
            "data:mission:sizing:cs23:sizing_factor:ultimate_aircraft",
        ] = k_factor_vt * (
            (
                0.376
                * area_vt**0.873
                * mtow
                * taper_vt**0.039
                * (ar_vt / np.cos(np.pi / 180.0 * sweep_25_vt) ** 2) ** 0.357
                * (0.0146 * has_t_tail + 0.073)
                * (0.010443 * rho_cruise * v_cruise_ktas**2) ** 0.122
            )
            / (
                (mtow * sizing_factor_ultimate) ** 0.624
                * ((100.0 * t_c_vt) / np.cos(np.pi / 180.0 * sweep_25_vt)) ** 0.49
            )
        )
        partials["data:weight:airframe:vertical_tail:mass", "data:weight:aircraft:MTOW"] = (
            k_factor_vt
            * (
                (
                    0.376
                    * area_vt**0.873
                    * sizing_factor_ultimate
                    * taper_vt**0.039
                    * (ar_vt / np.cos(np.pi / 180.0 * sweep_25_vt) ** 2) ** 0.357
                    * (0.0146 * has_t_tail + 0.073)
                    * (0.010443 * rho_cruise * v_cruise_ktas**2) ** 0.122
                )
                / (
                    (mtow * sizing_factor_ultimate) ** 0.624
                    * ((100.0 * t_c_vt) / np.cos(np.pi / 180.0 * sweep_25_vt)) ** 0.49
                )
            )
        )
        partials["data:weight:airframe:vertical_tail:mass", "data:TLAR:v_cruise"] = k_factor_vt * (
            (
                0.002548
                * area_vt**0.873
                * rho_cruise
                * taper_vt**0.039
                * v_cruise_ktas
                * (mtow * sizing_factor_ultimate) ** 0.376
                * (ar_vt / np.cos(np.pi / 180.0 * sweep_25_vt) ** 2) ** 0.357
                * (0.0146 * has_t_tail + 0.073)
            )
            / (
                ((100.0 * t_c_vt) / np.cos(np.pi / 180.0 * sweep_25_vt)) ** 0.49
                * (0.010443 * rho_cruise * v_cruise_ktas**2) ** 0.878
            )
        )
        partials["data:weight:airframe:vertical_tail:mass", "data:geometry:has_T_tail"] = (
            k_factor_vt
            * (
                (
                    0.0146
                    * area_vt**0.873
                    * taper_vt**0.039
                    * (mtow * sizing_factor_ultimate) ** 0.376
                    * (ar_vt / np.cos(np.pi / 180.0 * sweep_25_vt) ** 2) ** 0.357
                    * (0.010443 * rho_cruise * v_cruise_ktas**2) ** 0.122
                )
                / ((100.0 * t_c_vt) / np.cos(np.pi / 180.0 * sweep_25_vt)) ** 0.49
            )
        )
        partials["data:weight:airframe:vertical_tail:mass", "data:geometry:vertical_tail:area"] = (
            k_factor_vt
            * (
                (
                    0.873
                    * taper_vt**0.039
                    * (mtow * sizing_factor_ultimate) ** 0.376
                    * (ar_vt / np.cos(np.pi / 180.0 * sweep_25_vt) ** 2) ** 0.357
                    * (0.0146 * has_t_tail + 0.073)
                    * (0.010443 * rho_cruise * v_cruise_ktas**2) ** 0.122
                )
                / (
                    area_vt ** (127 / 1000)
                    * ((100.0 * t_c_vt) / np.cos(np.pi / 180.0 * sweep_25_vt)) ** 0.49
                )
            )
        )
        partials[
            "data:weight:airframe:vertical_tail:mass", "data:geometry:vertical_tail:thickness_ratio"
        ] = k_factor_vt * (
            -(
                49.0
                * area_vt**0.873
                * taper_vt**0.039
                * (mtow * sizing_factor_ultimate) ** 0.376
                * (ar_vt / np.cos(np.pi / 180.0 * sweep_25_vt) ** 2) ** 0.357
                * (0.0146 * has_t_tail + 0.073)
                * (0.0104427 * rho_cruise * v_cruise_ktas**2) ** 0.122
            )
            / (
                np.cos(np.pi / 180.0 * sweep_25_vt)
                * ((100.0 * t_c_vt) / np.cos(np.pi / 180.0 * sweep_25_vt)) ** 1.49
            )
        )
        partials[
            "data:weight:airframe:vertical_tail:mass", "data:geometry:vertical_tail:sweep_25"
        ] = k_factor_vt * (
            (
                0.012462
                * ar_vt
                * area_vt**0.873
                * taper_vt**0.039
                * np.sin(np.pi / 180.0 * sweep_25_vt)
                * (mtow * sizing_factor_ultimate) ** 0.376
                * (0.0146 * has_t_tail + 0.073)
                * (0.010443 * rho_cruise * v_cruise_ktas**2) ** 0.122
            )
            / (
                np.cos(np.pi / 180.0 * sweep_25_vt) ** 3
                * (ar_vt / np.cos(np.pi / 180.0 * sweep_25_vt) ** 2) ** 0.643
                * ((100.0 * t_c_vt) / np.cos(np.pi / 180.0 * sweep_25_vt)) ** 0.49
            )
            - (
                0.85521
                * area_vt**0.873
                * t_c_vt
                * taper_vt**0.039
                * np.sin(np.pi / 180.0 * sweep_25_vt)
                * (mtow * sizing_factor_ultimate) ** 0.376
                * (ar_vt / np.cos(np.pi / 180.0 * sweep_25_vt) ** 2) ** 0.357
                * (0.0146 * has_t_tail + 0.073)
                * (0.010443 * rho_cruise * v_cruise_ktas**2) ** 0.122
            )
            / (
                np.cos(np.pi / 180.0 * sweep_25_vt) ** 2
                * ((100.0 * t_c_vt) / np.cos(np.pi / 180.0 * sweep_25_vt)) ** 1.49
            )
        )
        partials[
            "data:weight:airframe:vertical_tail:mass", "data:geometry:vertical_tail:aspect_ratio"
        ] = k_factor_vt * (
            (
                0.357
                * area_vt**0.873
                * taper_vt**0.039
                * (mtow * sizing_factor_ultimate) ** 0.376
                * (0.0146 * has_t_tail + 0.073)
                * (0.010443 * rho_cruise * v_cruise_ktas**2) ** 0.122
            )
            / (
                np.cos(np.pi / 180.0 * sweep_25_vt) ** 2
                * (ar_vt / np.cos(np.pi / 180.0 * sweep_25_vt) ** 2) ** 0.643
                * ((100.0 * t_c_vt) / np.cos(np.pi / 180.0 * sweep_25_vt)) ** 0.49
            )
        )
        partials[
            "data:weight:airframe:vertical_tail:mass", "data:geometry:vertical_tail:taper_ratio"
        ] = k_factor_vt * (
            (
                0.039
                * area_vt**0.873
                * (mtow * sizing_factor_ultimate) ** 0.376
                * (ar_vt / np.cos(np.pi / 180.0 * sweep_25_vt) ** 2) ** 0.357
                * (0.0146 * has_t_tail + 0.073)
                * (0.010443 * rho_cruise * v_cruise_ktas**2) ** 0.122
            )
            / (taper_vt**0.9610 * ((100.0 * t_c_vt) / np.cos(np.pi / 180.0 * sweep_25_vt)) ** 0.49)
        )
        partials[
            "data:weight:airframe:vertical_tail:mass", "data:weight:airframe:vertical_tail:k_factor"
        ] = a32


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

        tmp = (
            area_ht**0.5840
            * (mtow * sizing_factor_ultimate) ** 0.8130
            * (mac_ht / lp_ht) ** 0.2800
            * (span_ht / (root_chord_ht * t_c_ht)) ** 0.0330
        ) ** 0.0850

        partials[
            "data:weight:airframe:horizontal_tail:mass",
            "data:mission:sizing:cs23:sizing_factor:ultimate_aircraft",
        ] = k_factor_ht * (
            (
                0.0025292
                * area_ht**0.5840
                * mtow
                * (mac_ht / lp_ht) ** 0.2800
                * (span_ht / (root_chord_ht * t_c_ht)) ** 0.0330
            )
            / ((mtow * sizing_factor_ultimate) ** 0.1870 * tmp)
        )
        partials["data:weight:airframe:horizontal_tail:mass", "data:weight:aircraft:MTOW"] = (
            k_factor_ht
            * (
                (
                    0.0025292
                    * area_ht**0.5840
                    * sizing_factor_ultimate
                    * (mac_ht / lp_ht) ** 0.2800
                    * (span_ht / (root_chord_ht * t_c_ht)) ** 0.0330
                )
                / ((mtow * sizing_factor_ultimate) ** 0.1870 * tmp)
            )
        )
        partials[
            "data:weight:airframe:horizontal_tail:mass", "data:geometry:horizontal_tail:area"
        ] = k_factor_ht * (
            (
                0.0018168
                * (mtow * sizing_factor_ultimate) ** 0.8130
                * (mac_ht / lp_ht) ** 0.2800
                * (span_ht / (root_chord_ht * t_c_ht)) ** 0.0330
            )
            / (area_ht**0.4160 * tmp)
        )
        partials[
            "data:weight:airframe:horizontal_tail:mass", "data:geometry:horizontal_tail:span"
        ] = k_factor_ht * (
            (
                1.0266e-4
                * area_ht**0.5840
                * (mtow * sizing_factor_ultimate) ** 0.8130
                * (mac_ht / lp_ht) ** 0.2800
            )
            / (root_chord_ht * t_c_ht * (span_ht / (root_chord_ht * t_c_ht)) ** 0.9670 * tmp)
        )
        partials[
            "data:weight:airframe:horizontal_tail:mass",
            "data:geometry:horizontal_tail:thickness_ratio",
        ] = k_factor_ht * (
            -(
                1.0266e-4
                * area_ht**0.5840
                * span_ht
                * (mtow * sizing_factor_ultimate) ** 0.8130
                * (mac_ht / lp_ht) ** 0.2800
            )
            / (root_chord_ht * t_c_ht**2 * (span_ht / (root_chord_ht * t_c_ht)) ** 0.9670 * tmp)
        )
        partials[
            "data:weight:airframe:horizontal_tail:mass", "data:geometry:horizontal_tail:root:chord"
        ] = k_factor_ht * (
            -(
                1.0266e-4
                * area_ht**0.5840
                * span_ht
                * (mtow * sizing_factor_ultimate) ** 0.8130
                * (mac_ht / lp_ht) ** 0.2800
            )
            / (root_chord_ht**2 * t_c_ht * (span_ht / (root_chord_ht * t_c_ht)) ** 0.9670 * tmp)
        )
        partials[
            "data:weight:airframe:horizontal_tail:mass", "data:geometry:horizontal_tail:MAC:length"
        ] = k_factor_ht * (
            (
                8.7108e-4
                * area_ht**0.5840
                * (mtow * sizing_factor_ultimate) ** 0.8130
                * (span_ht / (root_chord_ht * t_c_ht)) ** 0.0330
            )
            / (lp_ht * (mac_ht / lp_ht) ** 0.7200 * tmp)
        )
        partials[
            "data:weight:airframe:horizontal_tail:mass",
            "data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25",
        ] = k_factor_ht * (
            -(
                8.7108e-4
                * area_ht**0.5840
                * mac_ht
                * (mtow * sizing_factor_ultimate) ** 0.8130
                * (span_ht / (root_chord_ht * t_c_ht)) ** 0.0330
            )
            / (lp_ht**2 * (mac_ht / lp_ht) ** 0.7200 * tmp)
        )
        partials[
            "data:weight:airframe:horizontal_tail:mass",
            "data:weight:airframe:horizontal_tail:k_factor",
        ] = a31

        partials[
            "data:weight:airframe:vertical_tail:mass",
            "data:mission:sizing:cs23:sizing_factor:ultimate_aircraft",
        ] = k_factor_vt * (
            (
                0.06994
                * ar_vt**0.337
                * area_vt**1.089
                * mach_h**0.601
                * mtow
                * (has_t_tail + 1.0) ** 0.5
                * (rudder_chord_ratio + 1.0) ** 0.217
                * (taper_vt + 1.0) ** 0.363
                * (
                    (
                        ar_vt**0.337
                        * area_vt**1.089
                        * mach_h**0.601
                        * (has_t_tail + 1.0) ** 0.5
                        * (rudder_chord_ratio + 1.0) ** 0.217
                        * (taper_vt + 1.0) ** 0.363
                        * (mtow * sizing_factor_ultimate) ** 0.363
                    )
                    / (lp_vt**0.726 * np.cos(sweep_25_vt) ** 0.484)
                )
                ** 0.014
            )
            / (
                lp_vt**0.726
                * np.cos(sweep_25_vt) ** 0.484
                * (mtow * sizing_factor_ultimate) ** 0.637
            )
        )
        partials["data:weight:airframe:vertical_tail:mass", "data:weight:aircraft:MTOW"] = (
            k_factor_vt
            * (
                (
                    0.06994
                    * ar_vt**0.337
                    * area_vt**1.089
                    * mach_h**0.601
                    * sizing_factor_ultimate
                    * (has_t_tail + 1.0) ** 0.5
                    * (rudder_chord_ratio + 1.0) ** 0.217
                    * (taper_vt + 1.0) ** 0.363
                    * (
                        (
                            ar_vt**0.337
                            * area_vt**1.089
                            * mach_h**0.601
                            * (has_t_tail + 1.0) ** 0.5
                            * (rudder_chord_ratio + 1.0) ** 0.217
                            * (taper_vt + 1.0) ** 0.363
                            * (mtow * sizing_factor_ultimate) ** 0.363
                        )
                        / (lp_vt**0.726 * np.cos(sweep_25_vt) ** 0.484)
                    )
                    ** 0.014
                )
                / (
                    lp_vt**0.726
                    * np.cos(sweep_25_vt) ** 0.484
                    * (mtow * sizing_factor_ultimate) ** 0.637
                )
            )
        )
        partials["data:weight:airframe:vertical_tail:mass", "data:geometry:has_T_tail"] = (
            k_factor_vt
            * (
                (
                    0.09633
                    * ar_vt**0.337
                    * area_vt**1.089
                    * mach_h**0.601
                    * (rudder_chord_ratio + 1.0) ** 0.217
                    * (taper_vt + 1.0) ** 0.363
                    * (mtow * sizing_factor_ultimate) ** 0.363
                    * (
                        (
                            ar_vt**0.337
                            * area_vt**1.089
                            * mach_h**0.601
                            * (has_t_tail + 1.0) ** 0.5
                            * (rudder_chord_ratio + 1.0) ** 0.217
                            * (taper_vt + 1.0) ** 0.363
                            * (mtow * sizing_factor_ultimate) ** 0.363
                        )
                        / (lp_vt**0.726 * np.cos(sweep_25_vt) ** 0.484)
                    )
                    ** 0.014
                )
                / (lp_vt**0.726 * np.cos(sweep_25_vt) ** 0.484 * (has_t_tail + 1.0) ** 0.5)
            )
        )
        partials["data:weight:airframe:vertical_tail:mass", "data:geometry:vertical_tail:area"] = (
            k_factor_vt
            * (
                (
                    0.2098
                    * ar_vt**0.337
                    * area_vt**0.0890
                    * mach_h**0.601
                    * (has_t_tail + 1.0) ** 0.5
                    * (rudder_chord_ratio + 1.0) ** 0.217
                    * (taper_vt + 1.0) ** 0.363
                    * (mtow * sizing_factor_ultimate) ** 0.363
                    * (
                        (
                            ar_vt**0.337
                            * area_vt**1.089
                            * mach_h**0.601
                            * (has_t_tail + 1.0) ** 0.5
                            * (rudder_chord_ratio + 1.0) ** 0.217
                            * (taper_vt + 1.0) ** 0.363
                            * (mtow * sizing_factor_ultimate) ** 0.363
                        )
                        / (lp_vt**0.726 * np.cos(sweep_25_vt) ** 0.484)
                    )
                    ** 0.014
                )
                / (lp_vt**0.726 * np.cos(sweep_25_vt) ** 0.484)
            )
        )
        partials[
            "data:weight:airframe:vertical_tail:mass",
            "data:geometry:vertical_tail:MAC:at25percent:x:from_wingMAC25",
        ] = k_factor_vt * (
            -(
                0.1399
                * ar_vt**0.337
                * area_vt**1.089
                * mach_h**0.601
                * (has_t_tail + 1.0) ** 0.5
                * (rudder_chord_ratio + 1.0) ** 0.217
                * (taper_vt + 1.0) ** 0.363
                * (mtow * sizing_factor_ultimate) ** 0.363
                * (
                    (
                        ar_vt**0.337
                        * area_vt**1.089
                        * mach_h**0.601
                        * (has_t_tail + 1.0) ** 0.5
                        * (rudder_chord_ratio + 1.0) ** 0.217
                        * (taper_vt + 1.0) ** 0.363
                        * (mtow * sizing_factor_ultimate) ** 0.363
                    )
                    / (lp_vt**0.726 * np.cos(sweep_25_vt) ** 0.484)
                )
                ** 0.014
            )
            / (lp_vt**1.7260 * np.cos(sweep_25_vt) ** 0.484)
        )
        partials[
            "data:weight:airframe:vertical_tail:mass",
            "data:geometry:vertical_tail:rudder:chord_ratio",
        ] = k_factor_vt * (
            (
                0.04181
                * ar_vt**0.337
                * area_vt**1.089
                * mach_h**0.601
                * (has_t_tail + 1.0) ** 0.5
                * (taper_vt + 1.0) ** 0.363
                * (mtow * sizing_factor_ultimate) ** 0.363
                * (
                    (
                        ar_vt**0.337
                        * area_vt**1.089
                        * mach_h**0.601
                        * (has_t_tail + 1.0) ** 0.5
                        * (rudder_chord_ratio + 1.0) ** 0.217
                        * (taper_vt + 1.0) ** 0.363
                        * (mtow * sizing_factor_ultimate) ** 0.363
                    )
                    / (lp_vt**0.726 * np.cos(sweep_25_vt) ** 0.484)
                )
                ** 0.014
            )
            / (lp_vt**0.726 * np.cos(sweep_25_vt) ** 0.484 * (rudder_chord_ratio + 1.0) ** 0.7830)
        )
        partials[
            "data:weight:airframe:vertical_tail:mass", "data:geometry:vertical_tail:sweep_25"
        ] = k_factor_vt * (
            (
                0.09325
                * ar_vt**0.337
                * area_vt**1.089
                * mach_h**0.601
                * np.sin(sweep_25_vt)
                * (has_t_tail + 1.0) ** 0.5
                * (rudder_chord_ratio + 1.0) ** 0.217
                * (taper_vt + 1.0) ** 0.363
                * (mtow * sizing_factor_ultimate) ** 0.363
                * (
                    (
                        ar_vt**0.337
                        * area_vt**1.089
                        * mach_h**0.601
                        * (has_t_tail + 1.0) ** 0.5
                        * (rudder_chord_ratio + 1.0) ** 0.217
                        * (taper_vt + 1.0) ** 0.363
                        * (mtow * sizing_factor_ultimate) ** 0.363
                    )
                    / (lp_vt**0.726 * np.cos(sweep_25_vt) ** 0.484)
                )
                ** 0.014
            )
            / (lp_vt**0.726 * np.cos(sweep_25_vt) ** 1.4840)
        )
        partials[
            "data:weight:airframe:vertical_tail:mass", "data:geometry:vertical_tail:aspect_ratio"
        ] = k_factor_vt * (
            (
                0.06493
                * area_vt**1.089
                * mach_h**0.601
                * (has_t_tail + 1.0) ** 0.5
                * (rudder_chord_ratio + 1.0) ** 0.217
                * (taper_vt + 1.0) ** 0.363
                * (mtow * sizing_factor_ultimate) ** 0.363
                * (
                    (
                        ar_vt**0.337
                        * area_vt**1.089
                        * mach_h**0.601
                        * (has_t_tail + 1.0) ** 0.5
                        * (rudder_chord_ratio + 1.0) ** 0.217
                        * (taper_vt + 1.0) ** 0.363
                        * (mtow * sizing_factor_ultimate) ** 0.363
                    )
                    / (lp_vt**0.726 * np.cos(sweep_25_vt) ** 0.484)
                )
                ** 0.014
            )
            / (ar_vt**0.6630 * lp_vt**0.726 * np.cos(sweep_25_vt) ** 0.484)
        )
        partials[
            "data:weight:airframe:vertical_tail:mass", "data:geometry:vertical_tail:taper_ratio"
        ] = k_factor_vt * (
            (
                0.06994
                * ar_vt**0.337
                * area_vt**1.089
                * mach_h**0.601
                * (has_t_tail + 1.0) ** 0.5
                * (rudder_chord_ratio + 1.0) ** 0.217
                * (mtow * sizing_factor_ultimate) ** 0.363
                * (
                    (
                        ar_vt**0.337
                        * area_vt**1.089
                        * mach_h**0.601
                        * (has_t_tail + 1.0) ** 0.5
                        * (rudder_chord_ratio + 1.0) ** 0.217
                        * (taper_vt + 1.0) ** 0.363
                        * (mtow * sizing_factor_ultimate) ** 0.363
                    )
                    / (lp_vt**0.726 * np.cos(sweep_25_vt) ** 0.484)
                )
                ** 0.014
            )
            / (lp_vt**0.726 * np.cos(sweep_25_vt) ** 0.484 * (taper_vt + 1.0) ** 0.637)
        )
        d_a32_d_mach = (
            0.1158
            * ar_vt**0.337
            * area_vt**1.089
            * (has_t_tail + 1.0) ** 0.5
            * (rudder_chord_ratio + 1.0) ** 0.217
            * (taper_vt + 1.0) ** 0.363
            * (mtow * sizing_factor_ultimate) ** 0.363
            * (
                (
                    ar_vt**0.337
                    * area_vt**1.089
                    * mach_h**0.601
                    * (has_t_tail + 1.0) ** 0.5
                    * (rudder_chord_ratio + 1.0) ** 0.217
                    * (taper_vt + 1.0) ** 0.363
                    * (mtow * sizing_factor_ultimate) ** 0.363
                )
                / (lp_vt**0.726 * np.cos(sweep_25_vt) ** 0.484)
            )
            ** 0.014
        ) / (lp_vt**0.726 * mach_h ** (399 / 1000) * np.cos(sweep_25_vt) ** 0.484)
        d_mach_d_vh = 1 / atm0.speed_of_sound
        partials["data:weight:airframe:vertical_tail:mass", "data:TLAR:v_max_sl"] = k_factor_vt * (
            d_a32_d_mach * d_mach_d_vh
        )
        partials[
            "data:weight:airframe:vertical_tail:mass", "data:weight:airframe:vertical_tail:k_factor"
        ] = a32


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

        atm0 = Atmosphere(0)
        atm0.true_airspeed = v_h
        mach_h = atm0.mach

        a31 = area_ht * (3.81 * area_ht**0.2 * vd / (1000.0 * np.cos(sweep_25)) - 0.287)
        # Mass formula in lb

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
        ] = a31

        partials[
            "data:weight:airframe:vertical_tail:mass",
            "data:mission:sizing:cs23:sizing_factor:ultimate_aircraft",
        ] = k_factor_vt * (
            (
                0.06994
                * ar_vt**0.337
                * area_vt**1.089
                * mach_h**0.601
                * mtow
                * (has_t_tail + 1.0) ** 0.5
                * (rudder_chord_ratio + 1.0) ** 0.217
                * (taper_vt + 1.0) ** 0.363
                * (
                    (
                        ar_vt**0.337
                        * area_vt**1.089
                        * mach_h**0.601
                        * (has_t_tail + 1.0) ** 0.5
                        * (rudder_chord_ratio + 1.0) ** 0.217
                        * (taper_vt + 1.0) ** 0.363
                        * (mtow * sizing_factor_ultimate) ** 0.363
                    )
                    / (lp_vt**0.726 * np.cos(sweep_25_vt) ** 0.484)
                )
                ** 0.014
            )
            / (
                lp_vt**0.726
                * np.cos(sweep_25_vt) ** 0.484
                * (mtow * sizing_factor_ultimate) ** 0.637
            )
        )
        partials["data:weight:airframe:vertical_tail:mass", "data:weight:aircraft:MTOW"] = (
            k_factor_vt
            * (
                (
                    0.06994
                    * ar_vt**0.337
                    * area_vt**1.089
                    * mach_h**0.601
                    * sizing_factor_ultimate
                    * (has_t_tail + 1.0) ** 0.5
                    * (rudder_chord_ratio + 1.0) ** 0.217
                    * (taper_vt + 1.0) ** 0.363
                    * (
                        (
                            ar_vt**0.337
                            * area_vt**1.089
                            * mach_h**0.601
                            * (has_t_tail + 1.0) ** 0.5
                            * (rudder_chord_ratio + 1.0) ** 0.217
                            * (taper_vt + 1.0) ** 0.363
                            * (mtow * sizing_factor_ultimate) ** 0.363
                        )
                        / (lp_vt**0.726 * np.cos(sweep_25_vt) ** 0.484)
                    )
                    ** 0.014
                )
                / (
                    lp_vt**0.726
                    * np.cos(sweep_25_vt) ** 0.484
                    * (mtow * sizing_factor_ultimate) ** 0.637
                )
            )
        )
        partials["data:weight:airframe:vertical_tail:mass", "data:geometry:has_T_tail"] = (
            k_factor_vt
            * (
                (
                    0.09633
                    * ar_vt**0.337
                    * area_vt**1.089
                    * mach_h**0.601
                    * (rudder_chord_ratio + 1.0) ** 0.217
                    * (taper_vt + 1.0) ** 0.363
                    * (mtow * sizing_factor_ultimate) ** 0.363
                    * (
                        (
                            ar_vt**0.337
                            * area_vt**1.089
                            * mach_h**0.601
                            * (has_t_tail + 1.0) ** 0.5
                            * (rudder_chord_ratio + 1.0) ** 0.217
                            * (taper_vt + 1.0) ** 0.363
                            * (mtow * sizing_factor_ultimate) ** 0.363
                        )
                        / (lp_vt**0.726 * np.cos(sweep_25_vt) ** 0.484)
                    )
                    ** 0.014
                )
                / (lp_vt**0.726 * np.cos(sweep_25_vt) ** 0.484 * (has_t_tail + 1.0) ** 0.5)
            )
        )
        partials["data:weight:airframe:vertical_tail:mass", "data:geometry:vertical_tail:area"] = (
            k_factor_vt
            * (
                (
                    0.2098
                    * ar_vt**0.337
                    * area_vt**0.089
                    * mach_h**0.601
                    * (has_t_tail + 1.0) ** 0.5
                    * (rudder_chord_ratio + 1.0) ** 0.217
                    * (taper_vt + 1.0) ** 0.363
                    * (mtow * sizing_factor_ultimate) ** 0.363
                    * (
                        (
                            ar_vt**0.337
                            * area_vt**1.089
                            * mach_h**0.601
                            * (has_t_tail + 1.0) ** 0.5
                            * (rudder_chord_ratio + 1.0) ** 0.217
                            * (taper_vt + 1.0) ** 0.363
                            * (mtow * sizing_factor_ultimate) ** 0.363
                        )
                        / (lp_vt**0.726 * np.cos(sweep_25_vt) ** 0.484)
                    )
                    ** 0.014
                )
                / (lp_vt**0.726 * np.cos(sweep_25_vt) ** 0.484)
            )
        )
        partials[
            "data:weight:airframe:vertical_tail:mass",
            "data:geometry:vertical_tail:MAC:at25percent:x:from_wingMAC25",
        ] = k_factor_vt * (
            -(
                0.1399
                * ar_vt**0.337
                * area_vt**1.089
                * mach_h**0.601
                * (has_t_tail + 1.0) ** 0.5
                * (rudder_chord_ratio + 1.0) ** 0.217
                * (taper_vt + 1.0) ** 0.363
                * (mtow * sizing_factor_ultimate) ** 0.363
                * (
                    (
                        ar_vt**0.337
                        * area_vt**1.089
                        * mach_h**0.601
                        * (has_t_tail + 1.0) ** 0.5
                        * (rudder_chord_ratio + 1.0) ** 0.217
                        * (taper_vt + 1.0) ** 0.363
                        * (mtow * sizing_factor_ultimate) ** 0.363
                    )
                    / (lp_vt**0.726 * np.cos(sweep_25_vt) ** 0.484)
                )
                ** 0.014
            )
            / (lp_vt**1.7260 * np.cos(sweep_25_vt) ** 0.484)
        )
        partials[
            "data:weight:airframe:vertical_tail:mass",
            "data:geometry:vertical_tail:rudder:chord_ratio",
        ] = k_factor_vt * (
            (
                0.04181
                * ar_vt**0.337
                * area_vt**1.089
                * mach_h**0.601
                * (has_t_tail + 1.0) ** 0.5
                * (taper_vt + 1.0) ** 0.363
                * (mtow * sizing_factor_ultimate) ** 0.363
                * (
                    (
                        ar_vt**0.337
                        * area_vt**1.089
                        * mach_h**0.601
                        * (has_t_tail + 1.0) ** 0.5
                        * (rudder_chord_ratio + 1.0) ** 0.217
                        * (taper_vt + 1.0) ** 0.363
                        * (mtow * sizing_factor_ultimate) ** 0.363
                    )
                    / (lp_vt**0.726 * np.cos(sweep_25_vt) ** 0.484)
                )
                ** 0.014
            )
            / (lp_vt**0.726 * np.cos(sweep_25_vt) ** 0.484 * (rudder_chord_ratio + 1.0) ** 0.7830)
        )
        partials[
            "data:weight:airframe:vertical_tail:mass", "data:geometry:vertical_tail:sweep_25"
        ] = k_factor_vt * (
            (
                0.09325
                * ar_vt**0.337
                * area_vt**1.089
                * mach_h**0.601
                * np.sin(sweep_25_vt)
                * (has_t_tail + 1.0) ** 0.5
                * (rudder_chord_ratio + 1.0) ** 0.217
                * (taper_vt + 1.0) ** 0.363
                * (mtow * sizing_factor_ultimate) ** 0.363
                * (
                    (
                        ar_vt**0.337
                        * area_vt**1.089
                        * mach_h**0.601
                        * (has_t_tail + 1.0) ** 0.5
                        * (rudder_chord_ratio + 1.0) ** 0.217
                        * (taper_vt + 1.0) ** 0.363
                        * (mtow * sizing_factor_ultimate) ** 0.363
                    )
                    / (lp_vt**0.726 * np.cos(sweep_25_vt) ** 0.484)
                )
                ** 0.014
            )
            / (lp_vt**0.726 * np.cos(sweep_25_vt) ** 1.4840)
        )
        partials[
            "data:weight:airframe:vertical_tail:mass", "data:geometry:vertical_tail:aspect_ratio"
        ] = k_factor_vt * (
            (
                0.06493
                * area_vt**1.089
                * mach_h**0.601
                * (has_t_tail + 1.0) ** 0.5
                * (rudder_chord_ratio + 1.0) ** 0.217
                * (taper_vt + 1.0) ** 0.363
                * (mtow * sizing_factor_ultimate) ** 0.363
                * (
                    (
                        ar_vt**0.337
                        * area_vt**1.089
                        * mach_h**0.601
                        * (has_t_tail + 1.0) ** 0.5
                        * (rudder_chord_ratio + 1.0) ** 0.217
                        * (taper_vt + 1.0) ** 0.363
                        * (mtow * sizing_factor_ultimate) ** 0.363
                    )
                    / (lp_vt**0.726 * np.cos(sweep_25_vt) ** 0.484)
                )
                ** 0.014
            )
            / (ar_vt**0.6630 * lp_vt**0.726 * np.cos(sweep_25_vt) ** 0.484)
        )
        partials[
            "data:weight:airframe:vertical_tail:mass", "data:geometry:vertical_tail:taper_ratio"
        ] = k_factor_vt * (
            (
                0.06994
                * ar_vt**0.337
                * area_vt**1.089
                * mach_h**0.601
                * (has_t_tail + 1.0) ** 0.5
                * (rudder_chord_ratio + 1.0) ** 0.217
                * (mtow * sizing_factor_ultimate) ** 0.363
                * (
                    (
                        ar_vt**0.337
                        * area_vt**1.089
                        * mach_h**0.601
                        * (has_t_tail + 1.0) ** 0.5
                        * (rudder_chord_ratio + 1.0) ** 0.217
                        * (taper_vt + 1.0) ** 0.363
                        * (mtow * sizing_factor_ultimate) ** 0.363
                    )
                    / (lp_vt**0.726 * np.cos(sweep_25_vt) ** 0.484)
                )
                ** 0.014
            )
            / (lp_vt**0.726 * np.cos(sweep_25_vt) ** 0.484 * (taper_vt + 1.0) ** 0.637)
        )
        d_a32_d_mach = (
            0.1158
            * ar_vt**0.337
            * area_vt**1.089
            * (has_t_tail + 1.0) ** 0.5
            * (rudder_chord_ratio + 1.0) ** 0.217
            * (taper_vt + 1.0) ** 0.363
            * (mtow * sizing_factor_ultimate) ** 0.363
            * (
                (
                    ar_vt**0.337
                    * area_vt**1.089
                    * mach_h**0.601
                    * (has_t_tail + 1.0) ** 0.5
                    * (rudder_chord_ratio + 1.0) ** 0.217
                    * (taper_vt + 1.0) ** 0.363
                    * (mtow * sizing_factor_ultimate) ** 0.363
                )
                / (lp_vt**0.726 * np.cos(sweep_25_vt) ** 0.484)
            )
            ** 0.014
        ) / (lp_vt**0.726 * mach_h ** (399 / 1000) * np.cos(sweep_25_vt) ** 0.484)
        d_mach_d_vh = 1 / atm0.speed_of_sound
        partials["data:weight:airframe:vertical_tail:mass", "data:TLAR:v_max_sl"] = k_factor_vt * (
            d_a32_d_mach * d_mach_d_vh
        )
        partials[
            "data:weight:airframe:vertical_tail:mass", "data:weight:airframe:vertical_tail:k_factor"
        ] = a32
