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

from .constants import SUBMODEL_TAIL_MASS


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
                "data:mission:sizing:cs23:sizing_factor:ultimate_aircraft",
                "data:weight:airframe:horizontal_tail:k_factor",
                "data:mission:sizing:cs23:characteristic_speed:vd",
                "data:geometry:horizontal_tail:sweep_25",
            ],
            method="fd",
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
            method="fd",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        sizing_factor_ultimate = inputs["data:mission:sizing:cs23:sizing_factor:ultimate_aircraft"]
        mtow = inputs["data:weight:aircraft:MTOW"]

        area_ht = inputs["data:geometry:horizontal_tail:area"]
        vd = inputs["data:mission:sizing:cs23:characteristic_speed:vd"]
        sweep_25 = inputs["data:geometry:horizontal_tail:sweep_25"]

        a31 = area_ht * (3.81 * area_ht ** 0.2 * vd / (1000.0 * np.cos(sweep_25)) - 0.287)
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
