"""
Estimation of landing gear weight.
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

import fastoad.api as oad

from .constants import SUBMODEL_LANDING_GEAR_MASS

oad.RegisterSubmodel.active_models[
    SUBMODEL_LANDING_GEAR_MASS
] = "fastga.submodel.weight.mass.airframe.landing_gear.legacy"


@oad.RegisterSubmodel(
    SUBMODEL_LANDING_GEAR_MASS, "fastga.submodel.weight.mass.airframe.landing_gear.legacy"
)
class ComputeLandingGearWeight(om.ExplicitComponent):
    """
    Weight estimation for landing gears

    Based on a statistical analysis. See :cite:`wells:2017` for the formula and
    :cite:`raymer:2012` for the weight reduction factor for non retractable landing gears
    """

    def setup(self):

        self.add_input("data:weight:aircraft:MLW", val=np.nan, units="lb")
        self.add_input("data:weight:aircraft:MTOW", val=np.nan, units="lb")
        self.add_input("data:geometry:landing_gear:height", val=np.nan, units="inch")
        self.add_input("data:geometry:landing_gear:type", val=np.nan)

        self.add_output("data:weight:airframe:landing_gear:main:mass", units="lb")
        self.add_output("data:weight:airframe:landing_gear:front:mass", units="lb")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        mlw = inputs["data:weight:aircraft:MLW"]
        mtow = inputs["data:weight:aircraft:MTOW"]
        lg_height = inputs["data:geometry:landing_gear:height"]
        is_retractable = inputs["data:geometry:landing_gear:type"]

        carrier_based = 0.0
        aircraft_type = 0.0  # One for fighter/attack aircraft

        # To prevent using obstruse data we put this failsafe here
        # TODO : Find a better way to do this
        if mlw < mtow / 2.0:
            mlw = mtow

        mlg_weight = (0.0117 - aircraft_type * 0.0012) * mlw ** 0.95 * lg_height ** 0.43
        nlg_weight = (
            (0.048 - aircraft_type * 0.008)
            * mlw ** 0.67
            * lg_height ** 0.43
            * (1.0 + 0.8 * carrier_based)
        )

        if not is_retractable:
            weight_reduction = 1.4 * mtow / 100.0
            weight_reduction_factor = (mlg_weight + nlg_weight - weight_reduction) / (
                mlg_weight + nlg_weight
            )
        else:
            weight_reduction_factor = 1.0

        outputs["data:weight:airframe:landing_gear:main:mass"] = (
            mlg_weight * weight_reduction_factor
        )
        outputs["data:weight:airframe:landing_gear:front:mass"] = (
            nlg_weight * weight_reduction_factor
        )
