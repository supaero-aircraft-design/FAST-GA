"""Estimation of landing gear center(s) of gravity."""

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
from openmdao.core.explicitcomponent import ExplicitComponent
from openmdao.core.group import Group

import fastoad.api as oad

from ..constants import SUBMODEL_LANDING_GEAR_CG

oad.RegisterSubmodel.active_models[
    SUBMODEL_LANDING_GEAR_CG
] = "fastga.submodel.weight.cg.airframe.landing_gear.legacy"


@oad.RegisterSubmodel(
    SUBMODEL_LANDING_GEAR_CG, "fastga.submodel.weight.cg.airframe.landing_gear.legacy"
)
class ComputeLandingGearCG(Group):
    # TODO: Document equations. Cite sources
    """
    Landing gear center of gravity estimation based on the ratio of weight supported by each
    gear.
    """

    def setup(self):

        self.add_subsystem("comp_nlg_cg", ComputeFrontCGX(), promotes=["*"])
        self.add_subsystem("comp_mlg_cg", ComputeMainCGX(), promotes=["*"])


class ComputeFrontCGX(ExplicitComponent):
    """NLG gravity center"""

    def setup(self):

        self.add_input("data:geometry:fuselage:front_length", val=np.nan, units="m")
        self.add_input(
            "settings:weight:airframe:landing_gear:front:front_fuselage_ratio",
            val=0.75,
        )

        self.add_output("data:weight:airframe:landing_gear:front:CG:x", units="m")

        self.declare_partials(
            of="data:weight:airframe:landing_gear:front:CG:x",
            wrt="data:geometry:fuselage:front_length",
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        lav = inputs["data:geometry:fuselage:front_length"]
        front_lg_fuselage = inputs[
            "settings:weight:airframe:landing_gear:front:front_fuselage_ratio"
        ]

        x_cg_a52 = lav * front_lg_fuselage

        outputs["data:weight:airframe:landing_gear:front:CG:x"] = x_cg_a52

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        front_lg_fuselage = inputs[
            "settings:weight:airframe:landing_gear:front:front_fuselage_ratio"
        ]

        partials[
            "data:weight:airframe:landing_gear:front:CG:x", "data:geometry:fuselage:front_length"
        ] = front_lg_fuselage


class ComputeMainCGX(ExplicitComponent):
    """MLG gravity center"""

    def setup(self):

        self.add_input("data:geometry:wing:MAC:length", val=np.nan, units="m")
        self.add_input("data:geometry:wing:MAC:at25percent:x", val=np.nan, units="m")
        self.add_input("data:weight:aircraft:CG:aft:MAC_position", val=np.nan)
        self.add_input(
            "settings:weight:airframe:landing_gear:front:weight_ratio",
            val=0.3,
        )
        self.add_input("data:weight:airframe:landing_gear:front:CG:x", val=np.nan, units="m")

        self.add_output("data:weight:airframe:landing_gear:main:CG:x", units="m")

        self.declare_partials(
            of="data:weight:airframe:landing_gear:main:CG:x",
            wrt="*",
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        l0_wing = inputs["data:geometry:wing:MAC:length"]
        fa_length = inputs["data:geometry:wing:MAC:at25percent:x"]
        cg_ratio = inputs["data:weight:aircraft:CG:aft:MAC_position"]
        front_lg_weight_ratio = inputs["settings:weight:airframe:landing_gear:front:weight_ratio"]
        x_cg_a52 = inputs["data:weight:airframe:landing_gear:front:CG:x"]

        x_cg_aft = fa_length + (cg_ratio - 0.25) * l0_wing
        x_cg_a51 = (x_cg_aft - front_lg_weight_ratio * x_cg_a52) / (1.0 - front_lg_weight_ratio)

        outputs["data:weight:airframe:landing_gear:main:CG:x"] = x_cg_a51

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        l0_wing = inputs["data:geometry:wing:MAC:length"]
        fa_length = inputs["data:geometry:wing:MAC:at25percent:x"]
        cg_ratio = inputs["data:weight:aircraft:CG:aft:MAC_position"]
        front_lg_weight_ratio = inputs["settings:weight:airframe:landing_gear:front:weight_ratio"]
        x_cg_a52 = inputs["data:weight:airframe:landing_gear:front:CG:x"]

        # Aft most CG position
        x_cg_aft = fa_length + (cg_ratio - 0.25) * l0_wing

        partials["data:weight:airframe:landing_gear:main:CG:x", "data:geometry:wing:MAC:length"] = (
            cg_ratio - 0.25
        ) / (1.0 - front_lg_weight_ratio)
        partials[
            "data:weight:airframe:landing_gear:main:CG:x", "data:geometry:wing:MAC:at25percent:x"
        ] = 1 / (1.0 - front_lg_weight_ratio)
        partials[
            "data:weight:airframe:landing_gear:main:CG:x",
            "data:weight:aircraft:CG:aft:MAC_position",
        ] = l0_wing / (1.0 - front_lg_weight_ratio)
        partials[
            "data:weight:airframe:landing_gear:main:CG:x",
            "settings:weight:airframe:landing_gear:front:weight_ratio",
        ] = (x_cg_aft - x_cg_a52) / (1.0 - front_lg_weight_ratio) ** 2.0
        partials[
            "data:weight:airframe:landing_gear:main:CG:x",
            "data:weight:airframe:landing_gear:front:CG:x",
        ] = -front_lg_weight_ratio / (1.0 - front_lg_weight_ratio)
