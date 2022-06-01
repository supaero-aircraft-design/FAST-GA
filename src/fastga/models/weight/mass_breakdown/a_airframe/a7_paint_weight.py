"""Estimation of paint weight."""
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

from .constants import SUBMODEL_PAINT_MASS

oad.RegisterSubmodel.active_models[
    SUBMODEL_PAINT_MASS
] = "fastga.submodel.weight.mass.airframe.paint.no_paint"


@oad.RegisterSubmodel(SUBMODEL_PAINT_MASS, "fastga.submodel.weight.mass.airframe.paint.no_paint")
class ComputeNoPaintWeight(om.IndepVarComp):
    """
    Paint weight estimation.

    Component that returns 0 kg of paint as for most aircraft this weight is negligible but kept
    as default submodels so that it doesn't change previous computation.
    """

    def setup(self):
        # In theory this component does not need an input as it will always return 0 but to
        # properly define the component we need to declare it even if it is not used. This
        # "ghost" input was chosen as it will be used for the actual component
        self.add_input("data:geometry:aircraft:wet_area", val=np.nan, units="m**2")

        self.add_output("data:weight:airframe:paint:mass", units="lb")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        outputs["data:weight:airframe:paint:mass"] = 0.0 * inputs["data:geometry:aircraft:wet_area"]


@oad.RegisterSubmodel(SUBMODEL_PAINT_MASS, "fastga.submodel.weight.mass.airframe.paint.by_wet_area")
class ComputePaintWeight(om.IndepVarComp):
    """
    Paint weight estimation.

    Component that returns the paint weight by using a value of surface density for the paint.
    """

    def setup(self):
        self.add_input("data:geometry:aircraft:wet_area", val=np.nan, units="m**2")
        self.add_input("settings:weight:airframe:paint:surface_density", val=0.33, units="kg/m**2")

        self.add_output("data:weight:airframe:paint:mass", units="kg")

        self.declare_partials(of="data:weight:airframe:paint:mass", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        outputs["data:weight:airframe:paint:mass"] = (
            inputs["data:geometry:aircraft:wet_area"]
            * inputs["settings:weight:airframe:paint:surface_density"]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        partials["data:weight:airframe:paint:mass", "data:geometry:aircraft:wet_area"] = inputs[
            "settings:weight:airframe:paint:surface_density"
        ]
        partials[
            "data:weight:airframe:paint:mass", "settings:weight:airframe:paint:surface_density"
        ] = inputs["data:geometry:aircraft:wet_area"]
