"""
Python module for paint weight calculation, part of the airframe mass computation.
"""
#  This file is part of FAST-OAD_CS23 : A framework for rapid Overall Aircraft Design
#  Copyright (C) 2025  ONERA & ISAE-SUPAERO
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

from .constants import (
    SERVICE_PAINT_MASS,
    SUBMODEL_PAINT_MASS_NO_PAINT,
    SUBMODEL_PAINT_MASS_BY_WET_AREA,
)

oad.RegisterSubmodel.active_models[SERVICE_PAINT_MASS] = SUBMODEL_PAINT_MASS_NO_PAINT


@oad.RegisterSubmodel(SERVICE_PAINT_MASS, SUBMODEL_PAINT_MASS_NO_PAINT)
class ComputeNoPaintWeight(om.ExplicitComponent):
    """
    Paint weight estimation.

    Component that returns 0 kg of paint as for most aircraft this weight is negligible but kept
    as default submodels so that it doesn't change previous computation.
    """

    # pylint: disable=missing-function-docstring
    # Overriding OpenMDAO setup
    def setup(self):
        # In theory this component does not need an input as it will always return 0 but to
        # properly define the component we need to declare it even if it is not used. This
        # "ghost" input was chosen as it will be used for the actual component
        self.add_input("data:geometry:aircraft:wet_area", val=np.nan, units="m**2")

        self.add_output("data:weight:airframe:paint:mass", units="lb")

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute, not all arguments are used
    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs["data:weight:airframe:paint:mass"] = 0.0 * inputs["data:geometry:aircraft:wet_area"]


@oad.RegisterSubmodel(SERVICE_PAINT_MASS, SUBMODEL_PAINT_MASS_BY_WET_AREA)
class ComputePaintWeight(om.ExplicitComponent):
    """
    Paint weight estimation.

    Component that returns the paint weight by using a value of surface density for the paint.
    """

    # pylint: disable=missing-function-docstring
    # Overriding OpenMDAO setup
    def setup(self):
        self.add_input("data:geometry:aircraft:wet_area", val=np.nan, units="m**2")
        self.add_input("settings:weight:airframe:paint:surface_density", val=0.33, units="kg/m**2")

        self.add_output("data:weight:airframe:paint:mass", units="kg")

        self.declare_partials(of="data:weight:airframe:paint:mass", wrt="*", method="exact")

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute, not all arguments are used
    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs["data:weight:airframe:paint:mass"] = (
            inputs["data:geometry:aircraft:wet_area"]
            * inputs["settings:weight:airframe:paint:surface_density"]
        )

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute_partials, not all arguments are used
    def compute_partials(self, inputs, partials, discrete_inputs=None):
        partials["data:weight:airframe:paint:mass", "data:geometry:aircraft:wet_area"] = inputs[
            "settings:weight:airframe:paint:surface_density"
        ]
        partials[
            "data:weight:airframe:paint:mass", "settings:weight:airframe:paint:surface_density"
        ] = inputs["data:geometry:aircraft:wet_area"]
