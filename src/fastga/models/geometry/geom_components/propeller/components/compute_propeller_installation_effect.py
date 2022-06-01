"""Estimation of propeller effective advance ratio."""
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

import warnings

import numpy as np
import openmdao.api as om

import fastoad.api as oad

from ..constants import SUBMODEL_PROPELLER_INSTALLATION


@oad.RegisterSubmodel(
    SUBMODEL_PROPELLER_INSTALLATION, "fastga.submodel.geometry.propeller.installation_effect.legacy"
)
class ComputePropellerInstallationEffect(om.ExplicitComponent):
    """Propeller effective advance ratio computation based on the blockage surface behind it."""

    def setup(self):

        self.add_input("data:geometry:propulsion:engine:layout", val=np.nan)
        self.add_input("data:geometry:propeller:diameter", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:master_cross_section", val=np.nan, units="m**2")
        self.add_input(
            "data:geometry:propulsion:nacelle:master_cross_section", val=np.nan, units="m**2"
        )

        self.add_output(
            "data:aerodynamics:propeller:installation_effect:effective_advance_ratio",
            val=1.0,
            desc="Value to multiply the flight advance ration with to obtain the effective "
            "advance ratio due to the presence of cowling (fuselage or nacelle) behind the "
            "propeller",
        )

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        engine_layout = inputs["data:geometry:propulsion:engine:layout"]

        if engine_layout == 3.0:
            cowling_master_cross_section = inputs["data:geometry:fuselage:master_cross_section"]
        elif engine_layout == 1.0 or engine_layout == 2.0:
            cowling_master_cross_section = inputs[
                "data:geometry:propulsion:nacelle:master_cross_section"
            ]
        else:
            cowling_master_cross_section = inputs["data:geometry:fuselage:master_cross_section"]
            warnings.warn(
                "Propulsion layout {} not implemented in model, replaced by layout 3!".format(
                    engine_layout
                )
            )

        disk_diameter = inputs["data:geometry:propeller:diameter"]

        disk_surface = np.pi * (disk_diameter / 2.0) ** 2.0

        effective_advance_ratio = 1.0 - 0.254 * cowling_master_cross_section / disk_surface

        outputs[
            "data:aerodynamics:propeller:installation_effect:effective_advance_ratio"
        ] = effective_advance_ratio
