"""
Python module for total aircraft wet area calculation, part of the geometry component.
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

import numpy as np
import openmdao.api as om
import fastoad.api as oad

from ..constants import SERVICE_AIRCRAFT_WET_AREA, SUBMODEL_AIRCRAFT_WET_AREA_LEGACY


@oad.RegisterSubmodel(SERVICE_AIRCRAFT_WET_AREA, SUBMODEL_AIRCRAFT_WET_AREA_LEGACY)
class ComputeTotalArea(om.ExplicitComponent):
    """Total aircraft wet area estimation, obtained from :cite:`supaero:2014`."""

    # pylint: disable=missing-function-docstring
    # Overriding OpenMDAO setup
    def setup(self):
        self.add_input("data:geometry:wing:wet_area", val=np.nan, units="m**2")
        self.add_input("data:geometry:fuselage:wet_area", val=np.nan, units="m**2")
        self.add_input("data:geometry:horizontal_tail:wet_area", val=np.nan, units="m**2")
        self.add_input("data:geometry:vertical_tail:wet_area", val=np.nan, units="m**2")
        self.add_input("data:geometry:propulsion:nacelle:wet_area", val=np.nan, units="m**2")
        self.add_input("data:geometry:propulsion:engine:count", val=np.nan)

        self.add_output("data:geometry:aircraft:wet_area", units="m**2")

        self.declare_partials(
            "*",
            [
                "data:geometry:wing:wet_area",
                "data:geometry:fuselage:wet_area",
                "data:geometry:horizontal_tail:wet_area",
                "data:geometry:vertical_tail:wet_area",
            ],
            val=1.0,
        )
        self.declare_partials(
            "*",
            ["data:geometry:propulsion:nacelle:wet_area", "data:geometry:propulsion:engine:count"],
            method="exact",
        )

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute, not all arguments are used
    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        wet_area_wing = inputs["data:geometry:wing:wet_area"]
        wet_area_fus = inputs["data:geometry:fuselage:wet_area"]
        wet_area_ht = inputs["data:geometry:horizontal_tail:wet_area"]
        wet_area_vt = inputs["data:geometry:vertical_tail:wet_area"]
        wet_area_nac = inputs["data:geometry:propulsion:nacelle:wet_area"]
        nacelle_nb = inputs["data:geometry:propulsion:engine:count"]

        wet_area_total = (
            wet_area_wing + wet_area_fus + wet_area_ht + wet_area_vt + nacelle_nb * wet_area_nac
        )

        outputs["data:geometry:aircraft:wet_area"] = wet_area_total

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute_partials, not all arguments are used
    def compute_partials(self, inputs, partials, discrete_inputs=None):
        wet_area_nac = inputs["data:geometry:propulsion:nacelle:wet_area"]
        nacelle_nb = inputs["data:geometry:propulsion:engine:count"]
        partials["data:geometry:aircraft:wet_area", "data:geometry:propulsion:nacelle:wet_area"] = (
            nacelle_nb
        )
        partials["data:geometry:aircraft:wet_area", "data:geometry:propulsion:engine:count"] = (
            wet_area_nac
        )
