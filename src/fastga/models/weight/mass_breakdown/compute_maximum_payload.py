"""
Maximum payload mass computation.
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

from .constants import SUBMODEL_MAX_PAYLOAD_MASS


@oad.RegisterSubmodel(SUBMODEL_MAX_PAYLOAD_MASS, "fastga.submodel.weight.mass.payload.max.legacy")
class ComputeMaxPayload(om.ExplicitComponent):
    """Computes maximum payload from NPAX with addition of 2 pilots"""

    def setup(self):

        self.add_input("data:geometry:cabin:seats:passenger:NPAX_max", val=np.nan)
        self.add_input("data:geometry:cabin:luggage:mass_max", val=np.nan, units="kg")
        self.add_input(
            "settings:weight:aircraft:payload:max_mass_per_passenger",
            val=90.0,
            units="kg",
            desc="Maximum value of mass per passenger",
        )

        self.add_output("data:weight:aircraft:max_payload", units="kg")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        npax_max = inputs["data:geometry:cabin:seats:passenger:NPAX_max"] + 2.0
        max_mass_per_pax = inputs["settings:weight:aircraft:payload:max_mass_per_passenger"]
        luggage_mass_max = inputs["data:geometry:cabin:luggage:mass_max"]

        outputs["data:weight:aircraft:max_payload"] = npax_max * max_mass_per_pax + luggage_mass_max
