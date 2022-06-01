"""
Payload mass computation.
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

from openmdao import api as om

import fastoad.api as oad

from .constants import SUBMODEL_PAYLOAD_MASS


@oad.RegisterSubmodel(SUBMODEL_PAYLOAD_MASS, "fastga.submodel.weight.mass.payload.legacy")
class ComputePayload(om.ExplicitComponent):
    """Computes payload from NPAX."""

    def setup(self):
        self.add_input("data:TLAR:NPAX_design", val=np.nan)
        self.add_input("data:geometry:cabin:seats:passenger:NPAX_max", val=np.nan)
        self.add_input("data:geometry:cabin:luggage:mass_max", val=np.nan, units="kg")
        self.add_input("data:TLAR:luggage_mass_design", val=np.nan, units="kg")
        self.add_input(
            "settings:weight:aircraft:payload:design_mass_per_passenger",
            val=80.0,
            units="kg",
            desc="Design value of mass per passenger",
        )
        self.add_input(
            "settings:weight:aircraft:payload:max_mass_per_passenger",
            val=90.0,
            units="kg",
            desc="Maximum value of mass per passenger",
        )

        self.add_output("data:weight:aircraft:payload", units="kg")
        self.add_output("data:weight:aircraft:max_payload", units="kg")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        npax_design = inputs["data:TLAR:NPAX_design"] + 2.0  # addition of 2 pilots
        npax_max = (
            inputs["data:geometry:cabin:seats:passenger:NPAX_max"] + 2.0
        )  # addition of 2 pilots
        mass_per_pax = inputs["settings:weight:aircraft:payload:design_mass_per_passenger"]
        max_mass_per_pax = inputs["settings:weight:aircraft:payload:max_mass_per_passenger"]
        luggage_mass_design = inputs["data:TLAR:luggage_mass_design"]
        luggage_mass_max = inputs["data:geometry:cabin:luggage:mass_max"]

        outputs["data:weight:aircraft:payload"] = npax_design * mass_per_pax + luggage_mass_design
        outputs["data:weight:aircraft:max_payload"] = npax_max * max_mass_per_pax + luggage_mass_max
