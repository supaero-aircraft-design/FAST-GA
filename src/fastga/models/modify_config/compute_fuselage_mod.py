"""
    Estimation of total aircraft wet area
"""

#  This file is part of FAST : A framework for rapid Overall Aircraft Design
#  Copyright (C) 2020  ONERA & ISAE-SUPAERO
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


class ComputeFuselageMod(om.ExplicitComponent):

    """
    yo
    """

    def initialize(self):
        self.options.declare("fuselage_mod", types=list, default=[0, 0, 0, 0, 0])

    def setup(self):
        self.add_input("data:geometry:cabin:seats:passenger:NPAX_max", val=np.nan)
        self.add_input("data:TLAR:NPAX_design", val=np.nan)
        self.add_input("data:geometry:cabin:seats:passenger:count_by_row", val=np.nan)
        self.add_input("data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25", val=np.nan, units="m")
        self.add_input("data:geometry:cabin:seats:passenger:length", val=np.nan, units="m")
        self.add_input("data:geometry:wing:MAC:at25percent:x", val=np.nan, units="m")
        self.add_input("data:TLAR:luggage_mass_design", val=np.nan, units="kg")

        self.add_output("data_mod:geometry:cabin:seats:passenger:NPAX_max")
        self.add_output("data_mod:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25", units="m")
        self.add_output("data_mod:geometry:wing:MAC:at25percent:x", units="m")
        self.add_output("data_mod:TLAR:NPAX_design")
        self.add_output("data_mod:TLAR:luggage_mass_design", units="kg")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        npax_max = inputs["data:geometry:cabin:seats:passenger:NPAX_max"]
        npax_design = inputs["data:TLAR:NPAX_design"]
        seats_p_row = inputs["data:geometry:cabin:seats:passenger:count_by_row"]
        ht_lp = inputs["data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25"]
        l_pass_seats = inputs["data:geometry:cabin:seats:passenger:length"]
        fa_length = inputs["data:geometry:wing:MAC:at25percent:x"]
        luggage_mass_design = inputs["data:TLAR:luggage_mass_design"]

        # Initialization of the output variables
        npax_max_mod = npax_max
        npax_design_mod = npax_design + self.options["fuselage_mod"][2]
        luggage_mass_design_mod = luggage_mass_design + self.options["fuselage_mod"][3]
        ht_lp_mod = ht_lp
        fa_length_mod = fa_length

        if self.options["fuselage_mod"][0] == 0:
            npax_max_mod = npax_max + seats_p_row
            added_length = l_pass_seats
        else:
            added_length = self.options["fuselage_mod"][0]

        x_ratio_front = self.options["fuselage_mod"][1]
        x_ratio_rear = self.options["fuselage_mod"][2]
        fa_length_mod += x_ratio_front * added_length
        ht_lp_mod += x_ratio_rear * added_length

        outputs["data_mod:geometry:cabin:seats:passenger:NPAX_max"] = npax_max_mod
        outputs["data_mod:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25"] = ht_lp_mod
        outputs["data_mod:geometry:wing:MAC:at25percent:x"] = fa_length_mod
        outputs["data_mod:TLAR:NPAX_design"] = npax_design_mod
        outputs["data_mod:TLAR:luggage_mass_design"] = luggage_mass_design_mod
