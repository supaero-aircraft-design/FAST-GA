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


class UpdateXML(om.Group):

    def initialize(self):
        self.options.declare("span_mod", types=list, default=[1.0, True, True])
        self.options.declare("fuselage_mod", types=list, default=[0, 0, 0, 0, 0])

    def setup(self):
        if self.options["span_mod"][0] != 1.0:
            self.add_subsystem("update_span", _UpdateSpan(), promotes=["*"])
        if self.options["fuselage_mod"][1] != 0 or self.options["fuselage_mod"][2] != 0:
            self.add_subsystem("update_fuselage", _UpdateFuselage(), promotes=["*"])


class _UpdateSpan(om.ExplicitComponent):

    def setup(self):

        self.add_input("data_mod:geometry:wing:aspect_ratio", val=np.nan)
        self.add_input("data_mod:geometry:wing:taper_ratio", val=np.nan)
        self.add_input("data_mod:geometry:wing:area", val=np.nan, units="m**2")
        self.add_input("data_mod:geometry:propulsion:y_ratio", val=np.nan)
        self.add_input("data_mod:geometry:propulsion:y_ratio_tank_beginning", val=np.nan)
        self.add_input("data_mod:geometry:propulsion:y_ratio_tank_end", val=np.nan)
        self.add_input("data_mod:geometry:flap:span_ratio", val=np.nan)
        self.add_input("data_mod:geometry:wing:aileron:span_ratio", val=np.nan)
        self.add_input("data_mod:settings:span_mod:span_multiplier", val=np.nan)

        self.add_output("data:geometry:wing:aspect_ratio")
        self.add_output("data:geometry:wing:taper_ratio")
        self.add_output("data:geometry:wing:area", units="m**2")
        self.add_output("data:geometry:propulsion:y_ratio")
        self.add_output("data:geometry:propulsion:y_ratio_tank_beginning")
        self.add_output("data:geometry:propulsion:y_ratio_tank_end")
        self.add_output("data:geometry:flap:span_ratio")
        self.add_output("data:geometry:wing:aileron:span_ratio")
        self.add_output("settings:span_mod:span_multiplier")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs["data:geometry:wing:aspect_ratio"] = inputs["data_mod:geometry:wing:aspect_ratio"]
        outputs["data:geometry:wing:taper_ratio"] = inputs["data_mod:geometry:wing:taper_ratio"]
        outputs["data:geometry:wing:area"] = inputs["data_mod:geometry:wing:area"]
        outputs["data:geometry:propulsion:y_ratio"] = inputs["data_mod:geometry:propulsion:y_ratio"]
        outputs["data:geometry:propulsion:y_ratio_tank_beginning"] = inputs["data_mod:geometry:propulsion:y_ratio_tank_beginning"]
        outputs["data:geometry:propulsion:y_ratio_tank_end"] = inputs["data_mod:geometry:propulsion:y_ratio_tank_end"]
        outputs["data:geometry:flap:span_ratio"] = inputs["data_mod:geometry:flap:span_ratio"]
        outputs["data:geometry:wing:aileron:span_ratio"] = inputs["data_mod:geometry:wing:aileron:span_ratio"]
        outputs["settings:span_mod:span_multiplier"] = inputs["data_mod:settings:span_mod:span_multiplier"]


class _UpdateFuselage(om.ExplicitComponent):

    def setup(self):

        self.add_input("data_mod:geometry:cabin:seats:passenger:NPAX_max", val=np.nan)
        self.add_input("data_mod:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25", val=np.nan, units="m")
        self.add_input("data_mod:geometry:wing:MAC:at25percent:x", val=np.nan, units="m")
        self.add_input("data_mod:TLAR:NPAX_design", val=np.nan)
        self.add_input("data_mod:TLAR:luggage_mass_design", val=np.nan, units="kg")

        self.add_output("data:geometry:cabin:seats:passenger:NPAX_max")
        self.add_output("data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25", units="m")
        self.add_output("data:geometry:wing:MAC:at25percent:x", units="m")
        self.add_output("data:TLAR:NPAX_design")
        self.add_output("data:TLAR:luggage_mass_design", units="kg")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs["data:geometry:cabin:seats:passenger:NPAX_max"] = inputs["data_mod:geometry:cabin:seats:passenger:NPAX_max"]
        outputs["data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25"] = inputs["data_mod:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25"]
        outputs["data:geometry:wing:MAC:at25percent:x"] = inputs["data_mod:geometry:wing:MAC:at25percent:x"]
        outputs["data:TLAR:NPAX_design"] = inputs["data_mod:TLAR:NPAX_design"]
        outputs["data:TLAR:luggage_mass_design"] = inputs["data_mod:TLAR:luggage_mass_design"]
