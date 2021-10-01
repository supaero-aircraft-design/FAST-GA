"""
    Computation of the wing span modifications.
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
from fastga.models.aerodynamics.constants import ENGINE_COUNT


class ComputeSpan(om.ExplicitComponent):
    """
    Please refer to the notebooks for the class parameters definition.
    """

    def initialize(self):
        self.options.declare("span_mod", types=list, default=[1.0, True, True])

    def setup(self):
        self.add_input("data:geometry:wing:aspect_ratio", val=np.nan)
        self.add_input("data:geometry:wing:taper_ratio", val=np.nan)
        self.add_input("data:geometry:wing:area", val=np.nan, units="m**2")
        self.add_input("data:geometry:propulsion:y_ratio", shape=ENGINE_COUNT, val=np.nan)
        self.add_input("data:geometry:propulsion:y_ratio_tank_beginning", val=np.nan)
        self.add_input("data:geometry:propulsion:y_ratio_tank_end", val=np.nan)
        self.add_input("data:geometry:flap:span_ratio", val=np.nan)
        self.add_input("data:geometry:aileron:span_ratio", val=np.nan)
        self.add_input("data:geometry:cabin:seats:pilot:width", val=np.nan, units="m")
        self.add_input("data:geometry:cabin:seats:passenger:width", val=np.nan, units="m")
        self.add_input("data:geometry:cabin:seats:passenger:count_by_row", val=np.nan)
        self.add_input("data:geometry:cabin:aisle_width", val=np.nan, units="m")

        self.add_output("data_mod:geometry:wing:aspect_ratio")
        self.add_output("data_mod:geometry:wing:taper_ratio")
        self.add_output("data_mod:geometry:wing:area", units="m**2")
        self.add_output("data_mod:geometry:propulsion:y_ratio", shape=ENGINE_COUNT)
        self.add_output("data_mod:geometry:propulsion:y_ratio_tank_beginning")
        self.add_output("data_mod:geometry:propulsion:y_ratio_tank_end")
        self.add_output("data_mod:geometry:flap:span_ratio")
        self.add_output("data_mod:geometry:aileron:span_ratio")
        self.add_output("data_mod:settings:span_mod:span_multiplier")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        aspect_ratio_ref = inputs["data:geometry:wing:aspect_ratio"]
        taper_ratio_ref = inputs["data:geometry:wing:taper_ratio"]
        area_ref = inputs["data:geometry:wing:area"]
        y_ratio = inputs["data:geometry:propulsion:y_ratio"]
        y_ratio_tank_beginning = inputs["data:geometry:propulsion:y_ratio_tank_beginning"]
        y_ratio_tank_end = inputs["data:geometry:propulsion:y_ratio_tank_end"]
        flap_span_ratio = inputs["data:geometry:flap:span_ratio"]
        aileron_span_ratio = inputs["data:geometry:aileron:span_ratio"]
        w_pilot_seats = inputs["data:geometry:cabin:seats:pilot:width"]
        w_pass_seats = inputs["data:geometry:cabin:seats:passenger:width"]
        seats_p_row = inputs["data:geometry:cabin:seats:passenger:count_by_row"]
        w_aisle = inputs["data:geometry:cabin:aisle_width"]

        # Compute reference architecture's fuselage radius to have the reference wing root and tip y coordinates
        # (same method as in compute_fuselage in geometry module)
        width_cabin_ref = max(2 * w_pilot_seats, seats_p_row * w_pass_seats + w_aisle)
        radius_int = width_cabin_ref / 2
        radius_fuselage = 1.06 * radius_int
        y_root_ref = radius_fuselage
        span_ref = np.sqrt(aspect_ratio_ref * area_ref)
        y_tip_ref = span_ref / 2

        multiplier = self.options["span_mod"][0]

        # Compute the quantities that define the modified geometry in the initial .xml file
        taper_ratio_mod = multiplier * taper_ratio_ref + (1 - multiplier)
        area_mod = (
            (2 * y_root_ref + (multiplier * y_tip_ref - y_root_ref) * (1 + taper_ratio_mod))
            / (2 * y_root_ref + (y_tip_ref - y_root_ref) * (1 + taper_ratio_ref))
            * area_ref
        )
        aspect_ratio_mod = (span_ref * multiplier) ** 2 / area_mod

        # Modify the y-ratio of the engines if their position is fixed along the span
        if self.options["span_mod"][1]:
            y_ratio = y_ratio / multiplier

        # Compute new y-ratio defining the beginning of the wing fuel tanks to take in account the span increase
        y_ratio_tank_beginning = y_ratio_tank_beginning / multiplier

        # Compute new y-ratio defining the end of the wing fuel tanks.
        if y_ratio_tank_end < 1.0 or not self.options["span_mod"][2]:
            y_ratio_tank_end = y_ratio_tank_end / multiplier

        # Compute new y-ratio for control surfaces. Flaps stay put, and an aileron is added on the extended part
        flap_span_ratio_mod = flap_span_ratio / multiplier
        aileron_span_ratio_mod = 1 + (aileron_span_ratio - 1) / multiplier

        outputs["data_mod:geometry:wing:aspect_ratio"] = aspect_ratio_mod
        outputs["data_mod:geometry:wing:taper_ratio"] = taper_ratio_mod
        outputs["data_mod:geometry:wing:area"] = area_mod
        outputs["data_mod:geometry:propulsion:y_ratio"] = y_ratio
        outputs["data_mod:geometry:propulsion:y_ratio_tank_beginning"] = y_ratio_tank_beginning
        outputs["data_mod:geometry:propulsion:y_ratio_tank_end"] = y_ratio_tank_end
        outputs["data_mod:geometry:flap:span_ratio"] = flap_span_ratio_mod
        outputs["data_mod:geometry:aileron:span_ratio"] = aileron_span_ratio_mod
        outputs["data_mod:settings:span_mod:span_multiplier"] = self.options["span_mod"][0]
