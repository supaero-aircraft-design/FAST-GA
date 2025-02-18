"""
Estimation of geometry of fuselage part A - Cabin (Commercial).
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
from openmdao.core.explicitcomponent import ExplicitComponent
from fuselage_dimension_components import (
    ComputeFuselageCabinLength,
    ComputeFuselageLuggageLength,
    ComputeFuselageMaxHeight,
    ComputeFuselageMaxWidth,
    ComputeFuselageNPAX,
    ComputeFuselagePAXLength,
    ComputeFuselageLengthFD,
    ComputeFuselageLengthFL,
    ComputeFuselageNoseLengthFD,
    ComputeFuselageNoseLengthFL,
    ComputePlaneLength,
    ComputeFuselageRearLength,
)


class ComputeFuselageGeometryBasic(ExplicitComponent):
    """
    Geometry of fuselage - Cabin length defined with total fuselage length input (no sizing).
    """

    def setup(self):
        self.add_input("data:geometry:fuselage:length", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:front_length", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:rear_length", val=np.nan, units="m")

        self.add_output("data:geometry:cabin:length", units="m")

        self.declare_partials("*", "data:geometry:fuselage:length", val=1.0)
        self.declare_partials(
            "*",
            ["data:geometry:fuselage:front_length", "data:geometry:fuselage:rear_length"],
            val=-1.0,
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        fus_length = inputs["data:geometry:fuselage:length"]
        lav = inputs["data:geometry:fuselage:front_length"]
        lar = inputs["data:geometry:fuselage:rear_length"]

        # Cabin total length
        cabin_length = fus_length - (lav + lar)

        outputs["data:geometry:cabin:length"] = cabin_length


class ComputeFuselageGeometryCabinSizingFD(om.Group):
    # TODO: Document equations. Cite sources
    """
    Geometry of fuselage - Cabin is sized based on layout (seats, aisle...) and HTP/VTP position
    (Fixed tail Distance).
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._engine_wrapper = None

    def initialize(self):
        self.options.declare("propulsion_id", default="", types=str)

    def setup(self):
        self.add_subsystem(
            "number_of_passenger",
            ComputeFuselageNPAX(),
            promotes=["*"],
        )
        self.add_subsystem(
            "passenger_area_length",
            ComputeFuselagePAXLength(),
            promotes=["*"],
        )
        self.add_subsystem(
            "maximum_width",
            ComputeFuselageMaxWidth(),
            promotes=["*"],
        )
        self.add_subsystem(
            "maximum_height",
            ComputeFuselageMaxHeight(),
            promotes=["*"],
        )
        self.add_subsystem(
            "luggage_area_length",
            ComputeFuselageLuggageLength(),
            promotes=["*"],
        )
        self.add_subsystem(
            "cabin_length",
            ComputeFuselageCabinLength(),
            promotes=["*"],
        )
        self.add_subsystem(
            "nose_length",
            ComputeFuselageNoseLengthFD(),
            promotes=["*"],
        )




class ComputeFuselageGeometryCabinSizingFL(ExplicitComponent):
    # TODO: Document equations. Cite sources
    """
    Geometry of fuselage - Cabin is sized based on layout (seats, aisle...) and additional rear
    length (Fixed Length).
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._engine_wrapper = None

    def initialize(self):
        self.options.declare("propulsion_id", default="", types=str)

    def setup(self):
        self.add_input("data:geometry:cabin:seats:passenger:NPAX_max", val=np.nan)
        self.add_input("data:geometry:cabin:seats:pilot:length", val=np.nan, units="m")
        self.add_input("data:geometry:cabin:seats:pilot:width", val=np.nan, units="m")
        self.add_input("data:geometry:cabin:seats:passenger:length", val=np.nan, units="m")
        self.add_input("data:geometry:cabin:seats:passenger:width", val=np.nan, units="m")
        self.add_input("data:geometry:cabin:seats:passenger:count_by_row", val=np.nan)
        self.add_input("data:geometry:cabin:aisle_width", val=np.nan, units="m")
        self.add_input("data:geometry:cabin:luggage:mass_max", val=np.nan, units="kg")
        self.add_input("data:geometry:fuselage:rear_length", units="m")
        self.add_input("data:geometry:propulsion:nacelle:length", val=np.nan, units="m")
        self.add_input("data:geometry:propulsion:engine:layout", val=np.nan)

        self.add_output("data:geometry:cabin:NPAX")
        self.add_output("data:geometry:fuselage:length", val=10.0, units="m")
        self.add_output("data:geometry:fuselage:maximum_width", units="m")
        self.add_output("data:geometry:fuselage:maximum_height", units="m")
        self.add_output("data:geometry:fuselage:front_length", units="m")
        self.add_output("data:geometry:fuselage:PAX_length", units="m")
        self.add_output("data:geometry:cabin:length", units="m")
        self.add_output("data:geometry:fuselage:luggage_length", units="m")

        self.declare_partials(
            "*", "*", method="fd"
        )  # FIXME: declare proper partials without int values

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        nacelle_length = inputs["data:geometry:propulsion:nacelle:length"]
        npax_max = inputs["data:geometry:cabin:seats:passenger:NPAX_max"]
        l_pilot_seats = inputs["data:geometry:cabin:seats:pilot:length"]
        w_pilot_seats = inputs["data:geometry:cabin:seats:pilot:width"]
        l_pass_seats = inputs["data:geometry:cabin:seats:passenger:length"]
        w_pass_seats = inputs["data:geometry:cabin:seats:passenger:width"]
        seats_p_row = inputs["data:geometry:cabin:seats:passenger:count_by_row"]
        w_aisle = inputs["data:geometry:cabin:aisle_width"]
        luggage_mass_max = inputs["data:geometry:cabin:luggage:mass_max"]
        prop_layout = inputs["data:geometry:propulsion:engine:layout"]
        lar = inputs["data:geometry:fuselage:rear_length"]

        # Length of instrument panel
        l_instr = 0.7
        # Length of pax cabin
        # noinspection PyBroadException
        npax = np.ceil(float(npax_max) / float(seats_p_row)) * float(seats_p_row)
        n_rows = npax / float(seats_p_row)
        l_pax = l_pilot_seats + n_rows * l_pass_seats
        # Cabin width considered is for side by side seats
        w_cabin = max(2 * w_pilot_seats, seats_p_row * w_pass_seats + w_aisle)
        r_i = w_cabin / 2
        radius = 1.06 * r_i
        # Cylindrical fuselage
        b_f = 2 * radius
        # 0.14m is the distance between both lobe centers of the fuselage
        h_f = b_f + 0.14
        # Luggage length (80% of internal radius section can be filled with luggage)
        luggage_density = 161.0  # In kg/m3
        l_lug = (luggage_mass_max / luggage_density) / (0.8 * np.pi * r_i**2)
        # Cabin total length
        cabin_length = l_instr + l_pax + l_lug
        # Calculate nose length
        if prop_layout == 3.0:  # engine located in nose
            lav = nacelle_length
        else:
            lav = 1.7 * h_f
            # Calculate fuselage length
        fus_length = lav + cabin_length + lar

        outputs["data:geometry:cabin:NPAX"] = npax
        outputs["data:geometry:fuselage:length"] = fus_length
        outputs["data:geometry:fuselage:maximum_width"] = b_f
        outputs["data:geometry:fuselage:maximum_height"] = h_f
        outputs["data:geometry:fuselage:front_length"] = lav
        outputs["data:geometry:fuselage:PAX_length"] = l_pax
        outputs["data:geometry:cabin:length"] = cabin_length
        outputs["data:geometry:fuselage:luggage_length"] = l_lug
