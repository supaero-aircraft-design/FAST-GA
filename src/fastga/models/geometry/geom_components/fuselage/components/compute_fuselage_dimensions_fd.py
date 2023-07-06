"""
    Estimation of geometry of fuselage part A - Cabin (Commercial) based on Fixed tail Distance
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
from openmdao.core.explicitcomponent import ExplicitComponent
from openmdao.core.group import Group

# import fastoad.api as oad

# from ..constants import SUBMODEL_FUSELAGE_DIMENSIONS


# @oad.RegisterSubmodel(
#     SUBMODEL_FUSELAGE_DIMENSIONS, "fastga.submodel.geometry.fuselage.dimensions.fixed_tail_distance"
# )
class ComputeFuselageGeometryCabinSizingFD(Group):
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
        # FIXME: declare proper partials without int values

        self.add_subsystem("comp_npax", ComputeNPAX(), promotes=["*"])
        self.add_subsystem("comp_pax_length", ComputePAXLength(), promotes=["*"])
        self.add_subsystem("comp_max_width", ComputeMaxWidth(), promotes=["*"])
        self.add_subsystem("comp_max_height", ComputeMaxHeight(), promotes=["*"])
        self.add_subsystem("comp_lugg_length", ComputeLuggageLength(), promotes=["*"])
        self.add_subsystem("comp_cabin_length", ComputeCabinLength(), promotes=["*"])
        self.add_subsystem("comp_nose_length", ComputeNoseLength(), promotes=["*"])
        self.add_subsystem("comp_fuselage_length", ComputeFuselageLength(), promotes=["*"])
        self.add_subsystem("comp_plane_length", ComputePlaneLength(), promotes=["*"])
        self.add_subsystem("comp_rear_length", ComputeRearLength(), promotes=["*"])


class ComputeNPAX(ExplicitComponent):
    """
    Computes number of pax cabin.
    """

    def setup(self):

        self.add_input("data:geometry:cabin:seats:passenger:NPAX_max", val=np.nan)
        self.add_input("data:geometry:cabin:seats:passenger:count_by_row", val=np.nan)

        self.add_output("data:geometry:cabin:NPAX")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        
        npax_max = inputs["data:geometry:cabin:seats:passenger:NPAX_max"]
        seats_p_row = inputs["data:geometry:cabin:seats:passenger:count_by_row"]

        # noinspection PyBroadException
        npax = np.ceil(float(npax_max) / float(seats_p_row)) * float(seats_p_row)

        outputs["data:geometry:cabin:NPAX"] = npax


class ComputePAXLength(ExplicitComponent):
    """
    Computes Length of pax cabin.
    """
    
    def setup(self):

        self.add_input("data:geometry:cabin:seats:pilot:length", val=np.nan, units="m")
        self.add_input("data:geometry:cabin:seats:passenger:length", val=np.nan, units="m")
        self.add_input("data:geometry:cabin:seats:passenger:count_by_row", val=np.nan)
        self.add_input("data:geometry:cabin:NPAX", val=np.nan)

        self.add_output("data:geometry:fuselage:PAX_length", units="m")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        
        l_pilot_seats = inputs["data:geometry:cabin:seats:pilot:length"]
        l_pass_seats = inputs["data:geometry:cabin:seats:passenger:length"]
        seats_p_row = inputs["data:geometry:cabin:seats:passenger:count_by_row"]
        npax = inputs["data:geometry:cabin:NPAX"]

        n_rows = npax / float(seats_p_row)
        l_pax = l_pilot_seats + n_rows * l_pass_seats

        outputs["data:geometry:fuselage:PAX_length"] = l_pax


class ComputeMaxWidth(ExplicitComponent):
    """
    Computes maximum cabin width.

    Cabin width considered is for side by side seats and it is computed based on 
    cylindrical fuselage.
    """
    
    def setup(self):

        self.add_input("data:geometry:cabin:seats:pilot:width", val=np.nan, units="m")
        self.add_input("data:geometry:cabin:seats:passenger:width", val=np.nan, units="m")
        self.add_input("data:geometry:cabin:seats:passenger:count_by_row", val=np.nan)
        self.add_input("data:geometry:cabin:aisle_width", val=np.nan, units="m")
        
        self.add_output("data:geometry:fuselage:maximum_width", units="m")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        
        w_pilot_seats = inputs["data:geometry:cabin:seats:pilot:width"]
        w_pass_seats = inputs["data:geometry:cabin:seats:passenger:width"]
        seats_p_row = inputs["data:geometry:cabin:seats:passenger:count_by_row"]
        w_aisle = inputs["data:geometry:cabin:aisle_width"]

        w_cabin = max(2 * w_pilot_seats, seats_p_row * w_pass_seats + w_aisle)
        r_i = w_cabin / 2
        radius = 1.06 * r_i
        
        b_f = 2 * radius

        outputs["data:geometry:fuselage:maximum_width"] = b_f


class ComputeMaxHeight(ExplicitComponent):
    """
    Computes maximum cabin height.
    """
    
    def setup(self):

        self.add_input("data:geometry:fuselage:maximum_width", val=np.nan, units="m")

        self.add_output("data:geometry:fuselage:maximum_height", units="m")

        self.declare_partials("*", "*", val=1.0)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        
        b_f = inputs["data:geometry:fuselage:maximum_width"]

        # 0.14m is the distance between both lobe centers of the fuselage
        h_f = b_f + 0.14

        outputs["data:geometry:fuselage:maximum_height"] = h_f


class ComputeLuggageLength(ExplicitComponent):
    """
    Computes luggage length.
    
    80% of internal radius section can be filled with luggage.
    """
    
    def setup(self):

        self.add_input("data:geometry:cabin:seats:pilot:width", val=np.nan, units="m")
        self.add_input("data:geometry:cabin:seats:passenger:width", val=np.nan, units="m")
        self.add_input("data:geometry:cabin:seats:passenger:count_by_row", val=np.nan)
        self.add_input("data:geometry:cabin:aisle_width", val=np.nan, units="m")
        self.add_input("data:geometry:cabin:luggage:mass_max", val=np.nan, units="kg")
        
        self.add_output("data:geometry:fuselage:luggage_length", units="m")

        self.declare_partials("*", "*", method="fd")  

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        
        w_pilot_seats = inputs["data:geometry:cabin:seats:pilot:width"]
        w_pass_seats = inputs["data:geometry:cabin:seats:passenger:width"]
        seats_p_row = inputs["data:geometry:cabin:seats:passenger:count_by_row"]
        w_aisle = inputs["data:geometry:cabin:aisle_width"]
        luggage_mass_max = inputs["data:geometry:cabin:luggage:mass_max"]

        w_cabin = max(2 * w_pilot_seats, seats_p_row * w_pass_seats + w_aisle)
        r_i = w_cabin / 2
        
        luggage_density = 161.0  # In kg/m3
        l_lug = (luggage_mass_max / luggage_density) / (0.8 * np.pi * r_i ** 2)

        outputs["data:geometry:fuselage:luggage_length"] = l_lug


class ComputeCabinLength(ExplicitComponent):
    """
    Computes cabin total length.
    """
    
    def setup(self):

        self.add_input("data:geometry:fuselage:PAX_length", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:luggage_length", val=np.nan, units="m")

        self.add_output("data:geometry:cabin:length", units="m")

        self.declare_partials("*", "data:geometry:fuselage:PAX_length", val=1.0)
        self.declare_partials("*", "data:geometry:fuselage:luggage_length", val=1.0)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        
        l_pax = inputs["data:geometry:fuselage:PAX_length"]
        l_lug = inputs["data:geometry:fuselage:luggage_length"]

        # Length of instrument panel
        l_instr = 0.7

        cabin_length = l_instr + l_pax + l_lug

        outputs["data:geometry:cabin:length"] = cabin_length

        
class ComputeNoseLength(ExplicitComponent):
    """
    Computes nose length.
    """
    
    def setup(self):

        self.add_input("data:geometry:propulsion:nacelle:length", val=np.nan, units="m")
        self.add_input("data:geometry:propulsion:engine:layout", val=np.nan)
        self.add_input("data:geometry:fuselage:maximum_height", val=np.nan, units="m")
        self.add_input("data:geometry:propeller:depth", val=np.nan, units="m")

        self.add_output("data:geometry:fuselage:front_length", units="m")

        self.declare_partials(
            "*", 
            ["data:geometry:propulsion:nacelle:length", 
             "data:geometry:fuselage:maximum_height"], 
            method="fd"
        )  

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        
        nacelle_length = inputs["data:geometry:propulsion:nacelle:length"]
        prop_layout = inputs["data:geometry:propulsion:engine:layout"]
        h_f = inputs["data:geometry:fuselage:maximum_height"]
        spinner_length = inputs["data:geometry:propeller:depth"]

        if prop_layout == 3.0:  # engine located in nose
            lav = nacelle_length + spinner_length
        else:
            lav = 1.40 * h_f
            # Used to be 1.7, supposedly as an A320 according to FAST legacy. Results on the BE76
            # tend to say it is around 1.40, though it varies a lot depending on the airplane and
            # its use

        outputs["data:geometry:fuselage:front_length"] = lav


class ComputeFuselageLength(ExplicitComponent):
    """
    Computes fuselage length.
    """
    
    def setup(self):

        self.add_input("data:geometry:wing:MAC:at25percent:x", val=np.nan, units="m")
        self.add_input(
            "data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25", val=np.nan, units="m"
        )
        self.add_input(
            "data:geometry:vertical_tail:MAC:at25percent:x:from_wingMAC25", val=np.nan, units="m"
        )
        self.add_input("data:geometry:horizontal_tail:MAC:length", val=np.nan, units="m")
        self.add_input("data:geometry:vertical_tail:MAC:length", val=np.nan, units="m")
        
        self.add_output("data:geometry:fuselage:length", val=10.0, units="m")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        
        fa_length = inputs["data:geometry:wing:MAC:at25percent:x"]
        ht_lp = inputs["data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25"]
        vt_lp = inputs["data:geometry:vertical_tail:MAC:at25percent:x:from_wingMAC25"]
        ht_length = inputs["data:geometry:horizontal_tail:MAC:length"]
        vt_length = inputs["data:geometry:vertical_tail:MAC:length"]

        fus_length = fa_length + max(ht_lp + 0.75 * ht_length, vt_lp + 0.75 * vt_length)

        outputs["data:geometry:fuselage:length"] = fus_length


class ComputePlaneLength(ExplicitComponent):
    """
    Computes fuselage plane length.
    """

    def setup(self):

        self.add_input("data:geometry:wing:MAC:at25percent:x", val=np.nan, units="m")
        self.add_input(
            "data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25", val=np.nan, units="m"
        )
        self.add_input(
            "data:geometry:vertical_tail:MAC:at25percent:x:from_wingMAC25", val=np.nan, units="m"
        )
        self.add_input("data:geometry:horizontal_tail:MAC:length", val=np.nan, units="m")
        self.add_input("data:geometry:vertical_tail:MAC:length", val=np.nan, units="m")
        self.add_input("data:geometry:horizontal_tail:sweep_25", val=np.nan, units="deg")
        self.add_input("data:geometry:horizontal_tail:span", val=np.nan, units="m")
        self.add_input("data:geometry:vertical_tail:sweep_25", val=np.nan, units="deg")
        self.add_input("data:geometry:vertical_tail:span", val=np.nan, units="m")
        
        self.add_output("data:geometry:aircraft:length", units="m")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        
        fa_length = inputs["data:geometry:wing:MAC:at25percent:x"]
        ht_lp = inputs["data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25"]
        vt_lp = inputs["data:geometry:vertical_tail:MAC:at25percent:x:from_wingMAC25"]
        ht_length = inputs["data:geometry:horizontal_tail:MAC:length"]
        vt_length = inputs["data:geometry:vertical_tail:MAC:length"]
        sweep_25_vt = inputs["data:geometry:vertical_tail:sweep_25"]
        b_v = inputs["data:geometry:vertical_tail:span"]
        sweep_25_ht = inputs["data:geometry:horizontal_tail:sweep_25"]
        b_h = inputs["data:geometry:horizontal_tail:span"]

        plane_length = fa_length + max(
            ht_lp + 0.75 * ht_length + b_h / 2.0 * np.tan(sweep_25_ht * np.pi / 180),
            vt_lp + 0.75 * vt_length + b_v * np.tan(sweep_25_vt * np.pi / 180),
        )

        outputs["data:geometry:aircraft:length"] = plane_length


class ComputeRearLength(ExplicitComponent):
    """
    Computes fuselage rear length.
    """
    
    def setup(self):

        self.add_input("data:geometry:fuselage:length", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:front_length", val=np.nan, units="m")
        self.add_input("data:geometry:cabin:length", val=np.nan, units="m")
        
        self.add_output("data:geometry:fuselage:rear_length", units="m")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        
        fus_length = inputs["data:geometry:fuselage:length"]
        lav = inputs["data:geometry:fuselage:front_length"]
        cabin_length = inputs["data:geometry:cabin:length"]

        lar = fus_length - (lav + cabin_length)

        outputs["data:geometry:fuselage:rear_length"] = lar
