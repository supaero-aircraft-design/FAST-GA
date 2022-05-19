"""FAST - Copyright (c) 2016 ONERA ISAE. """
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
import math
import openmdao.api as om


class InFlightCGVariation(om.ExplicitComponent):
    """
    Computes the coefficient necessary to the calculation of the cg position at any point of
    the DESIGN flight.
    """

    def setup(self):
        self.add_input("data:TLAR:NPAX_design", val=np.nan)
        self.add_input("data:TLAR:luggage_mass_design", val=np.nan, units="kg")
        self.add_input("data:weight:payload:rear_fret:CG:x", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:front_length", val=np.nan, units="m")
        self.add_input("data:geometry:cabin:seats:pilot:length", val=np.nan, units="m")
        self.add_input("data:geometry:cabin:seats:passenger:length", val=np.nan, units="m")
        self.add_input("data:geometry:cabin:seats:passenger:count_by_row", val=np.nan)
        self.add_input("data:weight:aircraft_empty:CG:x", val=np.nan, units="m")
        self.add_input("data:weight:aircraft_empty:mass", val=np.nan, units="kg")
        self.add_input("data:weight:aircraft:payload", val=np.nan, units="kg")
        self.add_input(
            "settings:weight:aircraft:payload:design_mass_per_passenger",
            val=80.0,
            units="kg",
            desc="Design value of mass per passenger",
        )

        self.add_output(
            "data:weight:aircraft:in_flight_variation:fixed_mass_comp:equivalent_moment",
            units="kg*m",
        )
        self.add_output("data:weight:aircraft:in_flight_variation:fixed_mass_comp:mass", units="kg")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        npax = inputs["data:TLAR:NPAX_design"]
        count_by_row = inputs["data:geometry:cabin:seats:passenger:count_by_row"]
        luggage_weight = inputs["data:TLAR:luggage_mass_design"]
        cg_rear_fret = inputs["data:weight:payload:rear_fret:CG:x"]
        l_pilot_seat = inputs["data:geometry:cabin:seats:pilot:length"]
        l_pass_seat = inputs["data:geometry:cabin:seats:passenger:length"]
        design_mass_p_pax = inputs["settings:weight:aircraft:payload:design_mass_per_passenger"]
        x_cg_plane_aft = inputs["data:weight:aircraft_empty:CG:x"]
        m_empty = inputs["data:weight:aircraft_empty:mass"]
        lav = inputs["data:geometry:fuselage:front_length"]
        payload = inputs["data:weight:aircraft:payload"]

        l_instr = 0.7
        # Seats and passengers gravity center (hypothesis of 2 pilots)
        nrows = math.ceil(npax / count_by_row)
        x_cg_passenger = lav + l_instr + l_pilot_seat * 2.0 / (npax + 2.0)
        for idx in range(nrows):
            length = l_pilot_seat + (idx + 0.5) * l_pass_seat
            nb_pers = min(count_by_row, npax - idx * count_by_row)
            x_cg_passenger = x_cg_passenger + length * nb_pers / (npax + 2.0)

        x_cg_payload = (
            x_cg_passenger * (2.0 + npax) * design_mass_p_pax + cg_rear_fret * luggage_weight
        ) / payload

        equivalent_moment = m_empty * x_cg_plane_aft + payload * x_cg_payload
        mass = m_empty + payload

        outputs[
            "data:weight:aircraft:in_flight_variation:fixed_mass_comp:equivalent_moment"
        ] = equivalent_moment
        outputs["data:weight:aircraft:in_flight_variation:fixed_mass_comp:mass"] = mass
