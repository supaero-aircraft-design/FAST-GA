"""Computation of the systems total mass."""
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


class ComputeSystemMass(om.ExplicitComponent):
    """
    Computes the aircraft's systems total mass.

    By default, the insulation system, internal lighting system, and permanent security kits are neglected. However, users could define these values in the input data.
    Seats and installation of crew are already evaluated within d2_passenger_seats_weight.
    """

    def setup(self):

        self.add_input("data:weight:systems:power:electric_systems:mass", val=np.nan, units="lb")
        self.add_input("data:weight:systems:power:hydraulic_systems:mass", val=np.nan, units="lb")
        self.add_input(
            "data:weight:systems:life_support:air_conditioning:mass", val=np.nan, units="lb"
        )
        self.add_input("data:weight:systems:life_support:de_icing:mass", val=np.nan, units="lb")
        self.add_input("data:weight:systems:life_support:fixed_oxygen:mass", val=np.nan, units="lb")
        self.add_input("data:weight:systems:life_support:insulation:mass", val=0.0, units="lb")
        self.add_input(
            "data:weight:systems:life_support:internal_lighting:mass", val=0.0, units="lb"
        )
        self.add_input(
            "data:weight:systems:life_support:seat_installation:mass", val=0.0, units="lb"
        )
        self.add_input("data:weight:systems:life_support:security_kits:mass", val=0.0, units="lb")
        self.add_input("data:weight:systems:avionics:mass", val=np.nan, units="lb")
        self.add_input("data:weight:systems:recording:mass", val=np.nan, units="lb")

        self.add_output("data:weight:systems:mass", units="lb", desc="Mass of aircraft systems")

        self.declare_partials(of="*", wrt="*", val=1.0)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        m_electrical = inputs["data:weight:systems:power:electric_systems:mass"]
        m_hydraulic = inputs["data:weight:systems:power:hydraulic_systems:mass"]
        m_ac = inputs["data:weight:systems:life_support:air_conditioning:mass"]
        m_de_icing = inputs["data:weight:systems:life_support:de_icing:mass"]
        m_oxygen = inputs["data:weight:systems:life_support:fixed_oxygen:mass"]
        m_insulation = inputs["data:weight:systems:life_support:insulation:mass"]
        m_lighting = inputs["data:weight:systems:life_support:internal_lighting:mass"]
        m_seat_installation = inputs["data:weight:systems:life_support:seat_installation:mass"]
        m_security_kits = inputs["data:weight:systems:life_support:security_kits:mass"]
        m_avionics = inputs["data:weight:systems:avionics:mass"]
        m_recording = inputs["data:weight:systems:recording:mass"]

        outputs["data:weight:systems:mass"] = (
            m_electrical
            + m_hydraulic
            + m_ac
            + m_de_icing
            + m_oxygen
            + m_insulation
            + m_lighting
            + m_seat_installation
            + m_security_kits
            + m_avionics
            + m_recording
        )
