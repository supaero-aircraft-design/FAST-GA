"""Computation of the airframe total mass."""
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


class ComputeAirframeMass(om.ExplicitComponent):
    """
    Computes the airframe total mass.
    """

    def setup(self):

        self.add_input("data:weight:airframe:wing:mass", val=np.nan, units="kg")
        self.add_input("data:weight:airframe:fuselage:mass", val=np.nan, units="kg")
        self.add_input("data:weight:airframe:horizontal_tail:mass", val=np.nan, units="kg")
        self.add_input("data:weight:airframe:vertical_tail:mass", val=np.nan, units="kg")
        self.add_input("data:weight:airframe:flight_controls:mass", val=np.nan, units="kg")
        self.add_input("data:weight:airframe:landing_gear:main:mass", val=np.nan, units="kg")
        self.add_input("data:weight:airframe:landing_gear:front:mass", val=np.nan, units="kg")
        self.add_input("data:weight:airframe:paint:mass", val=np.nan, units="kg")

        self.add_output("data:weight:airframe:mass", units="kg", desc="Mass of the airframe")

        self.declare_partials(of="*", wrt="*", val=1.0)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        m_wing = inputs["data:weight:airframe:wing:mass"]
        m_fuse = inputs["data:weight:airframe:fuselage:mass"]
        m_ht = inputs["data:weight:airframe:horizontal_tail:mass"]
        m_vt = inputs["data:weight:airframe:vertical_tail:mass"]
        m_fc = inputs["data:weight:airframe:flight_controls:mass"]
        m_mlg = inputs["data:weight:airframe:landing_gear:main:mass"]
        m_nlg = inputs["data:weight:airframe:landing_gear:front:mass"]
        m_paint = inputs["data:weight:airframe:paint:mass"]

        outputs["data:weight:airframe:mass"] = (
            m_wing + m_fuse + m_ht + m_vt + m_fc + m_mlg + m_nlg + m_paint
        )
