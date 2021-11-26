"""
Estimation of navigation systems weight.
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

from openmdao.core.explicitcomponent import ExplicitComponent
from fastoad.model_base.atmosphere import Atmosphere


class ComputeNavigationSystemsWeight(ExplicitComponent):
    """
    Weight estimation for navigation systems

    Based on a statistical analysis. See :cite:`roskampart5:1985` Torenbeek method
    """

    def setup(self):

        self.add_input("data:weight:aircraft:MTOW", val=np.nan, units="lb")
        self.add_input("data:geometry:propulsion:engine:count", val=np.nan)
        self.add_input("data:geometry:cabin:seats:passenger:NPAX_max", val=np.nan)

        self.add_output("data:weight:systems:navigation:mass", units="lb")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        mtow = inputs["data:weight:aircraft:MTOW"]
        n_eng = inputs["data:geometry:propulsion:engine:count"]
        n_pax = inputs["data:geometry:cabin:seats:passenger:NPAX_max"]

        n_occ = n_pax + 2.0
        # The formula differs depending on the number of propeller on the engine

        if n_eng == 1.0:
            c3 = 33.0 * n_occ

        else:
            c3 = 40 + 0.008 * mtow  # mass formula in lb

        outputs["data:weight:systems:navigation:mass"] = c3


class ComputeNavigationSystemsWeightFLOPS(ExplicitComponent):
    """
    Weight estimation for navigation systems (includes avionics and instruments)

    Based on a statistical analysis. See :cite:`wells:2017`
    """

    def setup(self):

        self.add_input("data:mission:sizing:cs23:characteristic_speed:vd", val=np.nan, units="m/s")
        self.add_input("data:TLAR:range", val=np.nan, units="nm")

        self.add_input("data:geometry:fuselage:maximum_width", val=np.nan, units="ft")
        self.add_input("data:geometry:fuselage:length", val=np.nan, units="ft")
        self.add_input("data:geometry:propulsion:engine:layout", val=np.nan)
        self.add_input("data:geometry:propulsion:engine:count", val=np.nan)

        self.add_input("data:mission:sizing:main_route:cruise:altitude", val=np.nan, units="ft")

        self.add_output("data:weight:systems:navigation:mass", units="lb")
        self.add_output("data:weight:systems:navigation:instruments:mass", units="lb")
        self.add_output("data:weight:systems:navigation:avionics:mass", units="lb")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        v_limit = inputs["data:mission:sizing:cs23:characteristic_speed:vd"]
        design_range = inputs["data:TLAR:range"]

        fus_width = inputs["data:geometry:fuselage:maximum_width"]
        fus_length = inputs["data:geometry:fuselage:length"]
        prop_layout = inputs["data:geometry:propulsion:engine:layout"]
        prop_count = inputs["data:geometry:propulsion:engine:count"]

        cruise_alt = inputs["data:mission:sizing:main_route:cruise:altitude"]

        n_pilot = 2.0

        atm_cruise = Atmosphere(cruise_alt, altitude_in_feet=True)
        m_limit = v_limit / atm_cruise.speed_of_sound

        fus_plan_area = fus_width * fus_length

        # TODO: Adjust formula in case there is a strange engine location configuration (on nose and on the wing)
        if prop_layout == 3.0 or prop_layout == 2.0:  # engine located in nose or in the rear
            prop_nb_on_wing = 0.0
            prop_nb_on_fus = prop_count

        else:
            prop_nb_on_wing = prop_count
            prop_nb_on_fus = 0.0

        c31 = (
            0.48
            * fus_plan_area ** 0.57
            * m_limit ** 0.5
            * (10.0 + 2.5 * n_pilot + prop_nb_on_wing + 1.5 * prop_nb_on_fus)
        )
        c32 = 15.8 * design_range ** 0.1 * n_pilot ** 0.7 * fus_plan_area ** 0.43
        c3 = c31 + c32

        outputs["data:weight:systems:navigation:mass"] = c3
        outputs["data:weight:systems:navigation:instruments:mass"] = c31
        outputs["data:weight:systems:navigation:avionics:mass"] = c32
