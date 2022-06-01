"""
    New estimation method of center of gravity for all load cases.
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
import scipy.optimize as optimize

from scipy.constants import g

from openmdao.core.explicitcomponent import ExplicitComponent

from stdatm import Atmosphere

# noinspection PyProtectedMember
from fastoad.module_management._bundle_loader import BundleLoader
import fastoad.api as oad
from fastoad.constants import EngineSetting

from .constants import SUBMODEL_LOADCASE_GROUND_X, SUBMODEL_LOADCASE_FLIGHT_X


@oad.RegisterSubmodel(
    SUBMODEL_LOADCASE_GROUND_X, "fastga.submodel.weight.cg.loadcase.ground.legacy"
)
class ComputeGroundCGCase(ExplicitComponent):
    # TODO: Document equations. Cite sources
    """Center of gravity estimation for all load cases on ground."""

    def setup(self):
        self.add_input("data:geometry:cabin:luggage:mass_max", val=np.nan, units="kg")
        self.add_input("data:geometry:wing:MAC:length", val=np.nan, units="m")
        self.add_input("data:geometry:wing:MAC:at25percent:x", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:front_length", val=np.nan, units="m")
        self.add_input("data:geometry:cabin:seats:pilot:length", val=np.nan, units="m")
        self.add_input("data:weight:furniture:passenger_seats:CG:x", val=np.nan, units="m")
        self.add_input("data:weight:payload:rear_fret:CG:x", val=np.nan, units="m")
        self.add_input("data:weight:aircraft_empty:CG:x", val=np.nan, units="m")
        self.add_input("data:weight:aircraft_empty:mass", val=np.nan, units="kg")
        self.add_input("data:weight:propulsion:unusable_fuel:mass", val=np.nan, units="kg")
        self.add_input("data:weight:propulsion:tank:CG:x", val=np.nan, units="m")

        self.add_output("data:weight:aircraft:CG:ground_condition:max:MAC_position")
        self.add_output("data:weight:aircraft:CG:ground_condition:min:MAC_position")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        luggage_mass_max = float(inputs["data:geometry:cabin:luggage:mass_max"])
        l0_wing = inputs["data:geometry:wing:MAC:length"]
        fa_length = inputs["data:geometry:wing:MAC:at25percent:x"]
        cg_pax = inputs["data:weight:furniture:passenger_seats:CG:x"]
        lav = inputs["data:geometry:fuselage:front_length"]
        l_pilot_seat = inputs["data:geometry:cabin:seats:pilot:length"]
        cg_rear_fret = inputs["data:weight:payload:rear_fret:CG:x"]
        x_cg_plane_aft = inputs["data:weight:aircraft_empty:CG:x"]
        m_empty = inputs["data:weight:aircraft_empty:mass"]
        m_unusable_fuel = inputs["data:weight:propulsion:unusable_fuel:mass"]
        cg_tank = inputs["data:weight:propulsion:tank:CG:x"]

        l_instr = 0.7
        cg_pilot = lav + l_instr + l_pilot_seat / 2.0

        m_pilot = 77.0

        cg_list = []

        m_pax_array = np.zeros(1)
        m_pilot_array = np.array([0.0, 2.0 * m_pilot])  # Without the pilots and with the 2 pilots
        m_fuel_array = np.array([m_unusable_fuel])
        m_lug_array = np.array([0.0, luggage_mass_max])

        for m_lug in m_lug_array:

            for m_fuel in m_fuel_array:

                for m_pilot in m_pilot_array:

                    for m_pax in m_pax_array:
                        mass = m_pax + m_pilot + m_fuel + m_lug + m_empty
                        cg = (
                            m_empty * x_cg_plane_aft
                            + m_pax * cg_pax
                            + m_pilot * cg_pilot
                            + m_fuel * cg_tank
                            + m_lug * cg_rear_fret
                        ) / mass
                        cg_list.append(cg)

        cg_fwd = min(cg_list)
        cg_aft = max(cg_list)
        cg_fwd_ratio_pl = (cg_fwd - fa_length + 0.25 * l0_wing) / l0_wing
        cg_aft_ratio_pl = (cg_aft - fa_length + 0.25 * l0_wing) / l0_wing

        outputs["data:weight:aircraft:CG:ground_condition:max:MAC_position"] = cg_aft_ratio_pl
        outputs["data:weight:aircraft:CG:ground_condition:min:MAC_position"] = cg_fwd_ratio_pl


@oad.RegisterSubmodel(
    SUBMODEL_LOADCASE_FLIGHT_X, "fastga.submodel.weight.cg.loadcase.flight.legacy"
)
class ComputeFlightCGCase(ExplicitComponent):
    """Center of gravity estimation for all load cases in flight"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._engine_wrapper = None

    def initialize(self):
        self.options.declare("propulsion_id", default="", types=str)

    def setup(self):
        self._engine_wrapper = BundleLoader().instantiate_component(self.options["propulsion_id"])
        self._engine_wrapper.setup(self)

        self.add_input("data:geometry:cabin:luggage:mass_max", val=np.nan, units="kg")
        self.add_input("data:geometry:wing:area", val=np.nan, units="m**2")
        self.add_input("data:aerodynamics:aircraft:cruise:CD0", val=np.nan)
        self.add_input("data:aerodynamics:wing:cruise:induced_drag_coefficient", val=np.nan)
        self.add_input("data:geometry:cabin:seats:passenger:NPAX_max", val=np.nan)
        self.add_input("data:geometry:wing:MAC:length", val=np.nan, units="m")
        self.add_input("data:geometry:wing:MAC:at25percent:x", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:front_length", val=np.nan, units="m")
        self.add_input("data:geometry:cabin:seats:pilot:length", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:PAX_length", val=np.nan, units="m")
        self.add_input("data:geometry:cabin:seats:passenger:count_by_row", val=np.nan)
        self.add_input("data:geometry:cabin:seats:passenger:length", val=np.nan, units="m")
        self.add_input("data:weight:payload:rear_fret:CG:x", val=np.nan, units="m")
        self.add_input("data:weight:aircraft_empty:CG:x", val=np.nan, units="m")
        self.add_input("data:weight:aircraft:MTOW", val=np.nan, units="kg")
        self.add_input("data:weight:aircraft_empty:mass", val=np.nan, units="kg")
        self.add_input("data:weight:propulsion:unusable_fuel:mass", val=np.nan, units="kg")
        self.add_input("data:weight:propulsion:tank:CG:x", val=np.nan, units="m")
        self.add_input("data:weight:aircraft:MFW", val=np.nan, units="kg")

        self.add_output("data:weight:aircraft:CG:flight_condition:max:MAC_position")
        self.add_output("data:weight:aircraft:CG:flight_condition:min:MAC_position")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        luggage_mass_max = float(inputs["data:geometry:cabin:luggage:mass_max"])
        n_pax_max = inputs["data:geometry:cabin:seats:passenger:NPAX_max"]
        l0_wing = inputs["data:geometry:wing:MAC:length"]
        fa_length = inputs["data:geometry:wing:MAC:at25percent:x"]
        lav = inputs["data:geometry:fuselage:front_length"]
        l_pax = inputs["data:geometry:fuselage:PAX_length"]
        l_pilot_seat = inputs["data:geometry:cabin:seats:pilot:length"]
        count_by_row = inputs["data:geometry:cabin:seats:passenger:count_by_row"]
        l_pass_seat = inputs["data:geometry:cabin:seats:passenger:length"]
        cg_rear_fret = inputs["data:weight:payload:rear_fret:CG:x"]
        x_cg_plane_aft = inputs["data:weight:aircraft_empty:CG:x"]
        m_empty = inputs["data:weight:aircraft_empty:mass"]
        m_unusable_fuel = inputs["data:weight:propulsion:unusable_fuel:mass"]
        cg_tank = inputs["data:weight:propulsion:tank:CG:x"]
        mfw = inputs["data:weight:aircraft:MFW"]

        l_instr = 0.7
        cg_pilot = lav + l_instr + l_pilot_seat / 2.0

        n_pax_array = np.linspace(0.0, n_pax_max, int(n_pax_max) + 1)

        m_pilot_single = 77.0
        m_pilot_array = np.array([2.0 * m_pilot_single])  # Without the pilots and with the 2 pilots

        m_fuel_min = m_unusable_fuel + self.min_in_flight_fuel(inputs)

        m_fuel_array = np.array([m_fuel_min, mfw])

        m_lug_array = np.array([0.0, luggage_mass_max])

        cg_list = []

        for m_pilot in m_pilot_array:

            for m_fuel in m_fuel_array:

                for m_lug in m_lug_array:

                    for n_pax in n_pax_array:

                        n_row = np.ceil(n_pax / count_by_row)

                        x_cg_pax_fwd = 0.0
                        for idx in range(int(n_row)):
                            row_cg = (idx + 0.5) * l_pass_seat
                            nb_pers = min(count_by_row, n_pax_max - idx * count_by_row)
                            x_cg_pax_fwd += row_cg * nb_pers / n_pax_max

                        x_cg_pax_aft = 0.0
                        for idx in range(int(n_row)):
                            row_cg = l_pax - l_pilot_seat - (idx + 0.5) * l_pass_seat
                            nb_pers = min(count_by_row, n_pax_max - idx * count_by_row)
                            x_cg_pax_aft += row_cg * nb_pers / n_pax_max

                        cg_pax_array = np.array(
                            [
                                lav + l_instr + l_pilot_seat + x_cg_pax_fwd,
                                lav + l_instr + l_pilot_seat + x_cg_pax_aft,
                            ]
                        )

                        for cg_pax in cg_pax_array:

                            m_pax_array = np.array([n_pax * 80.0, n_pax * 90.0])

                            for m_pax in m_pax_array:

                                mass = m_pax + m_pilot + m_fuel + m_lug + m_empty
                                cg = (
                                    m_empty * x_cg_plane_aft
                                    + m_pax * cg_pax
                                    + m_pilot * cg_pilot
                                    + m_fuel * cg_tank
                                    + m_lug * cg_rear_fret
                                ) / mass
                                cg_list.append(cg)

        cg_aft = max(cg_list)
        cg_fwd = min(cg_list)

        cg_fwd_ratio_pl = (cg_fwd - fa_length + 0.25 * l0_wing) / l0_wing
        cg_aft_ratio_pl = (cg_aft - fa_length + 0.25 * l0_wing) / l0_wing

        outputs["data:weight:aircraft:CG:flight_condition:max:MAC_position"] = cg_aft_ratio_pl
        outputs["data:weight:aircraft:CG:flight_condition:min:MAC_position"] = cg_fwd_ratio_pl

    def min_in_flight_fuel(self, inputs):

        propulsion_model = self._engine_wrapper.get_model(inputs)

        # noinspection PyTypeChecker
        mtow = inputs["data:weight:aircraft:MTOW"]

        vh = self.max_speed(inputs, 0.0, mtow)

        atm = Atmosphere(0.0, altitude_in_feet=False)
        flight_point = oad.FlightPoint(
            mach=vh / atm.speed_of_sound,
            altitude=0.0,
            engine_setting=EngineSetting.TAKEOFF,
            thrust_rate=1.0,
        )

        propulsion_model.compute_flight_points(flight_point)
        m_fuel = propulsion_model.get_consumed_mass(flight_point, 30.0 * 60.0)
        # Fuel necessary for a half-hour at max continuous power

        return m_fuel

    def max_speed(self, inputs, altitude, mass):

        # noinspection PyTypeChecker
        roots = optimize.fsolve(self.delta_axial_load, 300.0, args=(inputs, altitude, mass))[0]

        return np.max(roots[roots > 0.0])

    def delta_axial_load(self, air_speed, inputs, altitude, mass):

        propulsion_model = self._engine_wrapper.get_model(inputs)
        wing_area = inputs["data:geometry:wing:area"]
        cd0 = inputs["data:aerodynamics:aircraft:cruise:CD0"]
        coef_k = inputs["data:aerodynamics:wing:cruise:induced_drag_coefficient"]

        # Get the available thrust from propulsion system
        atm = Atmosphere(altitude, altitude_in_feet=False)
        flight_point = oad.FlightPoint(
            mach=air_speed / atm.speed_of_sound,
            altitude=altitude,
            engine_setting=EngineSetting.TAKEOFF,
            thrust_rate=1.0,
        )
        propulsion_model.compute_flight_points(flight_point)
        thrust = float(flight_point.thrust)

        # Get the necessary thrust to overcome
        cl = (mass * g) / (0.5 * atm.density * wing_area * air_speed ** 2.0)
        cd = cd0 + coef_k * cl ** 2.0
        drag = 0.5 * atm.density * wing_area * cd * air_speed ** 2.0

        return thrust - drag
