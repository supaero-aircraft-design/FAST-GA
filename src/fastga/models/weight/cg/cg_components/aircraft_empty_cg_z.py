"""
    Estimation of aircraft empty center of gravity.
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
import openmdao.api as om

import fastoad.api as oad

from ..cg_components.constants import SUBMODEL_AIRCRAFT_Z_CG


@oad.RegisterSubmodel(SUBMODEL_AIRCRAFT_Z_CG, "fastga.submodel.weight.cg.aircraft_empty.z.legacy")
class ComputeAircraftZCG(om.ExplicitComponent):
    """
    Computes Z-position of aircraft empty center of gravity.
    """

    def initialize(self):

        self.options.declare(
            "mass_names",
            [
                "data:weight:airframe:wing:mass",
                "data:weight:airframe:fuselage:mass",
                "data:weight:airframe:horizontal_tail:mass",
                "data:weight:airframe:vertical_tail:mass",
                "data:weight:airframe:flight_controls:mass",
                "data:weight:airframe:landing_gear:main:mass",
                "data:weight:airframe:landing_gear:front:mass",
                "data:weight:propulsion:engine:mass",
                "data:weight:propulsion:fuel_lines:mass",
                "data:weight:systems:power:electric_systems:mass",
                "data:weight:systems:power:hydraulic_systems:mass",
                "data:weight:systems:life_support:air_conditioning:mass",
                "data:weight:systems:avionics:mass",
                "data:weight:furniture:passenger_seats:mass",
            ],
        )

    def setup(self):

        for mass_name in self.options["mass_names"]:
            self.add_input(mass_name, val=np.nan, units="kg")

        self.add_input("data:geometry:fuselage:maximum_height", val=np.nan, units="m")
        self.add_input("data:weight:propulsion:engine:CG:z", val=np.nan, units="m")
        self.add_input("data:geometry:landing_gear:height", val=np.nan, units="m")
        self.add_input("data:geometry:horizontal_tail:z:from_wingMAC25", val=np.nan, units="m")
        self.add_input("data:geometry:vertical_tail:span", val=np.nan, units="m")
        self.add_input("data:geometry:wing:MAC:length", val=np.nan, units="m")
        self.add_input("data:geometry:wing:thickness_ratio", val=np.nan)

        self.add_output("data:weight:aircraft_empty:CG:z", units="m")

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        masses = [inputs[mass_name][0] for mass_name in self.options["mass_names"]]
        height_max = inputs["data:geometry:fuselage:maximum_height"][0]
        cg_engine = inputs["data:weight:propulsion:engine:CG:z"][0]
        lg_height = inputs["data:geometry:landing_gear:height"][0]
        ht_height = inputs["data:geometry:horizontal_tail:z:from_wingMAC25"][0]
        vt_span = inputs["data:geometry:vertical_tail:span"][0]
        l0_wing = inputs["data:geometry:wing:MAC:length"][0]
        thickness_ratio = inputs["data:geometry:wing:thickness_ratio"][0]

        # TODO : For now we assume low wings only, change later
        cg_wing = lg_height + thickness_ratio * l0_wing / 2.0
        cg_fuselage = lg_height + height_max / 2.0
        cg_horizontal_tail = cg_wing + ht_height
        cg_vertical_tail = cg_fuselage + vt_span / 2.0
        # TODO : To be changed depending we want or not the case where LG are retractable
        cg_landing_gear = lg_height / 2.0
        # CS 23 gives a minimum ground clearance of 18 cm for nose wheel landing gear, but TB20,
        # SR22, BE76 all use a 23 cm clearance as recommended for tail wheel landing gear
        cg_fuel_lines = (cg_engine + cg_wing) / 2.0
        cgs = np.array(
            [
                cg_wing,
                cg_fuselage,
                cg_horizontal_tail,
                cg_vertical_tail,
                cg_fuselage,
                cg_landing_gear,
                cg_landing_gear,
                cg_engine,
                cg_fuel_lines,
                cg_fuselage,
                cg_fuselage,
                cg_fuselage,
                cg_fuselage,
                cg_fuselage,
            ]
        )

        weight_moment = np.dot(cgs, masses)
        z_cg_empty_aircraft = weight_moment / np.sum(masses)

        outputs["data:weight:aircraft_empty:CG:z"] = z_cg_empty_aircraft

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        height_max = inputs["data:geometry:fuselage:maximum_height"][0]
        cg_engine = inputs["data:weight:propulsion:engine:CG:z"][0]
        lg_height = inputs["data:geometry:landing_gear:height"][0]
        ht_height = inputs["data:geometry:horizontal_tail:z:from_wingMAC25"][0]
        vt_span = inputs["data:geometry:vertical_tail:span"][0]
        l0_wing = inputs["data:geometry:wing:MAC:length"][0]
        thickness_ratio = inputs["data:geometry:wing:thickness_ratio"][0]

        masses = [inputs[mass_name][0] for mass_name in self.options["mass_names"]]
        total_mass = np.sum(masses)

        # TODO : For now we assume low wings only, change later
        cg_wing = lg_height + thickness_ratio * l0_wing / 2.0
        cg_fuselage = lg_height + height_max / 2.0
        cg_horizontal_tail = cg_wing + ht_height
        cg_vertical_tail = cg_fuselage + vt_span / 2.0
        # TODO : To be changed depending we want or not the case where LG are retractable
        cg_landing_gear = lg_height / 2.0
        # CS 23 gives a minimum ground clearance of 18 cm for nose wheel landing gear, but TB20,
        # SR22, BE76 all use a 23 cm clearance as recommended for tail wheel landing gear
        cg_fuel_lines = (cg_engine + cg_wing) / 2.0
        cgs = np.array(
            [
                cg_wing,
                cg_fuselage,
                cg_horizontal_tail,
                cg_vertical_tail,
                cg_fuselage,
                cg_landing_gear,
                cg_landing_gear,
                cg_engine,
                cg_fuel_lines,
                cg_fuselage,
                cg_fuselage,
                cg_fuselage,
                cg_fuselage,
                cg_fuselage,
            ]
        )

        for cg, mass_name in zip(cgs, self.options["mass_names"]):
            partials["data:weight:aircraft_empty:CG:z", mass_name] = (
                cg * total_mass - np.dot(cgs, masses)
            ) / total_mass ** 2.0

        mass_inputs_for_cg_fuselage = np.array(
            [
                inputs["data:weight:airframe:fuselage:mass"],
                inputs["data:weight:airframe:flight_controls:mass"],
                inputs["data:weight:systems:power:electric_systems:mass"],
                inputs["data:weight:airframe:vertical_tail:mass"],
                inputs["data:weight:systems:power:hydraulic_systems:mass"],
                inputs["data:weight:systems:life_support:air_conditioning:mass"],
                inputs["data:weight:systems:avionics:mass"],
                inputs["data:weight:furniture:passenger_seats:mass"],
            ]
        )
        d_z_cg_d_cg_fuselage = np.sum(mass_inputs_for_cg_fuselage) / total_mass
        mass_inputs_for_cg_engine = np.array(
            [
                inputs["data:weight:propulsion:engine:mass"],
                inputs["data:weight:propulsion:fuel_lines:mass"] / 2.0,
            ]
        )
        d_z_cg_d_cg_engine = np.sum(mass_inputs_for_cg_engine) / total_mass
        mass_inputs_for_cg_wing = np.array(
            [
                inputs["data:weight:airframe:wing:mass"],
                inputs["data:weight:airframe:horizontal_tail:mass"],
                inputs["data:weight:propulsion:fuel_lines:mass"] / 2.0,
            ]
        )
        d_z_cg_d_cg_wing = np.sum(mass_inputs_for_cg_wing) / total_mass
        mass_inputs_for_cg_lg = np.array(
            [
                inputs["data:weight:airframe:landing_gear:main:mass"],
                inputs["data:weight:airframe:landing_gear:front:mass"],
            ]
        )
        d_z_cg_d_cg_lg = np.sum(mass_inputs_for_cg_lg) / total_mass

        partials["data:weight:aircraft_empty:CG:z", "data:geometry:fuselage:maximum_height"] = (
            d_z_cg_d_cg_fuselage * 0.5
        )
        partials[
            "data:weight:aircraft_empty:CG:z", "data:weight:propulsion:engine:CG:z"
        ] = d_z_cg_d_cg_engine
        partials["data:weight:aircraft_empty:CG:z", "data:geometry:landing_gear:height"] = (
            d_z_cg_d_cg_wing + d_z_cg_d_cg_fuselage + d_z_cg_d_cg_lg / 2.0
        )
        partials[
            "data:weight:aircraft_empty:CG:z", "data:geometry:horizontal_tail:z:from_wingMAC25"
        ] = (inputs["data:weight:airframe:horizontal_tail:mass"] / total_mass)
        partials["data:weight:aircraft_empty:CG:z", "data:geometry:vertical_tail:span"] = inputs[
            "data:weight:airframe:vertical_tail:mass"
        ] / (2.0 * total_mass)
        partials["data:weight:aircraft_empty:CG:z", "data:geometry:wing:MAC:length"] = (
            d_z_cg_d_cg_wing * thickness_ratio / 2.0
        )
        partials["data:weight:aircraft_empty:CG:z", "data:geometry:wing:thickness_ratio"] = (
            d_z_cg_d_cg_wing * l0_wing / 2.0
        )
