"""
    Estimation of center of gravity ratio with aft
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


class ComputeCGRatioAft(om.Group):
    def setup(self):
        self.add_subsystem("cg_all", ComputeCG(), promotes=["*"])
        self.add_subsystem("z_cg", ComputeZCG(), promotes=["*"])
        self.add_subsystem("cg_ratio", CGRatio(), promotes=["*"])


class ComputeCG(om.ExplicitComponent):
    def initialize(self):
        self.options.declare(
            "cg_names",
            default=[
                "data:weight:airframe:wing:CG:x",
                "data:weight:airframe:fuselage:CG:x",
                "data:weight:airframe:horizontal_tail:CG:x",
                "data:weight:airframe:vertical_tail:CG:x",
                "data:weight:airframe:flight_controls:CG:x",
                "data:weight:airframe:landing_gear:main:CG:x",
                "data:weight:airframe:landing_gear:front:CG:x",
                "data:weight:propulsion:engine:CG:x",
                "data:weight:propulsion:fuel_lines:CG:x",
                "data:weight:systems:power:electric_systems:CG:x",
                "data:weight:systems:power:hydraulic_systems:CG:x",
                "data:weight:systems:life_support:air_conditioning:CG:x",
                "data:weight:systems:navigation:CG:x",
                "data:weight:furniture:passenger_seats:CG:x",
            ],
        )

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
                "data:weight:systems:navigation:mass",
                "data:weight:furniture:passenger_seats:mass",
            ],
        )

    def setup(self):
        for cg_name in self.options["cg_names"]:
            self.add_input(cg_name, val=np.nan, units="m")
        for mass_name in self.options["mass_names"]:
            self.add_input(mass_name, val=np.nan, units="kg")

        self.add_output("data:weight:aircraft_empty:mass", units="kg")
        self.add_output("data:weight:aircraft_empty:CG:x", units="m")

        self.declare_partials("data:weight:aircraft_empty:mass", "*", method="fd")
        self.declare_partials("data:weight:aircraft_empty:CG:x", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        cgs = [inputs[cg_name][0] for cg_name in self.options["cg_names"]]
        masses = [inputs[mass_name][0] for mass_name in self.options["mass_names"]]

        weight_moment = np.dot(cgs, masses)
        test = []
        for i in range(len(cgs)):
            test.append(cgs[i]*masses[i]/np.sum(masses))
        outputs["data:weight:aircraft_empty:mass"] = np.sum(masses)
        x_cg_empty_aircraft = weight_moment / outputs["data:weight:aircraft_empty:mass"]
        outputs["data:weight:aircraft_empty:CG:x"] = x_cg_empty_aircraft


class CGRatio(om.ExplicitComponent):
    def setup(self):
        self.add_input("data:weight:aircraft_empty:CG:x", val=np.nan, units="m")
        self.add_input("data:geometry:wing:MAC:length", val=np.nan, units="m")
        self.add_input("data:geometry:wing:MAC:at25percent:x", val=np.nan, units="m")

        self.add_output("data:weight:aircraft:empty:CG:MAC_position")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        x_cg_all = inputs["data:weight:aircraft_empty:CG:x"]
        wing_position = inputs["data:geometry:wing:MAC:at25percent:x"]
        mac = inputs["data:geometry:wing:MAC:length"]

        outputs["data:weight:aircraft:empty:CG:MAC_position"] = (
                                                                        x_cg_all - wing_position + 0.25 * mac
                                                                ) / mac


class ComputeZCG(om.ExplicitComponent):

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
                "data:weight:systems:navigation:mass",
                "data:weight:furniture:passenger_seats:mass",
            ],
        )

    def setup(self):
        for mass_name in self.options["mass_names"]:
            self.add_input(mass_name, val=np.nan, units="kg")

        self.add_input("data:geometry:fuselage:maximum_height", val=np.nan, units="m")
        self.add_input("data:geometry:propulsion:propeller:diameter", val=np.nan, units="m")
        self.add_input("data:geometry:landing_gear:height", val=np.nan, units="m")
        self.add_input("data:geometry:horizontal_tail:z:from_wingMAC25", val=np.nan, units="m")
        self.add_input("data:geometry:vertical_tail:span", val=np.nan, units="m")
        self.add_input("data:geometry:wing:MAC:length", val=np.nan, units="m")
        self.add_input("data:geometry:wing:thickness_ratio", val=np.nan)

        self.add_output("data:weight:aircraft_empty:CG:z", units="m")
        self.add_output("data:weight:propulsion:engine:CG:z", units="m")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        masses = [inputs[mass_name][0] for mass_name in self.options["mass_names"]]
        height_max = inputs["data:geometry:fuselage:maximum_height"][0]
        prop_dia = inputs["data:geometry:propulsion:propeller:diameter"][0]
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
        # CS 23 gives a minimum ground clearance of 18 cm for nose wheel landing gear, but TB20, SR22, BE76 all use a
        # 23 cm clearance as recommended for tail wheel landing gear
        cg_engine = 0.23 + prop_dia / 2.0
        cg_fuel_lines = (cg_engine + cg_wing) / 2.0
        cgs = np.array([cg_wing, cg_fuselage, cg_horizontal_tail, cg_vertical_tail, cg_fuselage, cg_landing_gear,
                        cg_landing_gear, cg_engine, cg_fuel_lines, cg_fuselage, cg_fuselage, cg_fuselage, cg_fuselage,
                        cg_fuselage])

        weight_moment = np.dot(cgs, masses)
        z_cg_empty_aircraft = weight_moment / np.sum(masses)

        outputs["data:weight:aircraft_empty:CG:z"] = z_cg_empty_aircraft
        outputs["data:weight:propulsion:engine:CG:z"] = cg_engine
