"""
    Computation of the non-equilibrated aircraft polars
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
from stdatm import Atmosphere

from fastga.models.aerodynamics.constants import POLAR_POINT_COUNT, FIRST_INVALID_COEFF
from fastga.models.performances.mission.dynamic_equilibrium import DynamicEquilibrium


class ComputeEquilibratedPolar(DynamicEquilibrium):
    def initialize(self):
        self.options.declare("low_speed_aero", default=False, types=bool)
        self.options.declare("cg_ratio", default=-0.0, types=float)

    def setup(self):
        self.add_input("data:geometry:wing:MAC:leading_edge:x:local", val=np.nan, units="m")
        self.add_input("data:geometry:wing:MAC:length", np.nan, units="m")
        self.add_input("data:geometry:wing:root:virtual_chord", np.nan, units="m")
        self.add_input("data:geometry:fuselage:maximum_width", np.nan, units="m")
        self.add_input("data:geometry:wing:MAC:at25percent:x", np.nan, units="m")
        self.add_input("data:geometry:fuselage:length", np.nan, units="m")
        self.add_input("data:geometry:wing:area", np.nan, units="m**2")
        self.add_input(
            "data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25", np.nan, units="m"
        )
        self.add_input("data:aerodynamics:wing:low_speed:CL_max_clean", val=np.nan)
        self.add_input("data:weight:aircraft:CG:aft:x", np.nan, units="m")
        self.add_input("data:weight:aircraft:CG:fwd:x", np.nan, units="m")
        self.add_input("data:weight:aircraft_empty:CG:z", val=np.nan, units="m")
        self.add_input("data:weight:propulsion:engine:CG:z", val=np.nan, units="m")
        self.add_input(
            "data:weight:aircraft:in_flight_variation:fixed_mass_comp:equivalent_moment",
            np.nan,
            units="kg*m",
        )
        self.add_input("data:weight:propulsion:tank:CG:x", np.nan, units="m")
        self.add_input(
            "data:weight:aircraft:in_flight_variation:fixed_mass_comp:mass", np.nan, units="kg"
        )
        self.add_input("data:weight:aircraft:MTOW", np.nan, units="kg")
        self.add_input("data:aerodynamics:horizontal_tail:cruise:CL0", val=np.nan)
        self.add_input(
            "data:aerodynamics:horizontal_tail:cruise:CL_alpha", val=np.nan, units="rad**-1"
        )
        self.add_input(
            "data:aerodynamics:horizontal_tail:cruise:CL_alpha_isolated",
            val=np.nan,
            units="rad**-1",
        )
        self.add_input("data:aerodynamics:horizontal_tail:low_speed:CL0", val=np.nan)
        self.add_input(
            "data:aerodynamics:horizontal_tail:low_speed:CL_alpha", val=np.nan, units="rad**-1"
        )
        self.add_input(
            "data:aerodynamics:horizontal_tail:low_speed:CL_alpha_isolated",
            val=np.nan,
            units="rad**-1",
        )
        self.add_input("data:aerodynamics:horizontal_tail:low_speed:CL_max_clean", val=np.nan)
        self.add_input("data:aerodynamics:horizontal_tail:low_speed:CL_min_clean", val=np.nan)
        self.add_input("data:aerodynamics:elevator:low_speed:CL_delta", val=np.nan, units="rad**-1")
        self.add_input("data:aerodynamics:elevator:low_speed:CD_delta", val=np.nan, units="rad**-2")
        self.add_input("data:aerodynamics:fuselage:cm_alpha", val=np.nan, units="rad**-1")

        if self.options["low_speed_aero"]:
            self.add_input("data:aerodynamics:low_speed:mach", val=np.nan)
            self.add_input("data:aerodynamics:wing:low_speed:CL0_clean", val=np.nan)
            self.add_input("data:aerodynamics:wing:low_speed:induced_drag_coefficient", val=np.nan)
            self.add_input("data:aerodynamics:wing:low_speed:CL_alpha", val=np.nan, units="rad**-1")
            self.add_input("data:aerodynamics:aircraft:low_speed:CD0", val=np.nan)
            self.add_input("data:aerodynamics:wing:low_speed:CM0_clean", val=np.nan)
            self.add_input(
                "data:aerodynamics:horizontal_tail:low_speed:induced_drag_coefficient", val=np.nan
            )
            self.add_input("data:TLAR:v_approach", np.nan, units="m/s")

            self.add_output(
                "data:aerodynamics:aircraft:low_speed:equilibrated:CD",
                shape=POLAR_POINT_COUNT,
            )
            self.add_output(
                "data:aerodynamics:aircraft:low_speed:equilibrated:CL",
                shape=POLAR_POINT_COUNT,
            )

        else:
            self.add_input("data:aerodynamics:cruise:mach", val=np.nan)
            self.add_input("data:aerodynamics:wing:cruise:CL0_clean", val=np.nan)
            self.add_input("data:aerodynamics:wing:cruise:induced_drag_coefficient", val=np.nan)
            self.add_input("data:aerodynamics:wing:cruise:CL_alpha", val=np.nan, units="rad**-1")
            self.add_input("data:aerodynamics:aircraft:cruise:CD0", val=np.nan)
            self.add_input("data:aerodynamics:wing:cruise:CM0_clean", val=np.nan)
            self.add_input(
                "data:aerodynamics:horizontal_tail:cruise:induced_drag_coefficient", val=np.nan
            )
            self.add_input("data:mission:sizing:main_route:cruise:altitude", np.nan, units="m")
            self.add_input("data:TLAR:v_cruise", np.nan, units="m/s")

            self.add_output(
                "data:aerodynamics:aircraft:cruise:equilibrated:CD", shape=POLAR_POINT_COUNT
            )
            self.add_output(
                "data:aerodynamics:aircraft:cruise:equilibrated:CL", shape=POLAR_POINT_COUNT
            )

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        mtow = inputs["data:weight:aircraft:MTOW"]
        wing_area = inputs["data:geometry:wing:area"]
        cl_max_clean = inputs["data:aerodynamics:wing:low_speed:CL_max_clean"]
        x_cg_fwd = inputs["data:weight:aircraft:CG:fwd:x"]
        x_cg_aft = inputs["data:weight:aircraft:CG:aft:x"]

        if self.options["low_speed_aero"]:
            altitude = 0
            v_tas = inputs["data:TLAR:v_approach"]
        else:
            altitude = inputs["data:mission:sizing:main_route:cruise:altitude"]
            v_tas = inputs["data:TLAR:v_cruise"]

        cl_array = np.array([])
        cd_array = np.array([])

        atm = Atmosphere(altitude, altitude_in_feet=False)

        # Computation of the maximum aircraft mass that can be used before exceeding
        # the CL0 clean of the wing in found_cl_repartition
        init_mass_guess = (0.5 * atm.density * v_tas ** 2 * wing_area * cl_max_clean) / 9.81

        cg_ratio = self.options["cg_ratio"]
        x_cg = x_cg_aft + cg_ratio * (x_cg_fwd - x_cg_aft)

        mass_array = np.linspace(0.1 * mtow, 1.15 * init_mass_guess, POLAR_POINT_COUNT)
        # previous_step = ()
        for mass in mass_array:
            previous_step = self.dynamic_equilibrium(
                inputs,
                0.0,
                (0.5 * atm.density * v_tas ** 2),
                0.0,
                0.0,
                mass,
                "none",
                (),
                low_speed=self.options["low_speed_aero"],
                x_cg=x_cg,
            )
            if previous_step[-1]:
                break
            else:
                cl_wing = float(previous_step[2])
                cl_tail = float(previous_step[3])
                thrust = float(previous_step[1])
                cl_array = np.append(cl_array, cl_wing + cl_tail)
                cd = thrust / (0.5 * atm.density * v_tas ** 2 * wing_area)
                cd_array = np.append(cd_array, cd)

        additional_zeros = np.linspace(
            FIRST_INVALID_COEFF, 2 * FIRST_INVALID_COEFF, POLAR_POINT_COUNT - len(cd_array)
        )
        cd_array = np.append(cd_array, additional_zeros)
        cl_array = np.append(cl_array, additional_zeros)

        if self.options["low_speed_aero"]:
            outputs["data:aerodynamics:aircraft:low_speed:equilibrated:CD"] = cd_array
            outputs["data:aerodynamics:aircraft:low_speed:equilibrated:CL"] = cl_array
        else:
            outputs["data:aerodynamics:aircraft:cruise:equilibrated:CD"] = cd_array
            outputs["data:aerodynamics:aircraft:cruise:equilibrated:CL"] = cl_array
