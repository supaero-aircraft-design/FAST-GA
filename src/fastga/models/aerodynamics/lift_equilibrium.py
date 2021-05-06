"""
    FAST - Copyright (c) 2016 ONERA ISAE
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
from scipy.constants import g


class AircraftEquilibrium(om.ExplicitComponent):
    """
    Compute the mass-lift equilibrium
    """

    def setup(self):
        self.add_input("data:geometry:wing:MAC:leading_edge:x:local", np.nan, units="m")
        self.add_input("data:geometry:wing:MAC:length", np.nan, units="m")
        self.add_input("data:geometry:wing:root:virtual_chord", np.nan, units="m")
        self.add_input("data:geometry:fuselage:maximum_width", np.nan, units="m")
        self.add_input("data:geometry:wing:MAC:at25percent:x", np.nan, units="m")
        self.add_input("data:geometry:fuselage:length", np.nan, units="m")
        self.add_input("data:geometry:wing:area", np.nan, units="m**2")
        self.add_input("data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25", np.nan, units="m")
        self.add_input("data:aerodynamics:wing:cruise:CL_alpha", np.nan, units="rad**-1")
        self.add_input("data:aerodynamics:wing:cruise:CL0_clean", np.nan)
        self.add_input("data:aerodynamics:wing:cruise:CM0_clean", np.nan)
        self.add_input("data:aerodynamics:horizontal_tail:cruise:CL_alpha", np.nan, units="rad**-1")
        self.add_input("data:aerodynamics:wing:low_speed:CL_alpha", np.nan, units="rad**-1")
        self.add_input("data:aerodynamics:wing:low_speed:CL0_clean", np.nan)
        self.add_input("data:aerodynamics:wing:low_speed:CM0_clean", np.nan)
        self.add_input("data:aerodynamics:wing:low_speed:CL_max_clean", np.nan)
        self.add_input("data:aerodynamics:horizontal_tail:low_speed:CL_alpha", np.nan, units="rad**-1")
        self.add_input("data:weight:aircraft:CG:aft:x", np.nan, units="m")
        self.add_input("data:weight:aircraft:in_flight_variation:fixed_mass_comp:equivalent_moment", np.nan,
                       units="kg*m")
        self.add_input("data:weight:propulsion:tank:CG:x", np.nan, units="m")
        self.add_input("data:weight:aircraft:in_flight_variation:fixed_mass_comp:mass", np.nan, units="kg")


    @staticmethod
    def found_cl_repartition(inputs, load_factor, mass, dynamic_pressure, low_speed, x_cg=-1.0):

        x0_wing = inputs["data:geometry:wing:MAC:leading_edge:x:local"]
        l0_wing = inputs["data:geometry:wing:MAC:length"]
        l1_wing = inputs["data:geometry:wing:root:virtual_chord"]
        width_max = inputs["data:geometry:fuselage:maximum_width"]
        x_wing = inputs["data:geometry:wing:MAC:at25percent:x"]
        fus_length = inputs["data:geometry:fuselage:length"]
        wing_area = inputs["data:geometry:wing:area"]
        x_htp = x_wing + inputs["data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25"]
        if low_speed:
            cl_alpha_wing = inputs["data:aerodynamics:wing:low_speed:CL_alpha"]
            cl_alpha_htp = inputs["data:aerodynamics:horizontal_tail:low_speed:CL_alpha"]
            cl0_wing = inputs["data:aerodynamics:wing:low_speed:CL0_clean"]
            cm0_wing = inputs["data:aerodynamics:wing:low_speed:CM0_clean"]
        else:
            cl_alpha_wing = inputs["data:aerodynamics:wing:cruise:CL_alpha"]
            cl_alpha_htp = inputs["data:aerodynamics:horizontal_tail:cruise:CL_alpha"]
            cl0_wing = inputs["data:aerodynamics:wing:cruise:CL0_clean"]
            cm0_wing = inputs["data:aerodynamics:wing:cruise:CM0_clean"]
        cl_max_clean = inputs["data:aerodynamics:wing:low_speed:CL_max_clean"]

        if x_cg < 0.0:
            c1 = inputs["data:weight:aircraft:in_flight_variation:fixed_mass_comp:equivalent_moment"]
            cg_tank = inputs["data:weight:propulsion:tank:CG:x"]
            c3 = inputs["data:weight:aircraft:in_flight_variation:fixed_mass_comp:mass"]
            fuel_mass = mass - c3
            x_cg = (c1 + cg_tank * fuel_mass) / (c3 + fuel_mass)

        # Calculate cm_alpha_fus from Raymer equations (figure 16.14, eqn 16.22)
        x0_25 = x_wing - 0.25 * l0_wing - x0_wing + 0.25 * l1_wing
        ratio_x025 = x0_25 / fus_length
        k_h = 0.01222 - 7.40541e-4 * ratio_x025 * 100 + 2.1956e-5 * (ratio_x025 * 100) ** 2
        cm_alpha_fus = - k_h * width_max ** 2 * fus_length / (l0_wing * wing_area) * 180.0 / np.pi

        # Define matrix equilibrium (applying load and moment equilibrium)
        a11 = 1
        a12 = 1
        b1 = mass * g * load_factor / (dynamic_pressure * wing_area)
        a21 = (x_wing - x_cg) - (cm_alpha_fus / cl_alpha_wing) * l0_wing
        a22 = (x_htp - x_cg)
        b2 = (cm0_wing + (cm_alpha_fus / cl_alpha_wing) * cl0_wing) * l0_wing

        a = np.array([[a11, a12], [float(a21), float(a22)]])
        b = np.array([b1, b2])
        inv_a = np.linalg.inv(a)
        CL = np.dot(inv_a, b)

        CL_wing = CL[0]
        CL_tail = CL[1]

        alpha_avion = (CL_wing - cl0_wing) / cl_alpha_wing
        cl_htp_only = alpha_avion * cl_alpha_htp
        cl_elevator = CL_tail - cl_htp_only

        # Return equilibrated lift coefficients if low speed maximum clean Cl not exceeded otherwise only cl_wing, 3rd
        # term is an error flag returned by the function
        if CL[0] < cl_max_clean:
            return float(CL_wing), float(cl_htp_only), cl_elevator, False
        else:
            return 1.05*float(mass * g * load_factor / (dynamic_pressure * wing_area)), 0.0, 0.0, True
