"""Computes the mass of the tail cone, adapted from TASOPT by Lucas REMOND."""
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

import openmdao.api as om

import numpy as np

from stdatm import Atmosphere


class ComputeTailCone(om.ExplicitComponent):
    def setup(self):

        self.add_input("data:geometry:vertical_tail:area", val=np.nan, units="m**2")
        self.add_input("data:geometry:vertical_tail:span", val=np.nan, units="m")
        self.add_input("data:geometry:vertical_tail:taper_ratio", val=np.nan)
        self.add_input("data:geometry:vertical_tail:rudder:max_deflection", val=np.nan, units="rad")
        self.add_input("data:geometry:fuselage:rear_length", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:maximum_width", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:maximum_height", val=np.nan, units="m")
        self.add_input("data:aerodynamics:rudder:cruise:Cy_delta_r", val=np.nan, units="rad**-1")
        self.add_input("data:weight:airframe:fuselage:shell:added_weight_ratio", val=np.nan)
        self.add_input("data:weight:airframe:horizontal_tail:mass", val=np.nan, units="kg")
        self.add_input("data:weight:airframe:vertical_tail:mass", val=np.nan, units="kg")
        self.add_input("data:mission:sizing:main_route:cruise:altitude", val=np.nan, units="m")
        self.add_input("data:mission:sizing:cs23:characteristic_speed:vd", val=np.nan, units="m/s")

        self.add_input("settings:geometry:fuselage:cone:taper_ratio", val=0.2)
        self.add_input("settings:materials:fuselage:skin:density", val=np.nan, units="kg/m**3")
        self.add_input("settings:materials:fuselage:skin:max_shear_stress", val=np.nan, units="Pa")

        self.add_output("data:weight:airframe:fuselage:cone:mass", units="kg")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        vtp_area = inputs["data:geometry:vertical_tail:area"]
        vtp_span = inputs["data:geometry:vertical_tail:span"]
        vtp_taper_ratio = inputs["data:geometry:vertical_tail:taper_ratio"]
        delta_r_max = inputs["data:geometry:vertical_tail:rudder:max_deflection"]
        cy_delta_r = inputs["data:aerodynamics:rudder:cruise:Cy_delta_r"]
        added_mass_ratio = inputs["data:weight:airframe:fuselage:shell:added_weight_ratio"]
        cruise_altitude = inputs["data:mission:sizing:main_route:cruise:altitude"]
        vd = inputs["data:mission:sizing:cs23:characteristic_speed:vd"]
        # We assume that the distance between the end of the cylindrical shell and the end of the
        # cone is equal to lar
        lar = inputs["data:geometry:fuselage:rear_length"]
        fuselage_max_width = inputs["data:geometry:fuselage:maximum_width"]
        fuselage_max_height = inputs["data:geometry:fuselage:maximum_height"]
        vtp_mass = inputs["data:weight:airframe:vertical_tail:mass"]
        htp_mass = inputs["data:weight:airframe:horizontal_tail:mass"]

        tau_cone = inputs["settings:materials:fuselage:skin:max_shear_stress"]
        cone_taper_ratio = inputs["settings:geometry:fuselage:cone:taper_ratio"]
        rho_skin = inputs["settings:materials:fuselage:skin:density"]

        fuselage_radius = np.sqrt(fuselage_max_height * fuselage_max_width) / 2.0

        density = Atmosphere(cruise_altitude, altitude_in_feet=False).density
        cl_v_max = cy_delta_r * delta_r_max
        q_ne = 0.5 * density * vd ** 2
        max_lift_vtp = q_ne * cl_v_max * vtp_area

        torsion_moment_vtp = (
            max_lift_vtp * vtp_span / 3.0 * (1.0 + 2.0 * vtp_taper_ratio) / (1.0 + vtp_taper_ratio)
        )
        volume_cone = (
            torsion_moment_vtp / tau_cone * lar / fuselage_radius * 2.0 / (1.0 + cone_taper_ratio)
        )
        cone_mass_torsion = rho_skin * volume_cone * added_mass_ratio

        cone_mass_tail_support = 0.10 * (vtp_mass + htp_mass)

        outputs["data:weight:airframe:fuselage:cone:mass"] = (
            cone_mass_torsion + cone_mass_tail_support
        )
