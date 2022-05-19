"""Computes the additional horizontal bending mass, adapted from TASOPT by Lucas REMOND."""
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

FUSELAGE_MESH_POINT = 100


class ComputeAddBendingMassVertical(om.ExplicitComponent):
    def setup(self):
        self.add_input("data:geometry:vertical_tail:area", val=np.nan, units="m**2")
        self.add_input("data:geometry:vertical_tail:rudder:max_deflection", val=np.nan, units="rad")
        self.add_input("data:geometry:fuselage:front_length", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:rear_length", val=np.nan, units="m")
        self.add_input("data:geometry:cabin:length", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:maximum_width", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:maximum_height", val=np.nan, units="m")

        self.add_input("data:aerodynamics:rudder:cruise:Cy_delta_r", val=np.nan, units="rad**-1")

        self.add_input("data:weight:airframe:wing:CG:x", val=np.nan, units="m")
        self.add_input("data:weight:airframe:horizontal_tail:CG:x", val=np.nan, units="m")
        self.add_input("data:weight:airframe:vertical_tail:CG:x", val=np.nan, units="m")
        self.add_input("data:weight:airframe:horizontal_tail:mass", val=np.nan, units="kg")
        self.add_input("data:weight:airframe:vertical_tail:mass", val=np.nan, units="kg")
        self.add_input("data:weight:airframe:fuselage:cone:mass", val=np.nan, units="kg")

        self.add_input("data:weight:furniture:passenger_seats:mass", val=np.nan, units="lb")

        self.add_input("data:loads:fuselage:inertia", val=np.nan, units="m**4")
        self.add_input("data:loads:fuselage:sigmaMh", val=np.nan, units="N/m**2")

        self.add_input("data:mission:sizing:main_route:cruise:altitude", val=np.nan, units="m")
        self.add_input("data:mission:sizing:cs23:characteristic_speed:vd", val=np.nan, units="m/s")

        self.add_input("settings:materials:fuselage:skin:density", val=np.nan, units="kg/m**3")
        self.add_input("settings:materials:fuselage:skin:young_modulus", val=np.nan, units="Pa")
        self.add_input("settings:materials:fuselage:stringer:young_modulus", val=np.nan, units="Pa")

        self.add_output("data:weight:airframe:fuselage:additional_mass:vertical", units="kg")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        vtp_area = inputs["data:geometry:vertical_tail:area"]
        lav = inputs["data:geometry:fuselage:front_length"]
        lar = inputs["data:geometry:fuselage:rear_length"]
        cabin_length = inputs["data:geometry:cabin:length"]
        fuselage_max_width = inputs["data:geometry:fuselage:maximum_width"]
        fuselage_max_height = inputs["data:geometry:fuselage:maximum_height"]

        delta_r_max = inputs["data:geometry:vertical_tail:rudder:max_deflection"]
        cy_delta_r = inputs["data:aerodynamics:rudder:cruise:Cy_delta_r"]

        wing_centroid = inputs["data:weight:airframe:wing:CG:x"]
        htp_cg = inputs["data:weight:airframe:horizontal_tail:CG:x"]
        vtp_cg = inputs["data:weight:airframe:vertical_tail:CG:x"]
        htp_mass = inputs["data:weight:airframe:horizontal_tail:mass"]
        vtp_mass = inputs["data:weight:airframe:vertical_tail:mass"]
        cone_mass = inputs["data:weight:airframe:fuselage:cone:mass"]

        sigma_mh = inputs["data:loads:fuselage:sigmaMh"]
        bending_inertia = inputs["data:loads:fuselage:inertia"]

        cruise_altitude = inputs["data:mission:sizing:main_route:cruise:altitude"]

        vd = inputs["data:mission:sizing:cs23:characteristic_speed:vd"]

        rho_skin = inputs["settings:materials:fuselage:skin:density"]
        e_skin = inputs["settings:materials:fuselage:skin:young_modulus"]
        e_stringer = inputs["settings:materials:fuselage:stringer:young_modulus"]

        fuselage_radius = np.sqrt(fuselage_max_height * fuselage_max_width) / 2.0

        cl_max_vtp = cy_delta_r * delta_r_max

        density = Atmosphere(cruise_altitude, altitude_in_feet=False).density

        q_never_exceed = 0.5 * density * vd ** 2
        lift_max_v = q_never_exceed * vtp_area * cl_max_vtp
        # Inertia relief factor, typical value according to TASOPT
        rmv = 0.7

        # We will assume that the cone mass is evenly distributed, thus its center of gravity
        # will coincide with the centroid of the triangle whose base is the end of the
        # cylindrical part of the fuselage and tip is the end of the cone, hence with the CoG is
        # at 1/3 of the rear length of the fuselage
        tail_x_cg = (
            htp_cg * htp_mass + vtp_cg * vtp_mass + (lav + cabin_length + 0.333 * lar) * cone_mass
        ) / (htp_mass + vtp_mass + cone_mass)

        # Ratio of the cabin before and after the wing centroid point

        # The fuselage length is roughly discretized with a fixed length step.
        nb_points_front = int(FUSELAGE_MESH_POINT * wing_centroid / tail_x_cg)
        x_vector_rear = np.linspace(wing_centroid, tail_x_cg, FUSELAGE_MESH_POINT - nb_points_front)

        vertical_bending_vector = np.zeros_like(x_vector_rear)
        for x in x_vector_rear:
            bending = (rmv * lift_max_v) * (tail_x_cg - x)
            vertical_bending_vector = np.append(vertical_bending_vector, bending)
        # Calculation of x_v_bend
        index = np.where(
            vertical_bending_vector * fuselage_radius / sigma_mh
            - bending_inertia * e_skin / e_stringer
            < 0
        )[0]

        if index.size == 0:
            volume = 0.0
        else:
            x_v_bend = x_vector_rear[min(index)]
            # Integration of the bending area
            b1 = rmv * lift_max_v / (fuselage_radius * sigma_mh)
            b0 = -bending_inertia / (e_stringer / e_skin * fuselage_radius ** 2)
            volume = b1 * (
                (tail_x_cg - wing_centroid) ** 2 - (tail_x_cg - x_v_bend) ** 2
            ) / 2 + b0 * (x_v_bend - wing_centroid)
        additional_mass_vertical = volume * rho_skin

        outputs["data:weight:airframe:fuselage:additional_mass:vertical"] = additional_mass_vertical
