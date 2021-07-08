"""
    Estimation of bending moments on fuselage
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
from scipy.constants import g
from fastoad.utils.physics import Atmosphere

FUSELAGE_MESH_POINT = 100


class ComputeBendingMomentReversed(ExplicitComponent):
    # TODO: Document equations. Cite sources
    """

    """

    def setup(self):
        self.add_input("data:geometry:cabin:length", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:front_length", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:rear_length", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:length", val=np.nan, units="m")
        self.add_input("data:weight:airframe:wing:CG:x", val=np.nan, units="m")
        self.add_input("data:weight:airframe:horizontal_tail:CG:x", val=np.nan, units="m")
        self.add_input("data:weight:airframe:vertical_tail:CG:x", val=np.nan, units="m")
        self.add_input("data:weight:airframe:horizontal_tail:mass", val=np.nan, units="kg")
        self.add_input("data:weight:airframe:vertical_tail:mass", val=np.nan, units="kg")
        self.add_input("data:geometry:horizontal_tail:area", val=np.nan, units="m**2")
        self.add_input("data:geometry:vertical_tail:area", val=np.nan, units="m**2")
        self.add_input("data:mission:sizing:cs23:sizing_factor_ultimate", val=np.nan)
        self.add_input("data:weight:furniture:passenger_seats:mass", val=np.nan, units="kg")
        self.add_input("data:weight:airframe:fuselage:mass", val=np.nan, units="kg")
        self.add_input("data:weight:airframe:floor:mass", val=np.nan, units="kg")
        self.add_input("data:weight:systems:life_support:insulation:mass", val=np.nan, units="kg")
        self.add_input("data:weight:systems:life_support:air_conditioning:mass", val=np.nan, units="kg")
        self.add_input("data:weight:systems:life_support:internal_lighting:mass", val=np.nan, units="kg")
        self.add_input("data:weight:systems:life_support:seat_installation:mass", val=np.nan, units="kg")
        self.add_input("data:weight:systems:life_support:fixed_oxygen:mass", val=np.nan, units="kg")
        self.add_input("data:weight:systems:life_support:security_kits:mass", val=np.nan, units="kg")
        self.add_input("data:weight:aircraft:payload", val=np.nan, units="kg")
        self.add_input("data:TLAR:v_limit", val=np.nan, units="m/s")

        self.add_output("data:loads:fuselage:x_vector", units="m", shape=FUSELAGE_MESH_POINT)
        self.add_output("data:loads:fuselage:horizontal_bending_vector", units="N*m", shape=FUSELAGE_MESH_POINT)
        self.add_output("data:loads:fuselage:vertical_bending_vector", units="N*m", shape=FUSELAGE_MESH_POINT)

        self.declare_partials("data:loads:fuselage:x_vector", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        l_cabin = inputs["data:geometry:cabin:length"]
        l_front = inputs["data:geometry:fuselage:front_length"]
        l_rear = inputs["data:geometry:fuselage:rear_length"]
        l_fuselage = inputs["data:geometry:fuselage:length"]
        wing_centroid = inputs["data:weight:airframe:wing:CG:x"]
        htp_cg = inputs["data:weight:airframe:horizontal_tail:CG:x"]
        vtp_cg = inputs["data:weight:airframe:vertical_tail:CG:x"]
        htp_mass = inputs["data:weight:airframe:horizontal_tail:mass"]
        vtp_mass = inputs["data:weight:airframe:vertical_tail:mass"]
        htp_area = inputs["data:geometry:horizontal_tail:area"]
        vtp_area = inputs["data:geometry:vertical_tail:area"]
        n = inputs["data:mission:sizing:cs23:sizing_factor_ultimate"]
        seats_mass = inputs["data:weight:furniture:passenger_seats:mass"]
        fuselage_airframe_mass = inputs["data:weight:airframe:fuselage:mass"]
        floor_mass = inputs["data:weight:airframe:floor:mass"]
        insulation_mass = inputs["data:weight:systems:life_support:insulation:mass"]
        air_conditioning_mass = inputs["data:weight:systems:life_support:air_conditioning:mass"]
        internal_lighting_mass = inputs["data:weight:systems:life_support:internal_lighting:mass"]
        seat_installation_mass = inputs["data:weight:systems:life_support:seat_installation:mass"]
        fixed_oxygen_mass = inputs["data:weight:systems:life_support:fixed_oxygen:mass"]
        security_kits_mass = inputs["data:weight:systems:life_support:security_kits:mass"]
        payload_mass = inputs["data:weight:aircraft:payload"]
        max_speed = inputs["data:TLAR:v_limit"]

        # Airframe mass breakdown (to separate the tail cone from the rest of the structure).
        # Hypothesis : airframe mass uniformly distributed along the fuselage length
        w_shell = g * fuselage_airframe_mass * (l_front + l_cabin) / l_fuselage
        w_cone = g * fuselage_airframe_mass * l_rear / l_fuselage

        w_payload = g * payload_mass
        w_padd_uniform = g * (air_conditioning_mass + internal_lighting_mass + seat_installation_mass +
                              fixed_oxygen_mass + security_kits_mass)
        w_insulation = g * insulation_mass
        w_floor = g * floor_mass
        w_seats = g * seats_mass

        # Lumped tail weight computation
        w_tail = g * (htp_mass + vtp_mass) + w_cone
        tail_x_cg = (htp_cg * htp_mass + vtp_cg * vtp_mass + 0.5 * (l_front + l_cabin + l_fuselage) * w_cone / g) / \
                    (htp_mass + vtp_mass + w_cone / g)

        # Tail Aero Loads TODO replace Clmax with xml data and compute qNE
        rmh = 0.4
        rmv = 0.7
        cl_h_max = 0.2
        cl_v_max = 0.2
        density = Atmosphere(0, altitude_in_feet=False).density
        q_never_exceed = 0.5 * density * max_speed ** 2
        lift_max_h = q_never_exceed * htp_area * cl_h_max
        lift_max_v = q_never_exceed * vtp_area * cl_v_max

        x_ratio_centroid_to_cabin_front = (wing_centroid - l_front) / l_cabin
        x_ratio_centroid_to_cabin_rear = 1 - x_ratio_centroid_to_cabin_front

        nb_points_front = int(FUSELAGE_MESH_POINT * (wing_centroid - l_front) / (tail_x_cg - l_front))
        x_vector_front = np.linspace(l_front, wing_centroid, nb_points_front)
        x_vector_rear = np.linspace(wing_centroid, tail_x_cg, FUSELAGE_MESH_POINT - nb_points_front)
        horizontal_bending_vector_rear = np.array([])
        for x in x_vector_rear:
            if x <= l_front + l_cabin:
                bending = n * x_ratio_centroid_to_cabin_rear / l_cabin *\
                          (w_payload + w_padd_uniform + w_shell + w_insulation + w_floor + w_seats) *\
                          (l_front + l_cabin - x) ** 2 + \
                          (n * w_tail + rmh * lift_max_h) * (tail_x_cg - x)
            else:
                bending = (n * w_tail + rmh * lift_max_h) * (tail_x_cg - x)
            horizontal_bending_vector_rear = np.append(horizontal_bending_vector_rear, bending)

        horizontal_bending_vector_front = np.array([])
        for x in x_vector_front:
            bending = n * x_ratio_centroid_to_cabin_rear / l_cabin * \
                      (w_payload + w_padd_uniform + w_shell + w_insulation + w_floor + w_seats) * \
                      (l_front + l_cabin - wing_centroid) ** 2 / (wing_centroid ** 2 - l_front ** 2) * \
                      (x ** 2 - l_front ** 2) + \
                      (n * w_tail + rmh * lift_max_h) * (tail_x_cg - wing_centroid) / \
                      (wing_centroid - l_front) * (x - l_front)
            horizontal_bending_vector_front = np.append(horizontal_bending_vector_front, bending)

        vertical_bending_vector = np.array([])
        for x in x_vector_rear:
            bending = (rmv * lift_max_v) * (tail_x_cg - x)
            vertical_bending_vector = np.append(vertical_bending_vector, bending)
        x_vector = np.append(x_vector_front, x_vector_rear)
        horizontal_bending_vector = np.append(horizontal_bending_vector_front, horizontal_bending_vector_rear)
        vertical_bending_vector = np.append(np.zeros(len(horizontal_bending_vector_front)), vertical_bending_vector)

        outputs["data:loads:fuselage:x_vector"] = x_vector
        outputs["data:loads:fuselage:horizontal_bending_vector"] = horizontal_bending_vector
        outputs["data:loads:fuselage:vertical_bending_vector"] = vertical_bending_vector
