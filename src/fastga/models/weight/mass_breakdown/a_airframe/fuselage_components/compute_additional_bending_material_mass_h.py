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

from scipy.constants import g

FUSELAGE_MESH_POINT = 100


class ComputeAddBendingMassHorizontal(om.ExplicitComponent):
    """
    Computes the horizontal additional bending material based on the method described in TASOPT.
    """

    def setup(self):
        """
        Declaring inputs and outputs for the computation of the horizontal additional bending
        material.
        """
        self.add_input("data:geometry:wing:area", val=np.nan, units="m**2")
        self.add_input("data:geometry:fuselage:front_length", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:rear_length", val=np.nan, units="m")
        self.add_input("data:geometry:cabin:length", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:maximum_width", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:maximum_height", val=np.nan, units="m")
        self.add_input("data:geometry:propulsion:engine:layout", val=np.nan)

        self.add_input("data:aerodynamics:horizontal_tail:low_speed:CL_max_clean", val=np.nan)
        self.add_input("data:aerodynamics:elevator:low_speed:CL_delta", units="rad**-1")

        self.add_input("data:weight:propulsion:engine:CG:x", val=np.nan, units="m")
        self.add_input("data:weight:airframe:wing:CG:x", val=np.nan, units="m")
        self.add_input("data:weight:airframe:horizontal_tail:CG:x", val=np.nan, units="m")
        self.add_input("data:weight:airframe:vertical_tail:CG:x", val=np.nan, units="m")
        self.add_input("data:weight:airframe:horizontal_tail:mass", val=np.nan, units="kg")
        self.add_input("data:weight:airframe:vertical_tail:mass", val=np.nan, units="kg")
        self.add_input("data:weight:airframe:fuselage:shell:mass", val=np.nan, units="kg")
        self.add_input("data:weight:airframe:fuselage:cone:mass", val=np.nan, units="kg")
        self.add_input("data:weight:airframe:fuselage:insulation:mass", val=np.nan, units="kg")
        self.add_input("data:weight:airframe:fuselage:doors:mass", val=np.nan, units="kg")
        self.add_input("data:weight:airframe:fuselage:windows:mass", val=np.nan, units="kg")
        self.add_input("data:weight:airframe:fuselage:floor:mass", val=np.nan, units="kg")
        self.add_input("data:weight:airframe:fuselage:engine_support:mass", val=np.nan, units="kg")
        self.add_input("data:weight:aircraft:max_payload", val=np.nan, units="kg")
        self.add_input("data:weight:propulsion:engine:mass", val=np.nan, units="kg")
        self.add_input(
            "data:weight:systems:life_support:air_conditioning:mass", val=np.nan, units="kg"
        )
        self.add_input(
            "data:weight:systems:life_support:internal_lighting:mass", val=np.nan, units="kg"
        )
        self.add_input(
            "data:weight:systems:life_support:seat_installation:mass", val=np.nan, units="kg"
        )
        self.add_input("data:weight:systems:life_support:fixed_oxygen:mass", val=np.nan, units="kg")
        self.add_input(
            "data:weight:systems:life_support:security_kits:mass", val=np.nan, units="kg"
        )
        self.add_input("data:weight:furniture:passenger_seats:mass", val=np.nan, units="lb")

        self.add_input("data:loads:fuselage:inertia", val=np.nan, units="m**4")
        self.add_input("data:loads:fuselage:sigmaMh", val=np.nan, units="N/m**2")

        self.add_input("data:mission:sizing:landing:elevator_angle", val=np.nan, units="rad")
        self.add_input("data:mission:sizing:main_route:cruise:altitude", val=np.nan, units="m")
        self.add_input("data:mission:sizing:cs23:sizing_factor:ultimate_aircraft", val=np.nan)
        self.add_input("data:mission:sizing:cs23:characteristic_speed:vd", val=np.nan, units="m/s")
        self.add_input("data:mission:landing:cs23:sizing_factor:ultimate_aircraft", val=6.0)

        self.add_input("settings:materials:fuselage:skin:density", val=np.nan, units="kg/m**3")
        self.add_input("settings:materials:fuselage:skin:young_modulus", val=np.nan, units="Pa")
        self.add_input("settings:materials:fuselage:stringer:young_modulus", val=np.nan, units="Pa")

        self.add_output("data:weight:airframe:fuselage:additional_mass:horizontal", units="kg")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        """Computing the horizontal additional bending material."""
        wing_area = inputs["data:geometry:wing:area"]
        lav = inputs["data:geometry:fuselage:front_length"]
        lar = inputs["data:geometry:fuselage:rear_length"]
        cabin_length = inputs["data:geometry:cabin:length"]
        fuselage_max_width = inputs["data:geometry:fuselage:maximum_width"]
        fuselage_max_height = inputs["data:geometry:fuselage:maximum_height"]
        engine_layout = inputs["data:geometry:propulsion:engine:layout"]

        cl_max_htp_clean = inputs["data:aerodynamics:horizontal_tail:low_speed:CL_max_clean"]
        cl_delta_htp = inputs["data:aerodynamics:elevator:low_speed:CL_delta"]

        wing_centroid = inputs["data:weight:airframe:wing:CG:x"]
        htp_cg = inputs["data:weight:airframe:horizontal_tail:CG:x"]
        vtp_cg = inputs["data:weight:airframe:vertical_tail:CG:x"]
        engine_cg_x = inputs["data:weight:propulsion:engine:CG:x"]
        htp_mass = inputs["data:weight:airframe:horizontal_tail:mass"]
        vtp_mass = inputs["data:weight:airframe:vertical_tail:mass"]
        shell_mass = inputs["data:weight:airframe:fuselage:shell:mass"]
        cone_mass = inputs["data:weight:airframe:fuselage:cone:mass"]
        insulation_mass = inputs["data:weight:airframe:fuselage:insulation:mass"]
        doors_mass = inputs["data:weight:airframe:fuselage:doors:mass"]
        windows_mass = inputs["data:weight:airframe:fuselage:windows:mass"]
        floor_mass = inputs["data:weight:airframe:fuselage:floor:mass"]
        engine_support = inputs["data:weight:airframe:fuselage:engine_support:mass"]
        engine_mass = inputs["data:weight:propulsion:engine:mass"]
        max_payload = inputs["data:weight:aircraft:max_payload"]
        air_conditioning_mass = inputs["data:weight:systems:life_support:air_conditioning:mass"]
        lighting_mass = inputs["data:weight:systems:life_support:internal_lighting:mass"]
        seat_installation_mass = inputs["data:weight:systems:life_support:seat_installation:mass"]
        fixed_oxygen_mass = inputs["data:weight:systems:life_support:fixed_oxygen:mass"]
        security_kit_mass = inputs["data:weight:systems:life_support:security_kits:mass"]
        seats_mass = inputs["data:weight:furniture:passenger_seats:mass"]

        sigma_mh = inputs["data:loads:fuselage:sigmaMh"]
        bending_inertia = inputs["data:loads:fuselage:inertia"]

        cruise_altitude = inputs["data:mission:sizing:main_route:cruise:altitude"]
        delta_e_max = inputs["data:mission:sizing:landing:elevator_angle"]

        v_d = inputs["data:mission:sizing:cs23:characteristic_speed:vd"]
        n_ult_flight = inputs["data:mission:sizing:cs23:sizing_factor:ultimate_aircraft"]
        n_ult_landing = inputs["data:mission:landing:cs23:sizing_factor:ultimate_aircraft"]

        rho_skin = inputs["settings:materials:fuselage:skin:density"]
        e_skin = inputs["settings:materials:fuselage:skin:young_modulus"]
        e_stringer = inputs["settings:materials:fuselage:stringer:young_modulus"]

        fuselage_radius = np.sqrt(fuselage_max_height * fuselage_max_width) / 2.0

        cl_max_htp = cl_max_htp_clean + cl_delta_htp * delta_e_max

        density = Atmosphere(cruise_altitude, altitude_in_feet=False).density

        q_never_exceed = 0.5 * density * v_d ** 2
        lift_max_h = q_never_exceed * wing_area * cl_max_htp
        # Inertia relief factor, typical value according to TASOPT
        rmh = 0.4

        # We will assume that the cone mass is evenly distributed, thus its center of gravity
        # will coincide with the centroid of the triangle whose base is the end of the
        # cylindrical part of the fuselage and tip is the end of the cone, hence with the CoG is
        # at 1/3 of the rear length of the fuselage
        tail_x_cg = (
            htp_cg * htp_mass + vtp_cg * vtp_mass + (lav + cabin_length + 0.333 * lar) * cone_mass
        ) / (htp_mass + vtp_mass + cone_mass)

        # Ratio of the cabin before and after the wing centroid point
        x_ratio_centroid_to_cabin_front = (wing_centroid - lav) / cabin_length
        x_ratio_centroid_to_cabin_rear = 1 - x_ratio_centroid_to_cabin_front

        # The fuselage length is roughly discretized with a fixed length step.
        nb_points_front = int(FUSELAGE_MESH_POINT * wing_centroid / tail_x_cg)
        x_vector_front = np.linspace(0, wing_centroid, nb_points_front)
        x_vector_rear = np.linspace(wing_centroid, tail_x_cg, FUSELAGE_MESH_POINT - nb_points_front)

        # The force that will impact the bending on the fuselage are of two types,
        # either distributed or punctual. for simplicity, the only punctual masses will be the
        # tail aero loads, their mass and the cone mass (all lumped in one) and the engine when
        # applicable. The rest will be considered as distributed over the cabin.

        # Punctual loads
        tail_weight = (htp_mass + vtp_mass + cone_mass) * g
        engine_weight = (engine_support + engine_mass) * g

        # Distributed loads
        distributed_cabin_weight = (
            max_payload
            + air_conditioning_mass
            + lighting_mass
            + seat_installation_mass
            + fixed_oxygen_mass
            + security_kit_mass
            + insulation_mass
            + seats_mass
            + shell_mass
            + doors_mass
            + windows_mass
            + floor_mass
        ) * g

        load_case_array = ((n_ult_flight, lift_max_h), (n_ult_landing, 0.0))

        # Initiate the value at zero which will be its ultimate value if the skin thickness sized
        # for pressurization is enough to hold the other bending moment.
        additional_mass_horizontal = 0.0

        for load_case in load_case_array:
            load_factor = load_case[0]
            lift_max_h = load_case[1]

            # Calculation of the horizontal bending moment distribution on the rear of the fuselage
            horizontal_bending_vector_rear = np.zeros_like(x_vector_rear)
            for x_position in x_vector_rear:
                if x_position <= lav + cabin_length:
                    bending = load_factor * distributed_cabin_weight * (
                        lav + cabin_length - x_position
                    ) ** 2 + (load_factor * tail_weight + rmh * lift_max_h) * (
                        tail_x_cg - x_position
                    )
                else:
                    bending = (load_factor * tail_weight + rmh * lift_max_h) * (
                        tail_x_cg - x_position
                    )
                horizontal_bending_vector_rear[np.where(x_vector_rear == x_position)[0]] = bending

            # Calculation of x_h_bend
            index = np.where(
                horizontal_bending_vector_rear * fuselage_radius / sigma_mh
                - bending_inertia * e_skin / e_stringer
                > 0
            )[0]
            if index.size == 0:
                volume_rear = 0.0
            else:
                x_h_bend = x_vector_rear[max(index)]

                # Integration of the bending area
                if x_h_bend <= lav + cabin_length:
                    # Definition of the added horizontal-axis bending area terms
                    a2_rear = (
                        load_factor
                        * x_ratio_centroid_to_cabin_rear
                        / (2.0 * cabin_length * fuselage_radius * sigma_mh)
                        * distributed_cabin_weight
                    )
                    a1_rear = (load_factor * tail_weight + rmh * lift_max_h) / (
                        fuselage_radius * sigma_mh
                    )
                    a0_rear = -bending_inertia / (fuselage_radius ** 2 * e_stringer / e_skin)
                    volume_rear = (
                        a2_rear
                        * (
                            (lav + cabin_length - wing_centroid) ** 3
                            - (lav + cabin_length - x_h_bend) ** 3
                        )
                        / 3.0
                        + a1_rear
                        * ((tail_x_cg - wing_centroid) ** 2 - (tail_x_cg - x_h_bend) ** 2)
                        / 2.0
                        + a0_rear * (x_h_bend - wing_centroid)
                    )
                else:
                    # Decomposition in 2 integrals : cabin zone and then up to x_h_bend in the
                    # tail cone
                    a2_rear = (
                        load_factor
                        * x_ratio_centroid_to_cabin_rear
                        / (cabin_length * fuselage_radius * sigma_mh)
                        * distributed_cabin_weight
                    )
                    a1_rear = (load_factor * tail_weight + rmh * lift_max_h) / (
                        fuselage_radius * sigma_mh
                    )
                    a0_rear = -bending_inertia / (fuselage_radius ** 2 * e_stringer / e_skin)
                    volume_rear_cabin = (
                        a2_rear * (lav + cabin_length - wing_centroid) ** 3 / 3.0
                        + a1_rear
                        * (
                            (tail_x_cg - wing_centroid) ** 2
                            - (tail_x_cg - (lav + cabin_length)) ** 2
                        )
                        / 2.0
                        + a0_rear * (x_h_bend - wing_centroid)
                    )
                    volume_rear_tail = a1_rear * (
                        (tail_x_cg - (lav + cabin_length)) ** 2 - (tail_x_cg - x_h_bend) ** 2
                    ) / 2 + a0_rear * (x_h_bend - (lav + cabin_length))
                    volume_rear = volume_rear_cabin + volume_rear_tail

            # Calculation of the horizontal bending moment on the front part of the fuselage
            horizontal_bending_vector_front = np.zeros_like(x_vector_front)

            # The same expression as in the bending moment expression for the front of the
            # fuselage is needed here, applied at x = wing_centroid. And without the lift. This
            # is to calculate the lift needed to match the front and the rear distributions.
            max_moment_front = load_factor * distributed_cabin_weight * (wing_centroid - lav) ** 2

            if engine_layout == 3:
                max_moment_front += load_factor * engine_weight * (wing_centroid - engine_cg_x)

            moment_to_compensate_with_lift = horizontal_bending_vector_rear[0] - max_moment_front

            for x_position in x_vector_front:
                if engine_layout == 3:
                    if x_position <= engine_cg_x:
                        bending = moment_to_compensate_with_lift / wing_centroid * x_position
                    elif x_position <= lav:
                        bending = (
                            moment_to_compensate_with_lift / wing_centroid * x_position
                            + load_factor * engine_weight * (x_position - engine_cg_x)
                        )
                    else:
                        bending = (
                            load_factor * engine_weight * (x_position - engine_cg_x)
                            + load_factor * distributed_cabin_weight * (x_position - lav) ** 2
                            + moment_to_compensate_with_lift / wing_centroid * x_position
                        )
                else:
                    if x_position <= lav:
                        bending = moment_to_compensate_with_lift / wing_centroid * x_position
                    else:
                        bending = (
                            load_factor * distributed_cabin_weight * (x_position - lav) ** 2
                            + moment_to_compensate_with_lift / wing_centroid * x_position
                        )
                horizontal_bending_vector_front[np.where(x_vector_front == x_position)[0]] = bending

            # Calculation of x_h_bend
            index = np.where(
                horizontal_bending_vector_front * fuselage_radius / sigma_mh
                - bending_inertia * e_skin / e_stringer
                > 0
            )[0]

            if index.size == 0:
                volume_front = 0.0
            else:
                x_h_bend = x_vector_front[min(index)]

                # Integration of the bending area
                if x_h_bend <= lav:
                    # Decomposition in two integrals : nose zone and cabin zone
                    a0_front = -bending_inertia / (fuselage_radius ** 2 * e_stringer / e_skin)
                    a1_front_nose = moment_to_compensate_with_lift / (
                        wing_centroid * fuselage_radius * sigma_mh
                    )
                    a2_front_nose = 0.0
                    a1_front_cabin = (
                        moment_to_compensate_with_lift
                        / wing_centroid
                        / (fuselage_radius * sigma_mh)
                    )
                    a2_front_cabin = (
                        load_factor * distributed_cabin_weight / (fuselage_radius * sigma_mh)
                    )
                    volume_front_nose = (
                        a0_front * (lav - x_h_bend)
                        + a1_front_nose / 2 * (lav - x_h_bend) ** 2
                        + a2_front_nose / 3 * (lav - x_h_bend) ** 3
                    )
                    volume_front_cabin = (
                        a0_front * (wing_centroid - lav)
                        + a1_front_cabin / 2 * (wing_centroid - lav) ** 2
                        + a2_front_cabin / 3 * (wing_centroid - lav) ** 3
                    )
                    volume_front = volume_front_nose + volume_front_cabin
                else:
                    # One integral only : from x_h_bend to wing centroid
                    a0_front = -bending_inertia / (fuselage_radius ** 2 * e_stringer / e_skin)
                    a1_front_cabin = (moment_to_compensate_with_lift / wing_centroid) / (
                        fuselage_radius * sigma_mh
                    )
                    a2_front_cabin = (
                        load_factor * distributed_cabin_weight / (fuselage_radius * sigma_mh)
                    )
                    volume_front = (
                        a0_front * (wing_centroid - x_h_bend)
                        + a1_front_cabin / 2 * (wing_centroid - x_h_bend) ** 2
                        + a2_front_cabin / 3 * ((wing_centroid - lav) ** 3 - (x_h_bend - lav) ** 3)
                    )

            volume_load_case = volume_front + volume_rear
            additional_mass_load_case = volume_load_case * rho_skin
            additional_mass_horizontal = max(additional_mass_load_case, additional_mass_horizontal)

        outputs[
            "data:weight:airframe:fuselage:additional_mass:horizontal"
        ] = additional_mass_horizontal
