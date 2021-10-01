"""
    Estimation of the bending moments on the fuselage
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
from fastoad.model_base import Atmosphere

FUSELAGE_MESH_POINT = 100


class ComputeBendingMoment(ExplicitComponent):
    """
        The bending moment distribution is inspired by the TASOPT 2.0 (2010). The distribution is first computed from
        the wing mass centroid to the tail (rear distribution). Then the distribution from the nose to the wing mass
        centroid is computed (front distribution). The two distributions match at the wing centroid by taking in account
        the lift and assuming the equilibrium.
        Like in the paper, the code computes extra material mass and adds it to the fuselage shell mass if it detects
        that the fuselage cannot resist the bending moments of the defined load cases. This last section does not have
        currently any impact on the rest of the code and is purely indicative.
    """

    def setup(self):
        self.add_input("data:mission:sizing:main_route:cruise:altitude", val=np.nan, units="m")
        self.add_input("data:geometry:cabin:length", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:maximum_width", val=np.nan, units="m")
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
        self.add_input("data:weight:furniture:passenger_seats:mass", val=np.nan, units="kg")
        self.add_input("data:weight:systems:life_support:insulation:mass", val=np.nan, units="kg")
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
        self.add_input("data:weight:aircraft:payload", val=np.nan, units="kg")
        self.add_input("data:weight:propulsion:engine:mass", val=np.nan, units="kg")
        self.add_input("data:weight:propulsion:engine:CG:x", val=np.nan, units="m")
        self.add_input("data:geometry:propulsion:layout", val=np.nan)
        self.add_input("data:flight_domain:diving_speed", val=np.nan, units="m/s")
        self.add_input("data:weight:airframe:fuselage:tail_cone_mass", val=np.nan, units="kg")
        self.add_input("data:weight:airframe:fuselage:nose_mass", val=np.nan, units="kg")
        self.add_input("data:weight:airframe:fuselage:shell_mass", val=np.nan, units="kg")
        self.add_input("data:loads:fuselage:inertia", val=np.nan, units="m**4")
        self.add_input("data:loads:fuselage:sigmaMh", val=np.nan, units="N/m**2")
        self.add_input(
            "data:weight:airframe:fuselage:total_additional_mass", val=np.nan, units="kg"
        )
        self.add_input("settings:materials:fuselage:skin:density", val=np.nan, units="kg/m**3")
        self.add_input("settings:materials:fuselage:skin:young_modulus", val=np.nan, units="Pa")
        self.add_input("settings:materials:fuselage:stringer:young_modulus", val=np.nan, units="Pa")

        self.add_output("data:loads:fuselage:x_vector", units="m", shape=FUSELAGE_MESH_POINT)
        self.add_output(
            "data:loads:fuselage:horizontal_bending_vector", units="N*m", shape=FUSELAGE_MESH_POINT
        )
        self.add_output(
            "data:loads:fuselage:vertical_bending_vector", units="N*m", shape=FUSELAGE_MESH_POINT
        )
        self.add_output("data:loads:fuselage:x_h_bend", units="m")
        self.add_output("data:loads:fuselage:additional_mass:horizontal", units="kg")
        self.add_output("data:loads:fuselage:additional_mass:vertical", units="kg")
        self.add_output("data:loads:fuselage:airframe_mass", units="kg")
        self.add_output(
            "data:loads:fuselage:horizontal_bending_inertia",
            units="m**4",
            shape=FUSELAGE_MESH_POINT,
        )
        self.add_output(
            "data:loads:fuselage:vertical_bending_inertia", units="m**4", shape=FUSELAGE_MESH_POINT
        )

        self.declare_partials("data:loads:fuselage:x_vector", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        cruise_altitude = inputs["data:mission:sizing:main_route:cruise:altitude"]
        l_cabin = inputs["data:geometry:cabin:length"]
        fuselage_max_width = inputs["data:geometry:fuselage:maximum_width"]
        fuselage_radius = fuselage_max_width / 2
        l_front = inputs["data:geometry:fuselage:front_length"]
        l_fuselage = inputs["data:geometry:fuselage:length"]
        wing_centroid = inputs["data:weight:airframe:wing:CG:x"]
        htp_cg = inputs["data:weight:airframe:horizontal_tail:CG:x"]
        vtp_cg = inputs["data:weight:airframe:vertical_tail:CG:x"]
        htp_mass = inputs["data:weight:airframe:horizontal_tail:mass"]
        vtp_mass = inputs["data:weight:airframe:vertical_tail:mass"]
        htp_area = inputs["data:geometry:horizontal_tail:area"]
        vtp_area = inputs["data:geometry:vertical_tail:area"]
        seats_mass = inputs["data:weight:furniture:passenger_seats:mass"]
        insulation_mass = inputs["data:weight:systems:life_support:insulation:mass"]
        air_conditioning_mass = inputs["data:weight:systems:life_support:air_conditioning:mass"]
        internal_lighting_mass = inputs["data:weight:systems:life_support:internal_lighting:mass"]
        seat_installation_mass = inputs["data:weight:systems:life_support:seat_installation:mass"]
        fixed_oxygen_mass = inputs["data:weight:systems:life_support:fixed_oxygen:mass"]
        security_kits_mass = inputs["data:weight:systems:life_support:security_kits:mass"]
        payload_mass = inputs["data:weight:aircraft:payload"]
        engine_mass = inputs["data:weight:propulsion:engine:mass"]
        engine_cg_x = inputs["data:weight:propulsion:engine:CG:x"]
        engine_layout = inputs["data:geometry:propulsion:layout"]
        v_d = inputs["data:flight_domain:diving_speed"]
        cone_mass = inputs["data:weight:airframe:fuselage:tail_cone_mass"]
        nose_mass = inputs["data:weight:airframe:fuselage:nose_mass"]
        fuselage_shell_mass = inputs["data:weight:airframe:fuselage:shell_mass"]
        bending_inertia = inputs["data:loads:fuselage:inertia"]
        sigma_mh = inputs["data:loads:fuselage:sigmaMh"]
        airframe_add_mass = inputs["data:weight:airframe:fuselage:total_additional_mass"]
        rho_skin = inputs["settings:materials:fuselage:skin:density"]
        e_skin = inputs["settings:materials:fuselage:skin:young_modulus"]
        e_stringer = inputs["settings:materials:fuselage:stringer:young_modulus"]

        w_cabin = g * (fuselage_shell_mass - cone_mass - nose_mass)
        w_cone = g * cone_mass
        w_nose = g * nose_mass

        # The following weight (windows, doors, floor,...) is applied over the cabin but not the nose or the cone.
        w_airframe_add = g * airframe_add_mass

        w_payload = g * payload_mass
        w_padd_uniform = g * (
            air_conditioning_mass
            + internal_lighting_mass
            + seat_installation_mass
            + fixed_oxygen_mass
            + security_kits_mass
        )
        w_insulation = g * insulation_mass
        w_seats = g * seats_mass
        w_engine = g * engine_mass

        # Lumped tail weight computation
        w_tail = g * (htp_mass + vtp_mass) + w_cone
        tail_x_cg = (
            htp_cg * htp_mass
            + vtp_cg * vtp_mass
            + 0.5 * (l_front + l_cabin + l_fuselage) * cone_mass
        ) / (htp_mass + vtp_mass + cone_mass)

        # Tail Aero Loads (at sea level since we want the maximum value reachable)
        rmh = 0.4
        rmv = 0.7
        cl_h_max = 1.2
        cl_v_max = 0.55
        density = Atmosphere(cruise_altitude, altitude_in_feet=False).density
        q_never_exceed = 0.5 * density * v_d ** 2
        lift_max_h = q_never_exceed * htp_area * cl_h_max
        lift_max_v = q_never_exceed * vtp_area * cl_v_max

        # TASOPT assumes that the wing centroid is at the middle of the cabin length.
        # We prefer to use the right geometry. This will affect the weight distribution of the cabin in the rear and
        # front distributions.
        x_ratio_centroid_to_cabin_front = (wing_centroid - l_front) / l_cabin
        x_ratio_centroid_to_cabin_rear = 1 - x_ratio_centroid_to_cabin_front

        # The fuselage length is roughly discretized with a fixed length step.
        nb_points_front = int(FUSELAGE_MESH_POINT * wing_centroid / tail_x_cg)
        x_vector_front = np.linspace(0, wing_centroid, nb_points_front)
        x_vector_rear = np.linspace(wing_centroid, tail_x_cg, FUSELAGE_MESH_POINT - nb_points_front)
        x_vector_plot = np.append(x_vector_front, x_vector_rear)
        horizontal_bending_vector_plot = np.array([])

        # Definition of the use cases for the horizontal bending moment:
        # 1. Emergency Landing
        # 2. Cruise flight conditions with impulsive activation of the elevator
        # The associated load factors come from the CS23 norm.
        n_lift = 3.8
        n_emergency_landing = 6.0
        load_case_array = np.array([[n_lift, lift_max_h], [n_emergency_landing, 0]], dtype="object")
        additional_mass_horizontal = 0.0

        # x_h_bend is the x-coordinate of the point where the available inertia of the fuselage is below the
        # sollicitated inertia. We update its value if needed.
        x_h_bend = wing_centroid

        for load_case in load_case_array:
            n = load_case[0]
            lift_max_h = load_case[1]

            # Calculation of the horizontal bending moment distribution on the rear of the fuselage
            horizontal_bending_vector_rear = np.array([])
            for x in x_vector_rear:
                if x <= l_front + l_cabin:
                    bending = n * x_ratio_centroid_to_cabin_rear / l_cabin * (
                        w_payload
                        + w_padd_uniform
                        + w_cabin
                        + w_airframe_add
                        + w_insulation
                        + w_seats
                    ) * (l_front + l_cabin - x) ** 2 + (n * w_tail + rmh * lift_max_h) * (
                        tail_x_cg - x
                    )
                else:
                    bending = (n * w_tail + rmh * lift_max_h) * (tail_x_cg - x)
                horizontal_bending_vector_rear = np.append(horizontal_bending_vector_rear, bending)

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
                if x_h_bend <= l_front + l_cabin:
                    # Definition of the added horizontal-axis bending area terms
                    a2_rear = (
                        n
                        * x_ratio_centroid_to_cabin_rear
                        / (l_cabin * fuselage_radius * sigma_mh)
                        * (w_payload + w_padd_uniform + w_cabin + w_insulation + w_seats)
                    )
                    a1_rear = (n * w_tail + rmh * lift_max_h) / (fuselage_radius * sigma_mh)
                    a0_rear = -bending_inertia / (fuselage_radius ** 2 * e_stringer / e_skin)
                    volume_rear = (
                        a2_rear
                        * (
                            (l_front + l_cabin - wing_centroid) ** 3
                            - (l_front + l_cabin - x_h_bend) ** 3
                        )
                        / 3
                        + a1_rear
                        * ((tail_x_cg - wing_centroid) ** 2 - (tail_x_cg - x_h_bend) ** 2)
                        / 2
                        + a0_rear * (x_h_bend - wing_centroid)
                    )
                else:
                    # Decomposition in 2 integrals : cabin zone and then up to x_h_bend in the tail cone
                    a2_rear = (
                        n
                        * x_ratio_centroid_to_cabin_rear
                        / (l_cabin * fuselage_radius * sigma_mh)
                        * (w_payload + w_padd_uniform + w_cabin + w_insulation + w_seats)
                    )
                    a1_rear = (n * w_tail + rmh * lift_max_h) / (fuselage_radius * sigma_mh)
                    a0_rear = -bending_inertia / (fuselage_radius ** 2 * e_stringer / e_skin)
                    volume_rear_cabin = (
                        a2_rear * (l_front + l_cabin - wing_centroid) ** 3 / 3
                        + a1_rear
                        * (
                            (tail_x_cg - wing_centroid) ** 2
                            - (tail_x_cg - (l_front + l_cabin)) ** 2
                        )
                        / 2
                        + a0_rear * (x_h_bend - wing_centroid)
                    )
                    volume_rear_tail = a1_rear * (
                        (tail_x_cg - (l_front + l_cabin)) ** 2 - (tail_x_cg - x_h_bend) ** 2
                    ) / 2 + a0_rear * (x_h_bend - (l_front + l_cabin))
                    volume_rear = volume_rear_cabin + volume_rear_tail

            # Calculation of the horizontal bending moment on the front part of the fuselage
            horizontal_bending_vector_front = np.array([])

            # The same expression as in the bending moment expression for the front of the fuselage is needed here,
            # applied at x = wing_centroid. And without the lift. This is to calculate the lift needed to match the
            # front and the rear distributions.
            max_moment_front = (
                n * w_nose * wing_centroid
                + n
                * x_ratio_centroid_to_cabin_front
                / l_cabin
                * (w_payload + w_padd_uniform + w_cabin + w_airframe_add + w_insulation + w_seats)
                * (wing_centroid - l_front) ** 2
            )
            if engine_layout == 3:
                max_moment_front += n * w_engine * (wing_centroid - engine_cg_x)
            moment_to_compensate_with_lift = horizontal_bending_vector_rear[0] - max_moment_front
            for x in x_vector_front:
                if engine_layout == 3:
                    if x <= engine_cg_x:
                        bending = (
                            n * w_nose / l_front * x ** 2
                            + moment_to_compensate_with_lift / wing_centroid * x
                        )
                    elif x <= l_front:
                        bending = (
                            n * w_nose / l_front * x ** 2
                            + moment_to_compensate_with_lift / wing_centroid * x
                            + n * w_engine * (x - engine_cg_x)
                        )
                    else:
                        bending = (
                            n * w_nose * x
                            + n * w_engine * (x - engine_cg_x)
                            + n
                            * x_ratio_centroid_to_cabin_front
                            / l_cabin
                            * (
                                w_payload
                                + w_padd_uniform
                                + w_cabin
                                + w_airframe_add
                                + w_insulation
                                + w_seats
                            )
                            * (x - l_front) ** 2
                            + moment_to_compensate_with_lift / wing_centroid * x
                        )
                else:
                    if x <= l_front:
                        bending = (
                            n * w_nose / l_front * x ** 2
                            + moment_to_compensate_with_lift / wing_centroid * x
                        )
                    else:
                        bending = (
                            n * w_nose * x
                            + n
                            * x_ratio_centroid_to_cabin_front
                            / l_cabin
                            * (
                                w_payload
                                + w_padd_uniform
                                + w_cabin
                                + w_airframe_add
                                + w_insulation
                                + w_seats
                            )
                            * (x - l_front) ** 2
                            + moment_to_compensate_with_lift / wing_centroid * x
                        )
                horizontal_bending_vector_front = np.append(
                    horizontal_bending_vector_front, bending
                )
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
                if x_h_bend <= l_front:
                    # Decomposition in two integrals : nose zone and cabin zone
                    a0_front = -bending_inertia / (fuselage_radius ** 2 * e_stringer / e_skin)
                    a1_front_nose = moment_to_compensate_with_lift / (
                        wing_centroid * fuselage_radius * sigma_mh
                    )
                    a2_front_nose = n * w_nose / (l_front * fuselage_radius * sigma_mh)
                    a1_front_cabin = (
                        n * w_nose + moment_to_compensate_with_lift / wing_centroid
                    ) / (fuselage_radius * sigma_mh)
                    a2_front_cabin = (
                        n
                        * x_ratio_centroid_to_cabin_front
                        / l_cabin
                        * (
                            w_payload
                            + w_padd_uniform
                            + w_cabin
                            + w_airframe_add
                            + w_insulation
                            + w_seats
                        )
                        / (fuselage_radius * sigma_mh)
                    )
                    volume_front_nose = (
                        a0_front * (l_front - x_h_bend)
                        + a1_front_nose / 2 * (l_front - x_h_bend) ** 2
                        + a2_front_nose / 3 * (l_front - x_h_bend) ** 3
                    )
                    volume_front_cabin = (
                        a0_front * (wing_centroid - l_front)
                        + a1_front_cabin / 2 * (wing_centroid - l_front) ** 2
                        + a2_front_cabin / 3 * (wing_centroid - l_front) ** 3
                    )
                    volume_front = volume_front_nose + volume_front_cabin
                else:
                    # One integral only : from xhbend to wing centroid
                    a0_front = -bending_inertia / (fuselage_radius ** 2 * e_stringer / e_skin)
                    a1_front_cabin = (
                        n * w_nose + moment_to_compensate_with_lift / wing_centroid
                    ) / (fuselage_radius * sigma_mh)
                    a2_front_cabin = (
                        n
                        * x_ratio_centroid_to_cabin_front
                        / l_cabin
                        * (
                            w_payload
                            + w_padd_uniform
                            + w_cabin
                            + w_airframe_add
                            + w_insulation
                            + w_seats
                        )
                        / (fuselage_radius * sigma_mh)
                    )
                    volume_front = (
                        a0_front * (wing_centroid - x_h_bend)
                        + a1_front_cabin / 2 * (wing_centroid - x_h_bend) ** 2
                        + a2_front_cabin
                        / 3
                        * ((wing_centroid - l_front) ** 3 - (x_h_bend - l_front) ** 3)
                    )

            volume_load_case = volume_front + volume_rear
            additional_mass_load_case = volume_load_case * rho_skin
            additional_mass_horizontal = max(additional_mass_load_case, additional_mass_horizontal)

            if n == n_lift:
                horizontal_bending_vector_plot = np.append(
                    horizontal_bending_vector_front, horizontal_bending_vector_rear
                )

        # Computation of the vertical bending moment. The only load case is the impulsive activation of the rudder in
        # cruise conditions.
        vertical_bending_vector = np.array([])
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

        # Total fuselage airframe mass
        airframe_mass = (
            fuselage_shell_mass
            + airframe_add_mass
            + additional_mass_horizontal
            + additional_mass_vertical
        )

        # Arrays to plot with postprocessing function
        vertical_bending_vector_plot = np.append(
            np.zeros(len(x_vector_front)), vertical_bending_vector
        )
        horizontal_bending_inertia_plot = (
            horizontal_bending_vector_plot * fuselage_radius / sigma_mh
        )
        vertical_bending_inertia_plot = vertical_bending_vector_plot * fuselage_radius / sigma_mh

        outputs["data:loads:fuselage:x_vector"] = x_vector_plot
        outputs["data:loads:fuselage:horizontal_bending_vector"] = horizontal_bending_vector_plot
        outputs["data:loads:fuselage:vertical_bending_vector"] = vertical_bending_vector_plot
        outputs["data:loads:fuselage:x_h_bend"] = x_h_bend
        outputs["data:loads:fuselage:additional_mass:horizontal"] = additional_mass_horizontal
        outputs["data:loads:fuselage:additional_mass:vertical"] = additional_mass_vertical
        outputs["data:loads:fuselage:airframe_mass"] = airframe_mass
        outputs["data:loads:fuselage:horizontal_bending_inertia"] = horizontal_bending_inertia_plot
        outputs["data:loads:fuselage:vertical_bending_inertia"] = vertical_bending_inertia_plot
