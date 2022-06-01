"""FAST - Copyright (c) 2016 ONERA ISAE."""
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

import math

import numpy as np
import openmdao.api as om
import scipy.interpolate as inter

import fastoad.api as oad
from stdatm import Atmosphere

from .figure_digitization import FigureDigitization
from ..constants import (
    SUBMODEL_HINGE_MOMENTS_TAIL_2D,
    SUBMODEL_HINGE_MOMENTS_TAIL_3D,
    SUBMODEL_HINGE_MOMENTS_TAIL,
)


@oad.RegisterSubmodel(
    SUBMODEL_HINGE_MOMENTS_TAIL, "fastga.submodel.aerodynamics.tail.hinge_moments.legacy"
)
class ComputeHingeMomentsTail(om.Group):
    def setup(self):
        self.add_subsystem(
            "two_d_hinge_moments",
            oad.RegisterSubmodel.get_submodel(SUBMODEL_HINGE_MOMENTS_TAIL_2D),
            promotes=["*"],
        )
        self.add_subsystem(
            "three_d_hinge_moments",
            oad.RegisterSubmodel.get_submodel(SUBMODEL_HINGE_MOMENTS_TAIL_3D),
            promotes=["*"],
        )


@oad.RegisterSubmodel(
    SUBMODEL_HINGE_MOMENTS_TAIL_2D, "fastga.submodel.aerodynamics.tail.hinge_moments.2d.legacy"
)
class Compute2DHingeMomentsTail(FigureDigitization):
    """
    Based on : Roskam, Jan. Airplane Design: Part 6-Preliminary Calculation of Aerodynamic,
    Thrust and Power Characteristics. DAR corporation, 1985. Section 10.4.1, 10.4.2.
    """

    def setup(self):
        self.add_input("data:geometry:horizontal_tail:elevator_chord_ratio", val=np.nan)
        self.add_input("data:geometry:horizontal_tail:thickness_ratio", val=np.nan)
        self.add_input("data:geometry:horizontal_tail:area", val=np.nan, units="m**2")
        self.add_input("data:geometry:wing:area", val=np.nan, units="m**2")
        self.add_input(
            "data:aerodynamics:horizontal_tail:airfoil:CL_alpha", val=np.nan, units="rad**-1"
        )
        self.add_input("data:TLAR:v_cruise", val=np.nan, units="m/s")
        self.add_input("data:mission:sizing:main_route:cruise:altitude", val=np.nan, units="m")

        self.add_output(
            "data:aerodynamics:horizontal_tail:cruise:hinge_moment:CH_alpha_2D", units="rad**-1"
        )
        self.add_output(
            "data:aerodynamics:horizontal_tail:cruise:hinge_moment:CH_delta_2D", units="rad**-1"
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        elevator_chord_ratio = inputs["data:geometry:horizontal_tail:elevator_chord_ratio"]
        tail_thickness_ratio = inputs["data:geometry:horizontal_tail:thickness_ratio"]
        cl_alpha_airfoil_ht = inputs["data:aerodynamics:horizontal_tail:airfoil:CL_alpha"]
        v_cruise = inputs["data:TLAR:v_cruise"]
        cruise_alt = inputs["data:mission:sizing:main_route:cruise:altitude"]

        # Section 10.4.1.1
        # Step 1.
        def y(x):
            return (
                tail_thickness_ratio
                / 0.2
                * (
                    0.2969 * x ** 0.5
                    - 0.126 * x
                    - 0.3516 * x ** 2.0
                    + 0.2843 * x ** 3.0
                    - 0.105 * x ** 4.0
                )
            )

        y_90 = y(0.90)
        y_95 = y(0.95)
        y_99 = y(0.99)

        tan_0_5_phi_te = (y(1.0 - 1.0e-6) - y(1.0)) / 1e-6
        tan_0_5_phi_te_prime = (y_90 / 2.0 - y_99 / 2.0) / 9.0
        tan_0_5_phi_te_prime_prime = (y_95 / 2.0 - y_99 / 2.0) / 9.0

        condition = bool(
            (
                (tan_0_5_phi_te == tan_0_5_phi_te_prime)
                and (tan_0_5_phi_te_prime == tan_0_5_phi_te_prime_prime)
                and (tan_0_5_phi_te_prime_prime == tail_thickness_ratio)
            )
        )

        # Step 2.
        cl_alpha_ht_th = 6.3 + tail_thickness_ratio / 0.2 * (7.3 - 6.3)

        k_cl_alpha = float(cl_alpha_airfoil_ht) / float(cl_alpha_ht_th)

        k_ch_alpha = self.k_ch_alpha(
            tail_thickness_ratio, cl_alpha_airfoil_ht, elevator_chord_ratio
        )

        ch_alpha = self.ch_alpha_th(tail_thickness_ratio, elevator_chord_ratio)

        ch_prime_alpha = k_ch_alpha * ch_alpha

        # Step 3.

        if condition:
            ch_prime_prime_alpha = ch_prime_alpha

        else:
            ch_prime_prime_alpha = ch_prime_alpha + 2.0 * cl_alpha_ht_th * (1.0 - k_cl_alpha) * (
                tan_0_5_phi_te_prime_prime - tail_thickness_ratio
            )

        # Step 4. The overhang cb/cf will we taken equal to 3 as we assume a 1/4, 3/4 chord
        # repartition for the hinge line) We will also assume that the thickness ratio of the
        # elevator is the same as the tail and that it has a round nose

        balance_ratio = math.sqrt((1.0 / 3.0) ** 2.0 - (tail_thickness_ratio * 5.0 / 4) ** 2.0)

        if balance_ratio < 0.15:
            balance_ratio = 0.15
        elif balance_ratio > 0.5:
            balance_ratio = 0.5

        k_ch_alpha_balance_inter = inter.interp1d([0.15, 0.50], [0.93, 0.2])
        k_ch_alpha_balance = k_ch_alpha_balance_inter(balance_ratio)

        ch_alpha_balance = k_ch_alpha_balance * ch_prime_prime_alpha

        # Step 5.

        sos_cruise = Atmosphere(cruise_alt, altitude_in_feet=False).speed_of_sound
        mach = v_cruise / sos_cruise
        beta = math.sqrt(1.0 - mach ** 2.0)

        ch_alpha_fin = ch_alpha_balance / beta

        # Section 10.4.1.2
        # Step 1., same as step 1. in previous section

        # Step 2.

        k_ch_delta = self.k_ch_delta(
            tail_thickness_ratio, cl_alpha_airfoil_ht, elevator_chord_ratio
        )

        ch_delta = self.ch_delta_th(tail_thickness_ratio, elevator_chord_ratio)

        ch_prime_delta = k_ch_delta * ch_delta

        # Step 3.

        if condition:
            ch_prime_prime_delta = ch_prime_delta

        else:
            cl_delta_th = self.cl_delta_theory_plain_flap(
                tail_thickness_ratio, elevator_chord_ratio
            )

            k_cl_delta = self.k_cl_delta_plain_flap(
                tail_thickness_ratio, cl_alpha_airfoil_ht, elevator_chord_ratio
            )

            ch_prime_prime_delta = ch_prime_delta + (
                2.0
                * cl_delta_th
                * (1.0 - k_cl_delta)
                * (tan_0_5_phi_te_prime_prime - tail_thickness_ratio)
            )

        # Step 4. Same assumption as the step 4. of the previous section. Only this time we only
        # take the curve for the NACA 0009
        k_ch_delta_balance = inter.interp1d([0.15, 0.50], [0.93, 0.2])(balance_ratio)

        ch_delta_balance = k_ch_delta_balance * ch_prime_prime_delta

        # Step 5.
        ch_delta_fin = ch_delta_balance / beta

        outputs["data:aerodynamics:horizontal_tail:cruise:hinge_moment:CH_alpha_2D"] = ch_alpha_fin
        outputs["data:aerodynamics:horizontal_tail:cruise:hinge_moment:CH_delta_2D"] = ch_delta_fin


@oad.RegisterSubmodel(
    SUBMODEL_HINGE_MOMENTS_TAIL_3D, "fastga.submodel.aerodynamics.tail.hinge_moments.3d.legacy"
)
class Compute3DHingeMomentsTail(FigureDigitization):
    """
    Based on : Roskam, Jan. Airplane Design: Part 6-Preliminary Calculation of Aerodynamic,
    Thrust and Power Characteristics. DAR corporation, 1985. Section 10.4.1.
    """

    def setup(self):
        self.add_input(
            "data:aerodynamics:horizontal_tail:cruise:hinge_moment:CH_alpha_2D",
            val=np.nan,
            units="rad**-1",
        )
        self.add_input(
            "data:aerodynamics:horizontal_tail:cruise:hinge_moment:CH_delta_2D",
            val=np.nan,
            units="rad**-1",
        )
        self.add_input("data:geometry:horizontal_tail:sweep_25", val=np.nan, units="deg")
        self.add_input("data:geometry:horizontal_tail:aspect_ratio", val=np.nan)
        self.add_input("data:geometry:horizontal_tail:elevator_chord_ratio", val=np.nan)
        self.add_input("data:mission:sizing:takeoff:elevator_angle", val=np.nan, units="deg")

        self.add_output(
            "data:aerodynamics:horizontal_tail:cruise:hinge_moment:CH_alpha", units="rad**-1"
        )
        self.add_output(
            "data:aerodynamics:horizontal_tail:cruise:hinge_moment:CH_delta", units="rad**-1"
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        ch_alpha_2d = inputs["data:aerodynamics:horizontal_tail:cruise:hinge_moment:CH_alpha_2D"]
        ch_delta_2d = inputs["data:aerodynamics:horizontal_tail:cruise:hinge_moment:CH_delta_2D"]
        sweep_25_ht = inputs["data:geometry:horizontal_tail:sweep_25"] * math.pi / 180.0
        ar_ht = inputs["data:geometry:horizontal_tail:aspect_ratio"]
        elevator_chord_ratio = inputs["data:geometry:horizontal_tail:elevator_chord_ratio"]
        elevator_angle = float(abs(inputs["data:mission:sizing:takeoff:elevator_angle"]))

        # Section 10.4.2.1
        # Step 1. : delta_Ch_alpha we will ignore it for now, same for delta_Ch_delta

        ch_alpha_3d = (
            (ar_ht * math.cos(sweep_25_ht)) / (ar_ht + 2.0 * math.cos(sweep_25_ht))
        ) * ch_alpha_2d

        # Section 10.4.2.2
        # We assume the same sweep angle for hl and tail
        sweep_hl = sweep_25_ht

        # We'll compute the elevator effectiveness factor in the worst case scenario, i.e,
        # with the highest deflection angle which we will take at 25 degree
        a_delta = self.k_prime_single_slotted(elevator_angle, elevator_chord_ratio)

        ch_delta_3d = (
            math.cos(sweep_25_ht)
            * math.cos(sweep_hl)
            * (
                ch_delta_2d
                + a_delta
                * ((2.0 * math.cos(sweep_25_ht)) / (ar_ht + 2.0 * math.cos(sweep_25_ht)))
                * ch_alpha_2d
            )
        )

        outputs["data:aerodynamics:horizontal_tail:cruise:hinge_moment:CH_alpha"] = ch_alpha_3d
        outputs["data:aerodynamics:horizontal_tail:cruise:hinge_moment:CH_delta"] = ch_delta_3d
