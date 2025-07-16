"""FAST - Copyright (c) 2016 ONERA ISAE."""
#  This file is part of FAST-OAD_CS23 : A framework for rapid Overall Aircraft Design
#  Copyright (C) 2025  ONERA & ISAE-SUPAERO
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
import scipy.interpolate as inter

import fastoad.api as oad
from stdatm import Atmosphere

from .figure_digitization import FigureDigitization
from ..constants import (
    SUBMODEL_HINGE_MOMENTS_TAIL_2D,
    SUBMODEL_HINGE_MOMENTS_TAIL_3D,
    SUBMODEL_HINGE_MOMENTS_TAIL,
)

from .digitization.compute_k_prime_single_slotted import ComputeSingleSlottedLiftEffectiveness


@oad.RegisterSubmodel(
    SUBMODEL_HINGE_MOMENTS_TAIL, "fastga.submodel.aerodynamics.tail.hinge_moments.legacy"
)
class ComputeHingeMomentsTail(om.Group):
    """This group collect all tail hinge moment calculations."""

    # pylint: disable=missing-function-docstring
    # Overriding OpenMDAO setup
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

    # pylint: disable=missing-function-docstring
    # Overriding OpenMDAO setup
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

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute, not all arguments are used
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
                * (0.2969 * x**0.5 - 0.126 * x - 0.3516 * x**2.0 + 0.2843 * x**3.0 - 0.105 * x**4.0)
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
            float(tail_thickness_ratio), float(cl_alpha_airfoil_ht), float(elevator_chord_ratio)
        )

        ch_alpha = self.ch_alpha_th(float(tail_thickness_ratio), float(elevator_chord_ratio))

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

        balance_ratio = np.sqrt((1.0 / 3.0) ** 2.0 - (tail_thickness_ratio * 5.0 / 4) ** 2.0)

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
        beta = np.sqrt(1.0 - mach**2.0)

        ch_alpha_fin = ch_alpha_balance / beta

        # Section 10.4.1.2
        # Step 1., same as step 1. in previous section

        # Step 2.

        k_ch_delta = self.k_ch_delta(
            float(tail_thickness_ratio), float(cl_alpha_airfoil_ht), float(elevator_chord_ratio)
        )

        ch_delta = self.ch_delta_th(float(tail_thickness_ratio), float(elevator_chord_ratio))

        ch_prime_delta = k_ch_delta * ch_delta

        # Step 3.

        if condition:
            ch_prime_prime_delta = ch_prime_delta

        else:
            cl_delta_th = self.cl_delta_theory_plain_flap(
                float(tail_thickness_ratio), float(elevator_chord_ratio)
            )

            k_cl_delta = self.k_cl_delta_plain_flap(
                float(tail_thickness_ratio), float(cl_alpha_airfoil_ht), float(elevator_chord_ratio)
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
class Compute3DHingeMomentsTail(om.Group):
    """
    Based on : Roskam, Jan. Airplane Design: Part 6-Preliminary Calculation of Aerodynamic,
    Thrust and Power Characteristics. DAR corporation, 1985. Section 10.4.1.
    """

    # pylint: disable=missing-function-docstring
    # Overriding OpenMDAO setup
    def setup(self):
        self.add_subsystem(
            name="hinge_moment_alpha_three_d", subsys=Compute3DHingeMomentAlpha(), promotes=["*"]
        )

        self.add_subsystem(
            name="single_slotted_lift_effectiveness",
            subsys=ComputeSingleSlottedLiftEffectiveness(),
            promotes=[
                ("flap_angle", "data:mission:sizing:takeoff:elevator_angle"),
                ("chord_ratio", "data:geometry:horizontal_tail:elevator_chord_ratio"),
            ],
        )

        self.add_subsystem(
            name="hinge_moment_delta_three_d",
            subsys=Compute3DHingeMomentDelta(),
            promotes=["data:*"],
        )

        self.connect(
            "single_slotted_lift_effectiveness.lift_effectiveness",
            "hinge_moment_delta_three_d.max_lift_effectiveness",
        )


class Compute3DHingeMomentAlpha(om.ExplicitComponent):
    """
    Based on : Roskam, Jan. Airplane Design: Part 6-Preliminary Calculation of Aerodynamic,
    Thrust and Power Characteristics. DAR corporation, 1985. Section 10.4.1.
    delta_Ch_alpha we will ignore it for now, same for delta_Ch_delta
    """

    # pylint: disable=missing-function-docstring
    # Overriding OpenMDAO setup
    def setup(self):
        self.add_input(
            "data:aerodynamics:horizontal_tail:cruise:hinge_moment:CH_alpha_2D",
            val=np.nan,
            units="rad**-1",
        )
        self.add_input("data:geometry:horizontal_tail:sweep_25", val=np.nan, units="rad")
        self.add_input("data:geometry:horizontal_tail:aspect_ratio", val=np.nan)

        self.add_output(
            "data:aerodynamics:horizontal_tail:cruise:hinge_moment:CH_alpha", units="rad**-1"
        )

        self.declare_partials(
            of="data:aerodynamics:horizontal_tail:cruise:hinge_moment:CH_alpha",
            wrt="*",
            method="exact",
        )

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute, not all arguments are used
    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        ch_alpha_2d = inputs["data:aerodynamics:horizontal_tail:cruise:hinge_moment:CH_alpha_2D"]
        sweep_25_ht = inputs["data:geometry:horizontal_tail:sweep_25"]
        ar_ht = inputs["data:geometry:horizontal_tail:aspect_ratio"]

        outputs["data:aerodynamics:horizontal_tail:cruise:hinge_moment:CH_alpha"] = (
            (ar_ht * np.cos(sweep_25_ht)) / (ar_ht + 2.0 * np.cos(sweep_25_ht))
        ) * ch_alpha_2d

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute_partials, not all arguments are used
    def compute_partials(self, inputs, partials, discrete_inputs=None):
        ch_alpha_2d = inputs["data:aerodynamics:horizontal_tail:cruise:hinge_moment:CH_alpha_2D"]
        sweep_25_ht = inputs["data:geometry:horizontal_tail:sweep_25"]
        ar_ht = inputs["data:geometry:horizontal_tail:aspect_ratio"]

        partials[
            "data:aerodynamics:horizontal_tail:cruise:hinge_moment:CH_alpha",
            "data:aerodynamics:horizontal_tail:cruise:hinge_moment:CH_alpha_2D",
        ] = (ar_ht * np.cos(sweep_25_ht)) / (ar_ht + 2.0 * np.cos(sweep_25_ht))

        partials[
            "data:aerodynamics:horizontal_tail:cruise:hinge_moment:CH_alpha",
            "data:geometry:horizontal_tail:sweep_25",
        ] = (
            -ch_alpha_2d
            * ar_ht**2.0
            * np.sin(sweep_25_ht)
            / (2.0 * np.cos(sweep_25_ht) + ar_ht) ** 2.0
        )

        partials[
            "data:aerodynamics:horizontal_tail:cruise:hinge_moment:CH_alpha",
            "data:geometry:horizontal_tail:aspect_ratio",
        ] = (
            2.0
            * ch_alpha_2d
            * np.cos(sweep_25_ht) ** 2.0
            / (2.0 * np.cos(sweep_25_ht) + ar_ht) ** 2.0
        )


class Compute3DHingeMomentDelta(om.ExplicitComponent):
    """
    Based on : Roskam, Jan. Airplane Design: Part 6-Preliminary Calculation of Aerodynamic,
    Thrust and Power Characteristics. DAR corporation, 1985. Section 10.4.1.
    """

    # pylint: disable=missing-function-docstring
    # Overriding OpenMDAO setup
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
        self.add_input("data:geometry:horizontal_tail:sweep_25", val=np.nan, units="rad")
        self.add_input("data:geometry:horizontal_tail:aspect_ratio", val=np.nan)
        self.add_input("max_lift_effectiveness", val=np.nan)

        self.add_output(
            "data:aerodynamics:horizontal_tail:cruise:hinge_moment:CH_delta", units="rad**-1"
        )

        self.declare_partials(
            of="data:aerodynamics:horizontal_tail:cruise:hinge_moment:CH_delta",
            wrt="*",
            method="exact",
        )

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute, not all arguments are used
    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        ch_alpha_2d = inputs["data:aerodynamics:horizontal_tail:cruise:hinge_moment:CH_alpha_2D"]
        ch_delta_2d = inputs["data:aerodynamics:horizontal_tail:cruise:hinge_moment:CH_delta_2D"]
        sweep_25_ht = inputs["data:geometry:horizontal_tail:sweep_25"]
        ar_ht = inputs["data:geometry:horizontal_tail:aspect_ratio"]
        a_delta = inputs["max_lift_effectiveness"]

        # Section 10.4.2.2
        # We assume the same sweep angle for hl and tail
        sweep_hl = sweep_25_ht

        outputs["data:aerodynamics:horizontal_tail:cruise:hinge_moment:CH_delta"] = (
            np.cos(sweep_25_ht)
            * np.cos(sweep_hl)
            * (
                ch_delta_2d
                + a_delta
                * ((2.0 * np.cos(sweep_25_ht)) / (ar_ht + 2.0 * np.cos(sweep_25_ht)))
                * ch_alpha_2d
            )
        )

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute_partials, not all arguments are used
    def compute_partials(self, inputs, partials, discrete_inputs=None):
        ch_alpha_2d = inputs["data:aerodynamics:horizontal_tail:cruise:hinge_moment:CH_alpha_2D"]
        ch_delta_2d = inputs["data:aerodynamics:horizontal_tail:cruise:hinge_moment:CH_delta_2D"]
        sweep_25_ht = inputs["data:geometry:horizontal_tail:sweep_25"]
        ar_ht = inputs["data:geometry:horizontal_tail:aspect_ratio"]
        a_delta = inputs["max_lift_effectiveness"]
        common_denominator = 2.0 * np.cos(sweep_25_ht) + ar_ht

        partials[
            "data:aerodynamics:horizontal_tail:cruise:hinge_moment:CH_delta",
            "data:geometry:horizontal_tail:sweep_25",
        ] = (
            -2.0
            * np.cos(sweep_25_ht)
            * np.sin(sweep_25_ht)
            * (
                4.0 * (ch_delta_2d + a_delta * ch_alpha_2d) * np.cos(sweep_25_ht) ** 2.0
                + (4.0 * ch_delta_2d + 3.0 * a_delta * ch_alpha_2d) * ar_ht * np.cos(sweep_25_ht)
                + ch_delta_2d * ar_ht**2.0
            )
            / common_denominator**2.0
        )

        partials[
            "data:aerodynamics:horizontal_tail:cruise:hinge_moment:CH_delta",
            "data:aerodynamics:horizontal_tail:cruise:hinge_moment:CH_alpha_2D",
        ] = 2.0 * a_delta * np.cos(sweep_25_ht) ** 3.0 / common_denominator

        partials[
            "data:aerodynamics:horizontal_tail:cruise:hinge_moment:CH_delta",
            "data:aerodynamics:horizontal_tail:cruise:hinge_moment:CH_delta_2D",
        ] = np.cos(sweep_25_ht) ** 2.0

        partials[
            "data:aerodynamics:horizontal_tail:cruise:hinge_moment:CH_delta",
            "max_lift_effectiveness",
        ] = 2.0 * ch_alpha_2d * np.cos(sweep_25_ht) ** 3.0 / common_denominator

        partials[
            "data:aerodynamics:horizontal_tail:cruise:hinge_moment:CH_delta",
            "data:geometry:horizontal_tail:aspect_ratio",
        ] = -(2.0 * a_delta * ch_alpha_2d * np.cos(sweep_25_ht) ** 3.0) / common_denominator**2.0
