"""Computation of lift and drag increment due to high-lift devices."""
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
from typing import Union, Tuple

import numpy as np
import fastoad.api as oad
from scipy import interpolate

from .figure_digitization import FigureDigitization
from ..constants import SUBMODEL_DELTA_HIGH_LIFT


@oad.RegisterSubmodel(
    SUBMODEL_DELTA_HIGH_LIFT, "fastga.submodel.aerodynamics.high_lift.delta.legacy"
)
class ComputeDeltaHighLift(FigureDigitization):
    """
    Provides lift and drag increments due to high-lift devices.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.phase = None

    def setup(self):

        self.add_input("data:geometry:wing:span", val=np.nan, units="m")
        self.add_input("data:geometry:wing:area", val=np.nan, units="m**2")
        self.add_input("data:geometry:wing:taper_ratio", val=np.nan)
        self.add_input("data:geometry:fuselage:maximum_width", val=np.nan, units="m")
        self.add_input("data:geometry:wing:root:y", val=np.nan, units="m")
        self.add_input("data:geometry:wing:root:chord", val=np.nan, units="m")
        self.add_input("data:geometry:wing:thickness_ratio", val=np.nan)
        self.add_input("data:geometry:wing:aspect_ratio", val=np.nan)
        self.add_input("data:geometry:wing:sweep_25", val=np.nan, units="deg")
        self.add_input("data:geometry:flap:chord_ratio", val=0.2)
        self.add_input("data:geometry:flap:span_ratio", val=np.nan)
        self.add_input("data:geometry:flap_type", val=np.nan)
        self.add_input("data:aerodynamics:wing:low_speed:CL_alpha", val=np.nan, units="rad**-1")
        self.add_input("data:aerodynamics:low_speed:mach", val=np.nan)

        self.add_input("data:aerodynamics:wing:airfoil:CL_alpha", val=np.nan, units="rad**-1")
        self.add_input("data:mission:sizing:landing:flap_angle", val=30.0, units="deg")
        self.add_input("data:mission:sizing:takeoff:flap_angle", val=10.0, units="deg")

        self.add_output("data:aerodynamics:flaps:landing:CL")
        self.add_output("data:aerodynamics:flaps:landing:CL_2D")
        self.add_output("data:aerodynamics:flaps:landing:CL_max")
        self.add_output("data:aerodynamics:flaps:landing:CM")
        self.add_output("data:aerodynamics:flaps:landing:CM_2D")
        self.add_output("data:aerodynamics:flaps:landing:CD")
        self.add_output("data:aerodynamics:flaps:landing:CD_2D")
        self.add_output("data:aerodynamics:flaps:takeoff:CL")
        self.add_output("data:aerodynamics:flaps:takeoff:CL_2D")
        self.add_output("data:aerodynamics:flaps:takeoff:CL_max")
        self.add_output("data:aerodynamics:flaps:takeoff:CM")
        self.add_output("data:aerodynamics:flaps:takeoff:CM_2D")
        self.add_output("data:aerodynamics:flaps:takeoff:CD")
        self.add_output("data:aerodynamics:flaps:takeoff:CD_2D")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        mach_ls = inputs["data:aerodynamics:low_speed:mach"]
        flap_chord_ratio = inputs["data:geometry:flap:chord_ratio"]
        flap_area_ratio = self._compute_flap_area_ratio(inputs)

        # Computes flaps contribution during low speed operations (take-off/landing)
        for self.phase in ["landing", "takeoff"]:
            if self.phase == "landing":
                flap_angle = float(inputs["data:mission:sizing:landing:flap_angle"])
                (
                    outputs["data:aerodynamics:flaps:landing:CL"],
                    outputs["data:aerodynamics:flaps:landing:CL_max"],
                ) = self._get_flaps_delta_cl(
                    inputs,
                    flap_angle,
                    mach_ls,
                )
                cl_2d = self._compute_delta_cl_airfoil_2d(inputs, flap_angle, mach_ls)
                outputs["data:aerodynamics:flaps:landing:CL_2D"] = cl_2d
                outputs["data:aerodynamics:flaps:landing:CM"] = self._get_flaps_delta_cm(
                    inputs,
                    flap_angle,
                    mach_ls,
                )
                x_cp_c_prime = self.x_cp_c_prime(flap_chord_ratio)
                outputs["data:aerodynamics:flaps:landing:CM_2D"] = cl_2d * (0.25 - x_cp_c_prime)
                cd_3d = self._get_flaps_delta_cd(
                    inputs["data:geometry:flap_type"],
                    inputs["data:geometry:flap:chord_ratio"],
                    inputs["data:geometry:wing:thickness_ratio"],
                    flap_angle,
                    flap_area_ratio,
                )
                outputs["data:aerodynamics:flaps:landing:CD"] = cd_3d
                outputs["data:aerodynamics:flaps:landing:CD_2D"] = cd_3d / flap_area_ratio
            else:
                flap_angle = float(inputs["data:mission:sizing:takeoff:flap_angle"])
                (
                    outputs["data:aerodynamics:flaps:takeoff:CL"],
                    outputs["data:aerodynamics:flaps:takeoff:CL_max"],
                ) = self._get_flaps_delta_cl(
                    inputs,
                    flap_angle,
                    mach_ls,
                )
                cl_2d = self._compute_delta_cl_airfoil_2d(inputs, flap_angle, mach_ls)
                outputs["data:aerodynamics:flaps:takeoff:CL_2D"] = cl_2d
                outputs["data:aerodynamics:flaps:takeoff:CM"] = self._get_flaps_delta_cm(
                    inputs,
                    flap_angle,
                    mach_ls,
                )
                x_cp_c_prime = self.x_cp_c_prime(flap_chord_ratio)
                outputs["data:aerodynamics:flaps:takeoff:CM_2D"] = cl_2d * (0.25 - x_cp_c_prime)
                cd_3d = self._get_flaps_delta_cd(
                    inputs["data:geometry:flap_type"],
                    inputs["data:geometry:flap:chord_ratio"],
                    inputs["data:geometry:wing:thickness_ratio"],
                    flap_angle,
                    self._compute_flap_area_ratio(inputs),
                )
                outputs["data:aerodynamics:flaps:takeoff:CD"] = cd_3d
                outputs["data:aerodynamics:flaps:takeoff:CD_2D"] = cd_3d / flap_area_ratio

    def _get_elevator_delta_cl(
        self, inputs, elevator_angle: Union[float, np.array]
    ) -> Union[float, np.array]:
        """
        Computes the elevator lift increment as a plain flap following the method presented in
        Roskam part 6, section 8.1.2.1.a.

        :param elevator_angle: elevator angle (in Degree).
        :return: lift coefficient derivative.
        """

        ht_area = inputs["data:geometry:horizontal_tail:area"]
        wing_area = inputs["data:geometry:wing:area"]
        elevator_chord_ratio = inputs["data:geometry:horizontal_tail:elevator_chord_ratio"]
        htp_thickness_ratio = inputs["data:geometry:horizontal_tail:thickness_ratio"]
        cl_alpha_airfoil_ht = inputs["data:aerodynamics:horizontal_tail:airfoil:CL_alpha"]

        # Elevator (plain flap). Default: maximum deflection (25deg)
        cl_delta_theory = self.cl_delta_theory_plain_flap(htp_thickness_ratio, elevator_chord_ratio)
        k = self.k_prime_plain_flap(abs(elevator_angle), elevator_chord_ratio)
        k_cl_delta = self.k_cl_delta_plain_flap(
            htp_thickness_ratio, cl_alpha_airfoil_ht, elevator_chord_ratio
        )
        cl_alpha_elev = (cl_delta_theory * k * k_cl_delta) * ht_area / wing_area
        cl_alpha_elev *= 0.9  # Correction for the central fuselage part (no elevator there)

        return cl_alpha_elev

    def _get_flaps_delta_cl(self, inputs, flap_angle: float, mach: float) -> Tuple[float, float]:
        """
        Method based on...

        :param flap_angle: flap angle (in Degree)
        :param mach: air speed
        :return: increment of lift coefficient
        """

        cl_alpha_wing = inputs["data:aerodynamics:wing:low_speed:CL_alpha"]
        span_wing = inputs["data:geometry:wing:span"]
        y1_wing = inputs["data:geometry:fuselage:maximum_width"] / 2.0
        y2_wing = inputs["data:geometry:wing:root:y"]
        flap_span_ratio = inputs["data:geometry:flap:span_ratio"]
        taper_ratio_wing = inputs["data:geometry:wing:taper_ratio"]
        aspect_ratio_wing = float(inputs["data:geometry:wing:aspect_ratio"])
        cl_alpha_airfoil_wing = inputs["data:aerodynamics:wing:airfoil:CL_alpha"]

        # 2D flap lift coefficient
        delta_cl_airfoil = self._compute_delta_cl_airfoil_2d(inputs, flap_angle, mach)
        # Roskam 3D flap parameters
        eta_in = y1_wing / (span_wing / 2.0)
        eta_out = ((y2_wing - y1_wing) + flap_span_ratio * (span_wing / 2.0 - y2_wing)) / (
            span_wing / 2.0 - y2_wing
        )
        k_b = self.k_b_flaps(eta_in, eta_out, taper_ratio_wing)
        a_delta_flap = self.a_delta_airfoil(float(inputs["data:geometry:flap:chord_ratio"]))
        k_a_delta = self.k_a_delta(a_delta_flap, aspect_ratio_wing)
        delta_cl0_flaps = (
            k_b * delta_cl_airfoil * (cl_alpha_wing / cl_alpha_airfoil_wing) * k_a_delta
        )
        delta_cl_max_flaps = self._compute_delta_cl_max_flaps(inputs, flap_angle)

        return delta_cl0_flaps, delta_cl_max_flaps

    def _get_flaps_delta_cm(self, inputs, flap_angle: float, mach: float) -> float:
        """
        Method based on Roskam book.

        :param flap_angle: flap angle (in Degree).
        :param mach: air speed.
        :return: increment of moment coefficient.
        """

        cl_alpha_airfoil_wing = inputs["data:aerodynamics:wing:airfoil:CL_alpha"]
        span_wing = inputs["data:geometry:wing:span"]
        y1_wing = inputs["data:geometry:fuselage:maximum_width"] / 2.0
        y2_wing = inputs["data:geometry:wing:root:y"]
        flap_span_ratio = inputs["data:geometry:flap:span_ratio"]
        taper_ratio_wing = float(inputs["data:geometry:wing:taper_ratio"])
        aspect_ratio_wing = float(inputs["data:geometry:wing:aspect_ratio"])
        flap_chord_ratio = float(inputs["data:geometry:flap:chord_ratio"])
        wing_thickness_ratio = float(inputs["data:geometry:wing:thickness_ratio"])
        sweep_25 = float(inputs["data:geometry:wing:sweep_25"]) * np.pi / 180.0

        beta_ref = np.sqrt(1.0 - mach ** 2.0)
        k = cl_alpha_airfoil_wing / (2.0 * np.pi)
        cl_alpha_ref = (
            2.0 * np.pi * 6.0 / (2.0 + np.sqrt((36.0 * beta_ref ** 2.0 / k ** 2.0 + 4.0)))
        )

        # First we need to compute the increment in the lift coefficient for the reference case
        delta_cl_2d_ref = self._compute_delta_cl_airfoil_2d(inputs, flap_angle, mach)
        eta_in_ref = 0.0
        eta_out_ref = 1.0
        kb_ref = self.k_b_flaps(eta_in_ref, eta_out_ref, taper_ratio_wing)
        a_delta_flap_ref = self.a_delta_airfoil(float(inputs["data:geometry:flap:chord_ratio"]))
        k_a_delta_ref = self.k_a_delta(a_delta_flap_ref, 6.0)
        delta_cl_ref = (
            kb_ref * delta_cl_2d_ref * (cl_alpha_ref / cl_alpha_airfoil_wing) * k_a_delta_ref
        )

        # Now we can compute the coefficient
        eta_in = float(y1_wing / (span_wing / 2.0))
        eta_out = float(
            ((y2_wing - y1_wing) + flap_span_ratio * (span_wing / 2.0 - y2_wing))
            / (span_wing / 2.0 - y2_wing)
        )
        k_delta = self.k_delta_flaps(taper_ratio_wing, eta_in, eta_out)
        k_p = self.k_p_flaps(taper_ratio_wing, eta_in, eta_out)
        delta_cm_delta_cl_ref = self.pitch_to_reference_lift(wing_thickness_ratio, flap_chord_ratio)

        delta_cm_flap = (
            k_delta * aspect_ratio_wing / 1.5 * np.tan(sweep_25) + k_p * delta_cm_delta_cl_ref
        ) * delta_cl_ref

        return delta_cm_flap

    @staticmethod
    def _get_flaps_delta_cd(
        flap_type, chord_ratio, thickness_ratio, flap_angle: float, area_ratio
    ) -> float:
        """
        Method from Young (in Gudmundsson book; page 725).

        :param flap_angle: flap angle (in Degree).
        :param flap_type: flap type.
        :param area_ratio: ratio of control surface area over lifting surface area.
        :param chord_ratio: ratio of control surface chord over lifting surface chord.
        :param thickness_ratio: thickness ratio of the lifting surface.
        :return: increment of drag coefficient.
        """

        if flap_type == 0.0:  # Plain flap
            k1_0_12 = (
                -21.09 * chord_ratio ** 3
                + 14.091 * chord_ratio ** 2
                + 3.165 * chord_ratio
                - 0.00103
            )
            k1_0_21 = (
                -19.988 * chord_ratio ** 3 + 12.68 * chord_ratio ** 2 + 3.363 * chord_ratio - 0.0050
            )
            k1_0_30 = (
                -0.000 * chord_ratio ** 3 + 4.694 * chord_ratio ** 2 + 4.372 * chord_ratio - 0.0031
            )
            flap_chord_contribution = interpolate.interp1d(
                [0.12, 0.21, 0.30], [float(k1_0_12), float(k1_0_21), float(k1_0_30)]
            )(np.clip(thickness_ratio, 0.12, 0.30))
            flap_deflection_contribution = (
                -3.795e-7 * flap_angle ** 3
                + 5.387e-5 * flap_angle ** 2
                + 6.843e-4 * flap_angle
                - 1.4729e-3
            )

        elif flap_type == 1.0:  # slotted flap
            k1_0_12 = (
                179.32 * chord_ratio ** 4
                - 111.6 * chord_ratio ** 3
                + 28.929 * chord_ratio ** 2
                + 2.3705 * chord_ratio
                - 0.0089
            )
            k1_0_21 = (
                0.000 * chord_ratio ** 4
                - 0.000 * chord_ratio ** 3
                + 8.2658 * chord_ratio ** 2
                + 3.4564 * chord_ratio
                - 0.0054
            )
            flap_chord_contribution = interpolate.interp1d(
                [0.12, 0.21], [float(k1_0_12), float(k1_0_21)]
            )(np.clip(thickness_ratio, 0.12, 0.21))
            k2_0_12 = (
                -3.9877e-12 * flap_angle ** 6
                + 1.1685e-9 * flap_angle ** 5
                - 1.2846e-7 * flap_angle ** 4
                + 6.1742e-6 * flap_angle ** 3
                - 9.89444e-5 * flap_angle ** 2
                + 6.8324e-4 * flap_angle
                - 3.892e-4
            )
            k2_0_21 = (
                -0.0 * flap_angle ** 6
                - 4.6025e-11 * flap_angle ** 5
                + 1.0025e-8 * flap_angle ** 4
                - 9.8465e-7 * flap_angle ** 3
                + 5.6732e-5 * flap_angle ** 2
                - 2.64884e-4 * flap_angle
                - 3.3591e-4
            )
            k2_0_30 = (
                0.0 * flap_angle ** 6
                + 0.0 * flap_angle ** 5
                - 0.0 * flap_angle ** 4
                - 3.6841e-7 * flap_angle ** 3
                + 5.3342e-5 * flap_angle ** 2
                - 41677e-3 * flap_angle
                + 6.749e-4
            )
            flap_deflection_contribution = interpolate.interp1d(
                [0.12, 0.21, 0.30], [float(k2_0_12), float(k2_0_21), float(k2_0_30)]
            )(np.clip(thickness_ratio, 0.12, 0.30))

        else:  # Split flap
            k1_0_12 = (
                -21.09 * chord_ratio ** 3
                + 14.091 * chord_ratio ** 2
                + 3.165 * chord_ratio
                - 0.00103
            )
            k1_0_21 = (
                -19.988 * chord_ratio ** 3 + 12.68 * chord_ratio ** 2 + 3.363 * chord_ratio - 0.0050
            )
            k1_0_30 = (
                -0.000 * chord_ratio ** 3 + 4.694 * chord_ratio ** 2 + 4.372 * chord_ratio - 0.0031
            )
            flap_chord_contribution = interpolate.interp1d(
                [0.12, 0.21, 0.30], [float(k1_0_12), float(k1_0_21), float(k1_0_30)]
            )(np.clip(thickness_ratio, 0.12, 0.30))
            k2_0_12 = (
                -4.161e-7 * flap_angle ** 3
                + 5.5496e-5 * flap_angle ** 2
                + 1.0110e-3 * flap_angle
                - 2.219e-5
            )
            k2_0_21 = (
                -5.1007e-7 * flap_angle ** 3
                + 7.4060e-5 * flap_angle ** 2
                - 4.8877e-5 * flap_angle
                + 8.1775e-4
            )
            k2_0_30 = (
                -3.2740e-7 * flap_angle ** 3
                + 5.598e-5 * flap_angle ** 2
                - 1.2443e-4 * flap_angle
                + 5.1647e-4
            )
            flap_deflection_contribution = interpolate.interp1d(
                [0.12, 0.21, 0.30], [float(k2_0_12), float(k2_0_21), float(k2_0_30)]
            )(np.clip(thickness_ratio, 0.12, 0.30))
        delta_cd_flaps = flap_chord_contribution * flap_deflection_contribution * area_ratio

        return delta_cd_flaps

    def _compute_delta_cl_airfoil_2d(self, inputs, angle: float, mach: float) -> float:
        """
        Compute airfoil 2D lift contribution.

        :param angle: airfoil angle (in Degree).
        :param mach: air speed.
        :return: increment of lift coefficient.
        """

        flap_type = inputs["data:geometry:flap_type"]
        flap_chord_ratio = float(inputs["data:geometry:flap:chord_ratio"])
        wing_thickness_ratio = float(inputs["data:geometry:wing:thickness_ratio"])
        cl_alpha_airfoil_wing = inputs["data:aerodynamics:wing:airfoil:CL_alpha"]

        # 2D flap lift coefficient
        if flap_type == 1:  # Slotted flap
            alpha_flap = self.k_prime_single_slotted(angle, flap_chord_ratio)
            delta_cl_airfoil = (
                2 * math.pi / math.sqrt(1 - mach ** 2) * alpha_flap * (angle * math.pi / 180)
            )
        else:  # Plain flap
            cl_delta_theory = self.cl_delta_theory_plain_flap(
                wing_thickness_ratio, flap_chord_ratio
            )
            k = self.k_prime_plain_flap(abs(angle), flap_chord_ratio)
            k_cl_delta = self.k_cl_delta_plain_flap(
                wing_thickness_ratio, cl_alpha_airfoil_wing, flap_chord_ratio
            )
            delta_cl_airfoil = k_cl_delta * cl_delta_theory * k * (angle * math.pi / 180)

        return delta_cl_airfoil

    def _compute_delta_cl_max_flaps(self, inputs, flap_angle) -> float:
        """
        Method from Roskam vol.6.  Particularised for single slotted flaps in
        airfoils with 12% thickness (which is the design case); with
        chord ratio of 0.25 and typical flap deflections (30deg landing, 10deg TO).
        Plain flap included (40 deg landing deflection here).
        """

        flap_type = inputs["data:geometry:flap_type"]
        el_aero = inputs["data:geometry:wing:thickness_ratio"]
        flap_chord_ratio = inputs["data:geometry:flap:chord_ratio"]
        sweep_25 = inputs["data:geometry:wing:sweep_25"] * math.pi / 180.0
        flap_area_ratio = self._compute_flap_area_ratio(inputs)

        base_increment = self.base_max_lift_increment(el_aero * 100.0, flap_type)
        flap_chord_factor = self.k1_max_lift(flap_chord_ratio * 100.0, flap_type)
        flap_angle_factor = self.k2_max_lift(flap_angle, flap_type)
        flap_motion_factor = self.k3_max_lift(flap_angle, flap_type)

        k_planform = (1.0 - 0.08 * math.cos(sweep_25) ** 2.0) * math.cos(sweep_25) ** (3.0 / 4.0)
        delta_cl_max_flaps = (
            base_increment
            * flap_chord_factor
            * flap_angle_factor
            * flap_motion_factor
            * k_planform
            * flap_area_ratio
        )

        return delta_cl_max_flaps

    @staticmethod
    def _compute_flap_area_ratio(inputs) -> float:
        """
        Compute ratio of flap over wing (reference area).
        Takes into account the wing portion under the fuselage.
        """

        wing_span = inputs["data:geometry:wing:span"]
        wing_area = inputs["data:geometry:wing:area"]
        wing_taper_ratio = inputs["data:geometry:wing:taper_ratio"]
        y1_wing = inputs["data:geometry:fuselage:maximum_width"] / 2.0
        y2_wing = inputs["data:geometry:wing:root:y"]
        wing_root_chord = inputs["data:geometry:wing:root:chord"]
        flap_span_ratio = inputs["data:geometry:flap:span_ratio"]

        flap_area = (y2_wing - y1_wing) * wing_root_chord + flap_span_ratio * (
            wing_span / 2.0 - y2_wing
        ) * (wing_root_chord * (2 - (1 - wing_taper_ratio) * flap_span_ratio)) * 0.5

        flap_area_ratio = 2 * flap_area / wing_area

        return flap_area_ratio
