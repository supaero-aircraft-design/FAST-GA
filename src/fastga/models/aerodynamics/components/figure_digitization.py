"""
Generic class containing all the digitization needed to compute the aerodynamic
coefficient of the aircraft.
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

import logging
import functools
import os.path as pth

import numpy as np
import openmdao.api as om
from pandas import read_csv
from scipy import interpolate

from . import resources

K_PLAIN_FLAP = "k_plain_flap.csv"
K_DELTA = "k_delta.csv"
K_AR_FUSELAGE = "k_ar_fuselage.csv"
CL_BETA_SWEEP = "cl_beta_sweep_contribution.csv"
K_M_LAMBDA = "sweep_compressibility_correction.csv"
CL_BETA_AR = "cl_beta_ar_contribution.csv"
CL_BETA_GAMMA = "cl_beta_dihedral_contribution.csv"
K_ROLL_DAMPING = "cl_p_roll_damping_parameter.csv"
CL_R_LIFT_PART_A = "cl_r_lift_effect_part_a.csv"
CL_R_LIFT_PART_B = "cl_r_lift_effect_part_b.csv"
CN_DELTA_A_K_A = "cn_delta_a_correlation_cst.csv"
CN_R_LIFT_EFFECT = "cn_r_lift_effect.csv"
CN_R_DRAG_EFFECT = "cn_r_drag_effect.csv"

_LOGGER = logging.getLogger(__name__)


class FigureDigitization(om.ExplicitComponent):
    """Provides lift and drag increments due to high-lift devices."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.phase = None

    @staticmethod
    @functools.lru_cache(maxsize=128)
    def delta_cd_plain_flap(chord_ratio, control_deflection) -> float:
        """
        Surrogate model based on Roskam data to account for the profile drag increment due to the
        deployment of plain flap (figure 4.44).

        :param chord_ratio: control surface over lifting surface ratio.
        :param control_deflection: control surface deflection, in deg.
        :return delta_cd_flap: profile drag increment due to the deployment of flaps.
        """

        if chord_ratio != np.clip(chord_ratio, 0.1, 0.3):
            _LOGGER.warning("Chord ratio outside of the range in Roskam's book, value clipped")

        if control_deflection != np.clip(control_deflection, 0.0, 60.0):
            _LOGGER.warning(
                "Control surface deflection outside of the range in Roskam's book, value clipped"
            )

        control_deflection = np.clip(control_deflection, 0.0, 60.0)
        chord_ratio = np.clip(chord_ratio, 0.1, 0.3)

        delta_cd_flap = (
            -0.00203
            + 0.0001448 * control_deflection ** 2.0 * chord_ratio
            + 0.01083 * control_deflection * chord_ratio ** 2.0
            + 0.08919 * chord_ratio
            - 0.42947 * chord_ratio ** 2.0
            + 0.72158 * chord_ratio ** 3.0
        )

        return delta_cd_flap

    @staticmethod
    @functools.lru_cache(maxsize=128)
    def k_prime_plain_flap(flap_angle, chord_ratio):
        """
        Roskam data to estimate the correction factor to estimate non linear lift behaviour of
        plain flap (figure 8.13).

        :param flap_angle: the flap angle (in °).
        :param chord_ratio: flap chord over wing chord ratio.
        :return k_prime: correction factor to estimate non linear lift behaviour of plain flap.
        """

        file = pth.join(resources.__path__[0], K_PLAIN_FLAP)
        db = read_csv(file)

        x_10 = db["X_10"]
        y_10 = db["Y_10"]
        errors = np.logical_or(np.isnan(x_10), np.isnan(y_10))
        x_10 = x_10[np.logical_not(errors)].tolist()
        y_10 = y_10[np.logical_not(errors)].tolist()
        x_15 = db["X_15"]
        y_15 = db["Y_15"]
        errors = np.logical_or(np.isnan(x_15), np.isnan(y_15))
        x_15 = x_15[np.logical_not(errors)].tolist()
        y_15 = y_15[np.logical_not(errors)].tolist()
        x_25 = db["X_25"]
        y_25 = db["Y_25"]
        errors = np.logical_or(np.isnan(x_25), np.isnan(y_25))
        x_25 = x_25[np.logical_not(errors)].tolist()
        y_25 = y_25[np.logical_not(errors)].tolist()
        x_30 = db["X_30"]
        y_30 = db["Y_30"]
        errors = np.logical_or(np.isnan(x_30), np.isnan(y_30))
        x_30 = x_30[np.logical_not(errors)].tolist()
        y_30 = y_30[np.logical_not(errors)].tolist()
        x_40 = db["X_40"]
        y_40 = db["Y_40"]
        errors = np.logical_or(np.isnan(x_40), np.isnan(y_40))
        x_40 = x_40[np.logical_not(errors)].tolist()
        y_40 = y_40[np.logical_not(errors)].tolist()
        x_50 = db["X_50"]
        y_50 = db["Y_50"]
        errors = np.logical_or(np.isnan(x_50), np.isnan(y_50))
        x_50 = x_50[np.logical_not(errors)].tolist()
        y_50 = y_50[np.logical_not(errors)].tolist()
        k_chord10 = interpolate.interp1d(x_10, y_10)
        k_chord15 = interpolate.interp1d(x_15, y_15)
        k_chord25 = interpolate.interp1d(x_25, y_25)
        k_chord30 = interpolate.interp1d(x_30, y_30)
        k_chord40 = interpolate.interp1d(x_40, y_40)
        k_chord50 = interpolate.interp1d(x_50, y_50)

        if (
            (flap_angle != np.clip(flap_angle, min(x_10), max(x_10)))
            or (flap_angle != np.clip(flap_angle, min(x_15), max(x_15)))
            or (flap_angle != np.clip(flap_angle, min(x_25), max(x_25)))
            or (flap_angle != np.clip(flap_angle, min(x_30), max(x_30)))
            or (flap_angle != np.clip(flap_angle, min(x_40), max(x_40)))
            or (flap_angle != np.clip(flap_angle, min(x_50), max(x_50)))
        ):
            _LOGGER.warning("Flap angle value outside of the range in Roskam's book, value clipped")

        k_chord = [
            float(k_chord10(np.clip(flap_angle, min(x_10), max(x_10)))),
            float(k_chord15(np.clip(flap_angle, min(x_15), max(x_15)))),
            float(k_chord25(np.clip(flap_angle, min(x_25), max(x_25)))),
            float(k_chord30(np.clip(flap_angle, min(x_30), max(x_30)))),
            float(k_chord40(np.clip(flap_angle, min(x_40), max(x_40)))),
            float(k_chord50(np.clip(flap_angle, min(x_50), max(x_50)))),
        ]

        if chord_ratio != np.clip(chord_ratio, 0.1, 0.5):
            _LOGGER.warning(
                "Chord ratio value outside of the range in Roskam's book, value clipped"
            )

        k_prime = float(
            interpolate.interp1d([0.1, 0.15, 0.25, 0.3, 0.4, 0.5], k_chord)(
                np.clip(chord_ratio, 0.1, 0.5)
            )
        )

        return k_prime

    @staticmethod
    @functools.lru_cache(maxsize=128)
    def cl_delta_theory_plain_flap(thickness, chord_ratio):
        """
        Surrogate model based on Roskam data to estimate the theoretical airfoil lift
        effectiveness of a plain flap (figure 8.14).

        :param thickness: the airfoil thickness.
        :param chord_ratio: flap chord over wing chord ratio.
        :return cl_delta: theoretical airfoil lift effectiveness of the plain flap.
        """

        if chord_ratio != np.clip(chord_ratio, 0.05, 0.5):
            _LOGGER.warning(
                "Chord ratio value outside of the range in Roskam's book, value clipped"
            )

        if thickness != np.clip(thickness, 0.0, 0.15):
            _LOGGER.warning(
                "Thickness ratio value outside of the range in Roskam's book, value clipped"
            )

        thickness = np.clip(thickness, 0.0, 0.15)
        chord_ratio = np.clip(chord_ratio, 0.05, 0.5)

        cl_delta_th = (
            0.39403
            + 22.40938 * chord_ratio
            - 76.09814 * chord_ratio ** 2.0
            - 16.06188 * thickness * chord_ratio
            + 205.57323 * chord_ratio ** 3.0
            + 1.75898 * thickness
            - 312.43784 * chord_ratio ** 4.0
            + 192.52783 * chord_ratio ** 5.0
            + 329.31357 * thickness * chord_ratio ** 4.0
            - 6702.89145 * thickness ** 4.0 * chord_ratio
            + 1273.34256 * thickness ** 3.0 * chord_ratio
            - 296.20629 * thickness ** 2.0 * chord_ratio ** 2.0
            + 616.07146 * thickness ** 3.0 * chord_ratio ** 2.0
            - 401.79093 * thickness * chord_ratio ** 3.0
            + 170.05976 * thickness * chord_ratio ** 2.0
            + 92.84439 * thickness ** 5.0
            - 17.93677 * thickness ** 2.0
            + 251.43186 * thickness ** 2.0 * chord_ratio ** 3.0
            + 0.39403
            + 455.44427 * thickness ** 4.0
        )

        return cl_delta_th

    @staticmethod
    @functools.lru_cache(maxsize=128)
    def k_cl_delta_plain_flap(thickness_ratio, airfoil_lift_coefficient, chord_ratio):
        """
        Surrogate model based on Roskam data to estimate the correction factor to estimate
        difference from theoretical plain flap lift (figure 8.15).

        :param thickness_ratio: airfoil thickness ratio.
        :param airfoil_lift_coefficient: the lift coefficient of the airfoil, in rad**-1.
        :param chord_ratio: flap chord over wing chord ratio.
        :return k_cl_delta: correction factor to account for difference from theoretical plain
        flap lift.
        """

        # Figure 10.64 b
        cl_alpha_th = 6.3 + np.clip(thickness_ratio, 0.0, 0.2) / 0.2 * (7.3 - 6.3)

        if float(airfoil_lift_coefficient / cl_alpha_th) != np.clip(
            float(airfoil_lift_coefficient / cl_alpha_th), 0.7, 1.0
        ):
            _LOGGER.warning(
                "Airfoil lift slope ratio value outside of the range in Roskam's book, "
                "value clipped"
            )

        if chord_ratio != np.clip(chord_ratio, 0.05, 0.5):
            _LOGGER.warning(
                "Chord ratio value outside of the range in Roskam's book, value clipped"
            )

        chord_ratio = np.clip(chord_ratio, 0.05, 0.5)
        k_cl_alpha = np.clip(airfoil_lift_coefficient / cl_alpha_th, 0.7, 1.0)

        k_cl_delta = 10.0 ** (
            0.00014
            - 0.39775 * np.log10(chord_ratio) * np.log10(k_cl_alpha) ** 2.0
            + 0.96887 * np.log10(k_cl_alpha)
            - 1.06182 * np.log10(chord_ratio) * np.log10(k_cl_alpha)
            - 15.84119 * np.log10(k_cl_alpha) ** 3.0
            + 4.70862 * np.log10(chord_ratio) ** 2.0 * np.log10(k_cl_alpha) ** 3.0
            + 90.62652 * np.log10(chord_ratio) * np.log10(k_cl_alpha) ** 6.0
            - 32881.41822 * np.log10(k_cl_alpha) ** 7.0
            - 328.18865 * np.log10(k_cl_alpha) ** 4.0
            + 0.00009 * np.log10(chord_ratio) ** 7.0
        )

        return k_cl_delta

    @staticmethod
    @functools.lru_cache(maxsize=128)
    def k_prime_single_slotted(flap_angle, chord_ratio):
        """
        Surrogate model based on Roskam data to estimate the lift effectiveness of a single slotted
        flap (figure 8.17), noted here k_prime to match the notation of the plain flap but is written
        alpha_delta in the book.

        :param flap_angle: the control surface deflection angle angle (in °).
        :param chord_ratio: control surface chord over lifting surface chord ratio.
        :return k_prime: lift effectiveness factor of a single slotted flap.
        """

        if float(chord_ratio) != np.clip(float(chord_ratio), 0.15, 0.4):
            _LOGGER.warning(
                "Chord ratio value outside of the range in Roskam's book, value clipped"
            )

        if float(flap_angle) != np.clip(float(flap_angle), 0.0, 80.0):
            _LOGGER.warning("Flap angle value outside of the range in Roskam's book, value clipped")

        chord_ratio = np.clip(float(chord_ratio), 0.15, 0.4)
        flap_angle = np.clip(flap_angle, 0.0, 80.0) * 0.01
        # flap angle is scaled by 0.01 to ease the surrogate modeling process

        k_prime = (
            0.10882
            + 0.43162 * flap_angle
            + 1.81636 * chord_ratio
            + 23.23625 * chord_ratio ** 3.0 * flap_angle ** 2.0
            - 59.51140 * flap_angle ** 6.0
            - 6.54618 * flap_angle ** 2.0
            - 3.64380 * chord_ratio ** 3.0
            - 0.33308 * chord_ratio * flap_angle ** 2.0
            - 3.60115 * chord_ratio ** 3.0 * flap_angle
            + 126.71624 * flap_angle ** 5.0
            - 98.81282 * flap_angle ** 4.0
            + 35.78080 * flap_angle ** 3.0
            + 17.85760 * chord_ratio * flap_angle ** 5.0
            - 16.51125 * chord_ratio * flap_angle ** 4.0
            - 18.63048 * chord_ratio ** 3.0 * flap_angle ** 3.0
        )

        return k_prime

    @staticmethod
    @functools.lru_cache(maxsize=128)
    def base_max_lift_increment(thickness_ratio: float, flap_type: float) -> float:
        """
        Surrogate model based on Roskam data to estimate base lift increment used in the computation
        of flap delta_cl_max (figure 8.31).

        :param thickness_ratio: thickness ratio f the lifting surface, in %.
        :param flap_type: type of flap used as described in Roskam, for now can be 0.0 for plain,
        1.0 for single_slot.
        :return: delta_cl_base.
        """

        if flap_type == 0.0:
            if thickness_ratio != np.clip(thickness_ratio, 0.97561, 19.96748):
                _LOGGER.warning(
                    "Thickness ratio value outside of the range in Roskam's book, value clipped"
                )

            thickness_ratio = np.clip(thickness_ratio, 0.97561, 19.96748) / 2.0
            # thickness_ratio is scaled to ease the surrogate modeling process

            delta_cl_max_base = 10.0 ** (
                -0.00855
                - 4184.27800 * np.log10(thickness_ratio) ** 8.0
                - 3.01891 * np.log10(thickness_ratio)
                + 1043.18454 * np.log10(thickness_ratio) ** 15.0
                - 200.83494 * np.log10(thickness_ratio) ** 3.0
                - 3054.84080 * np.log10(thickness_ratio) ** 14.0
                + 1614.23260 * np.log10(thickness_ratio) ** 5.0
                + 7913.55910 * np.log10(thickness_ratio) ** 7.0
                + 43.50709 * np.log10(thickness_ratio) ** 2.0
                + 151.14820 * np.log10(thickness_ratio) ** 4.0
                + 2425.33784 * np.log10(thickness_ratio) ** 13.0
                - 5747.78286 * np.log10(thickness_ratio) ** 6.0
            )

        elif flap_type == 1.0:
            if thickness_ratio != np.clip(thickness_ratio, 0.97561, 19.96748):
                _LOGGER.warning(
                    "Thickness ratio value outside of the range in Roskam's book, value clipped"
                )

            thickness_ratio = np.clip(thickness_ratio, 0.97561, 19.96748) / 2.0
            # thickness_ratio is scaled to ease the surrogate modeling process

            delta_cl_max_base = 10.0 ** (
                0.00401
                + 0.53332 * np.log10(thickness_ratio) ** 4.0
                + 22.42966 * np.log10(thickness_ratio) ** 20.0
                - 0.20865 * np.log10(thickness_ratio) ** 5.0
                - 45.52539 * np.log10(thickness_ratio) ** 19.0
                + 23.05475 * np.log10(thickness_ratio) ** 18.0
                - 0.06516 * np.log10(thickness_ratio) ** 2.0
            )

        else:
            _LOGGER.warning("Flap type not recognized, used plain flap instead")
            if thickness_ratio != np.clip(thickness_ratio, 0.97561, 19.96748):
                _LOGGER.warning(
                    "Thickness ratio value outside of the range in Roskam's book, value clipped"
                )

            thickness_ratio = np.clip(thickness_ratio, 0.97561, 19.96748) / 2.0
            # thickness_ratio is scaled to ease the surrogate modeling process

            delta_cl_max_base = 10.0 ** (
                -0.00855
                - 4184.27800 * np.log10(thickness_ratio) ** 8.0
                - 3.01891 * np.log10(thickness_ratio)
                + 1043.18454 * np.log10(thickness_ratio) ** 15.0
                - 200.83494 * np.log10(thickness_ratio) ** 3.0
                - 3054.84080 * np.log10(thickness_ratio) ** 14.0
                + 1614.23260 * np.log10(thickness_ratio) ** 5.0
                + 7913.55910 * np.log10(thickness_ratio) ** 7.0
                + 43.50709 * np.log10(thickness_ratio) ** 2.0
                + 151.14820 * np.log10(thickness_ratio) ** 4.0
                + 2425.33784 * np.log10(thickness_ratio) ** 13.0
                - 5747.78286 * np.log10(thickness_ratio) ** 6.0
            )

        return delta_cl_max_base

    @staticmethod
    @functools.lru_cache(maxsize=128)
    def k1_max_lift(chord_ratio, flap_type) -> float:
        """
        Surrogate model based on Roskam data to correct the base lift increment to account for
        chord ratio difference wrt to the reference flap configuration (figure 8.32).

        :param chord_ratio: ration of the chord of the control surface over that of the whole
        surface, in %.
        :param flap_type: type of flap used as described in Roskam, for now can be 0.0 for plain,
        1.0 for single_slot.

        :return k1: correction factor to account for chord ratio difference wrt the reference
        configuration.
        """

        if flap_type != 1.0 and flap_type != 0.0:
            _LOGGER.warning("Flap type not recognized, used plain flap instead")

        if float(chord_ratio) != np.clip(float(chord_ratio), 0.0, 30.0):
            _LOGGER.warning(
                "Chord ratio value outside of the range in Roskam's book, value clipped"
            )

        chord_ratio = np.clip(float(chord_ratio), 0.0, 30.0)
        k1 = (
            -1.47218e-6 * chord_ratio ** 4.0
            + 1.29646e-4 * chord_ratio ** 3.0
            - 4.77302e-3 * chord_ratio ** 2.0
            + 1.01777e-1 * chord_ratio
            - 2.72363e-03
        )

        return k1

    @staticmethod
    @functools.lru_cache(maxsize=128)
    def k2_max_lift(angle, flap_type) -> float:
        """
        Surrogate model based on Roskam data to correct the base lift increment to account for
        the control surface deflection angle difference wrt to the reference flap configuration
        (figure 8.33).

        :param angle: control surface deflection angle, in °.
        :param flap_type: type of flap used as described in Roskam, for now can be 0.0 for plain,
        1.0 for single_slot.
        :return k2: correction factor to account for the control surface deflection angle wrt the
        reference configuration.
        """

        if flap_type == 0.0:
            if angle != np.clip(angle, 0.0, 60.0):
                _LOGGER.warning(
                    "Control surface deflection value outside of the range in Roskam's book, "
                    "value clipped"
                )
            angle = np.clip(angle, 0.0, 60.0)
            k2 = (
                3.51465e-8 * angle ** 4.0
                - 1.99067e-6 * angle ** 3.0
                - 3.26243e-4 * angle ** 2.0
                + 3.58225e-2 * angle
                - 1.14952e-03
            )

        elif flap_type == 1.0:
            if angle != np.clip(angle, 0.0, 45.0):
                _LOGGER.warning(
                    "Control surface deflection value outside of the range in Roskam's book, "
                    "value clipped"
                )
            angle = np.clip(angle, 0.0, 45.0)
            k2 = (
                3.16011e-8 * angle ** 4.0
                - 3.44449e-6 * angle ** 3.0
                - 1.85964e-4 * angle ** 2.0
                + 3.07683e-2 * angle
                + 1.76491e-01
            )

        else:
            _LOGGER.warning("Flap type not recognized, used plain flap instead")
            if angle != np.clip(angle, 0.0, 60.0):
                _LOGGER.warning(
                    "Control surface deflection value outside of the range in Roskam's book, "
                    "value clipped"
                )
            angle = np.clip(angle, 0.0, 60.0)
            angle = (
                3.51465e-8 * angle ** 4.0
                - 1.99067e-6 * angle ** 3.0
                - 3.26243e-4 * angle ** 2.0
                + 3.58225e-2 * angle
                - 1.14952e-03
            )

        return k2

    @staticmethod
    @functools.lru_cache(maxsize=128)
    def k3_max_lift(angle, flap_type) -> float:
        """
        Surrogate model based on Roskam data for flap motion correction factor (figure 8.34).

        :param angle: control surface deflection angle, in °.
        :param flap_type: type of flap used as described in Roskam, for now can be 0.0 for plain,
        1.0 for single_slot.
        :return k3: correction factor to account flap motion correction.
        """

        if flap_type == 0.0:
            k3 = 1.0
        elif flap_type == 1.0:
            reference_angle = 45.0
            if float(angle / reference_angle) != np.clip(float(angle / reference_angle), 0.0, 1.0):
                _LOGGER.warning(
                    "Control surface deflection value outside of the range in Roskam's book, "
                    "value clipped, reference value is %f",
                    reference_angle,
                )
            angle_ratio = np.clip(float(angle / reference_angle), 0.0, 1.0)
            k3 = (
                0.722572 * angle_ratio ** 4.0
                - 1.58742 * angle_ratio ** 3.0
                + 0.701810 * angle_ratio ** 2.0
                + 1.14872 * angle_ratio
                + 1.57847e-02
            )

        else:
            _LOGGER.warning("Flap type not recognized, used plain flap instead")
            k3 = 1.0

        return k3

    @staticmethod
    @functools.lru_cache(maxsize=128)
    def k_b_flaps(eta_in: float, eta_out: float, taper_ratio: float) -> float:
        """
        Surrogate model based on Roskam data to estimate the flap span factor Kb (figure 8.52)
        This factor accounts for a finite flap contribution to the 3D lift increase,
        depending on its position and size and the taper ratio of the wing.

        :param eta_in: position along the wing span of the start of the flaps divided by span.
        :param eta_out: position along the wing span of the end of the flaps divided by span.
        :param taper_ratio: taper ration of the surface.
        :return: kb factor contribution to 3D lift.
        """

        eta_in = float(eta_in)
        eta_out = float(eta_out)
        if taper_ratio != np.clip(taper_ratio, 0.0, 1.0):
            _LOGGER.warning(
                "Taper ratio value outside of the range in Roskam's book, value clipped"
            )

        if eta_in != np.clip(eta_in, 0.0, 1.0):
            _LOGGER.warning(
                "Flap inward position ratio value outside of the range in Roskam's book, "
                "value clipped"
            )

        if eta_out != np.clip(eta_out, 0.0, 1.0):
            _LOGGER.warning(
                "Flap inward position ratio value outside of the range in Roskam's book, "
                "value clipped"
            )

        taper_ratio = np.clip(taper_ratio, 0.0, 1.0)

        eta_in = np.clip(eta_in, 0.0, 1.0) + 1.0
        eta_out = np.clip(eta_out, 0.0, 1.0) + 1.0
        # eta_in and eta_out are translated to ease the surrogate modeling process

        k_b_in = _k_b(taper_ratio, eta_in)
        k_b_out = _k_b(taper_ratio, eta_out)

        return float(k_b_out - k_b_in)

    @staticmethod
    @functools.lru_cache(maxsize=128)
    def a_delta_airfoil(chord_ratio) -> float:
        """
        Surrogate model based on Roskam data to estimate the two-dimensional flap effectiveness
        factor (figure 8.53a) This factor can be then used in the computation of the 3D flap
        effectiveness factor which is often the coefficient of interest.

        :param chord_ratio: ration of the chord of the control surface over that of the whole
        surface.
        :return: kb factor contribution to 3D lift.
        """

        if chord_ratio != np.clip(chord_ratio, 0.0, 1.0):
            _LOGGER.warning(
                "Chord ratio value outside of the range in Roskam's book, value clipped"
            )

        chord_ratio = (np.clip(float(chord_ratio), 0.0, 1.0) + 1.0) / 2.0
        # chord_ratio is translated and scaled just to ease the surrogate modeling process

        a_delta = (
            2.0
            * 10.0
            ** (
                0.00203
                - 41.69445 * np.log10(chord_ratio) ** 3.0
                - 445675063.98145 * np.log10(chord_ratio) ** 12.0
                - 6.50632 * np.log10(chord_ratio) ** 2.0
                - 482857663.42795 * np.log10(chord_ratio) ** 11.0
                - 199107455.19802 * np.log10(chord_ratio) ** 10.0
                - 37297404.99141 * np.log10(chord_ratio) ** 9.0
                - 2709421.52197 * np.log10(chord_ratio) ** 8.0
            )
            - 1.0
        )

        return a_delta

    @staticmethod
    @functools.lru_cache(maxsize=128)
    def k_a_delta(a_delta_airfoil, aspect_ratio) -> float:
        """
        Surrogate model based on Roskam data to estimate the two dimensional to three dimensional
        control surface lift effectiveness parameter (figure 8.53b).

        :param a_delta_airfoil: control surface two-dimensional flap effectiveness factor.
        :param aspect_ratio: aspect ratio of the fixed surface.
        :return k_a_delta: two dimensional to three dimensional control surface lift effectiveness
        parameter.
        """

        if float(aspect_ratio) != np.clip(float(aspect_ratio), 0.0, 10.0):
            _LOGGER.warning(
                "Aspect ratio value outside of the range in Roskam's book, value clipped"
            )

        if a_delta_airfoil != np.clip(a_delta_airfoil, 0.1, 1.0):
            _LOGGER.warning(
                "Control surface effectiveness ratio value outside of the range in "
                "Roskam's book, value clipped"
            )

        aspect_ratio = np.clip(aspect_ratio, 0.0, 10.0) * 0.1 + 1.0
        a_delta_airfoil = np.clip(a_delta_airfoil, 0.1, 1.0) + 1.0
        # aspect_ratio and a_delta_airfoil are translated and scaled just to ease the surrogate modeling process

        k_a_delta = 10.0 ** (
            0.84171
            - 1.67074 * np.log10(a_delta_airfoil) * np.log10(aspect_ratio)
            - 12610.80614 * np.log10(a_delta_airfoil) ** 2.0 * np.log10(aspect_ratio) ** 2.0
            - 6.79137 * np.log10(a_delta_airfoil)
            - 13.89834 * np.log10(aspect_ratio)
            - 251937.79952 * np.log10(a_delta_airfoil) ** 3.0 * np.log10(aspect_ratio) ** 3.0
            - 380640.28988 * np.log10(a_delta_airfoil) ** 4.0 * np.log10(aspect_ratio) ** 4.0
            + 264.68115 * np.log10(aspect_ratio) ** 2.0
            + 1161.59281 * np.log10(a_delta_airfoil) ** 2.0 * np.log10(aspect_ratio)
            - 3479.56140 * np.log10(aspect_ratio) ** 3.0
            + 2592.40455 * np.log10(a_delta_airfoil) ** 2.0 * np.log10(aspect_ratio) ** 4.0
            + 27657.09600 * np.log10(aspect_ratio) ** 4.0
            - 124736.18274 * np.log10(a_delta_airfoil) ** 2.0 * np.log10(aspect_ratio) ** 5.0
            - 134260.92969 * np.log10(aspect_ratio) ** 5.0
            + 113920.17524 * np.log10(a_delta_airfoil) ** 2.0 * np.log10(aspect_ratio) ** 6.0
            - 8823.60640 * np.log10(a_delta_airfoil) ** 3.0 * np.log10(aspect_ratio)
            - 204.54218 * np.log10(a_delta_airfoil) * np.log10(aspect_ratio) ** 2.0
            + 120306.00346 * np.log10(a_delta_airfoil) ** 7.0 * np.log10(aspect_ratio)
            + 7804.57981 * np.log10(a_delta_airfoil) * np.log10(aspect_ratio) ** 3.0
            + 392202.02583 * np.log10(aspect_ratio) ** 6.0
            + 33762.73280 * np.log10(a_delta_airfoil) ** 2.0 * np.log10(aspect_ratio) ** 3.0
            - 635236.22141 * np.log10(aspect_ratio) ** 7.0
            - 60732.94975 * np.log10(a_delta_airfoil) * np.log10(aspect_ratio) ** 4.0
            + 92171.07536 * np.log10(a_delta_airfoil) ** 3.0 * np.log10(aspect_ratio) ** 2.0
            + 19420.77192 * np.log10(a_delta_airfoil) ** 3.0 * np.log10(aspect_ratio) ** 5.0
            + 215063.41304 * np.log10(a_delta_airfoil) * np.log10(aspect_ratio) ** 5.0
            + 225469.01479 * np.log10(a_delta_airfoil) ** 3.0 * np.log10(aspect_ratio) ** 4.0
            + 438721.38201 * np.log10(aspect_ratio) ** 8.0
            + 23967.79400 * np.log10(a_delta_airfoil) ** 4.0 * np.log10(aspect_ratio)
            - 100510.87507 * np.log10(a_delta_airfoil) ** 6.0 * np.log10(aspect_ratio)
            + 262794.28735 * np.log10(a_delta_airfoil) * np.log10(aspect_ratio) ** 7.0
            - 142671.53143 * np.log10(a_delta_airfoil) ** 6.0 * np.log10(aspect_ratio) ** 2.0
            - 439178.68850 * np.log10(a_delta_airfoil) ** 5.0 * np.log10(aspect_ratio) ** 3.0
            - 280635.47114 * np.log10(a_delta_airfoil) ** 4.0 * np.log10(aspect_ratio) ** 2.0
            + 598274.37658 * np.log10(a_delta_airfoil) ** 4.0 * np.log10(aspect_ratio) ** 3.0
            + 893.12627 * np.log10(a_delta_airfoil) ** 8.0
            + 26.89563 * np.log10(a_delta_airfoil) ** 2.0
            + 358478.23331 * np.log10(a_delta_airfoil) ** 5.0 * np.log10(aspect_ratio) ** 2.0
            - 373439.21464 * np.log10(a_delta_airfoil) * np.log10(aspect_ratio) ** 6.0
            - 47.45450 * np.log10(a_delta_airfoil) ** 3.0
        )

        return k_a_delta

    @staticmethod
    @functools.lru_cache(maxsize=128)
    def x_cp_c_prime(flap_chord_ratio: float) -> float:
        """
        Surrogate model based on Roskam data to estimate the location of the center of pressure
        due to Incremental Flap Load (figure 8.91).

        :param flap_chord_ratio: ratio of the control surface chord over the lifting surface chord.
        :return x_cp_c_prime: location of center of pressure due to flap deployment.
        """

        # Graph is simple so no csv is read, rather, a direct formula is used.

        if flap_chord_ratio != np.clip(flap_chord_ratio, 0.0, 1.0):
            _LOGGER.warning("Chord ratio outside of the range in Roskam's book, value clipped")

        x_cp_c_prime = -0.25 * np.clip(flap_chord_ratio, 0.0, 1.0) + 0.5

        return x_cp_c_prime

    @staticmethod
    @functools.lru_cache(maxsize=128)
    def k_p_flaps(taper_ratio, eta_in, eta_out) -> float:
        """
        Surrogate model based on Roskam data to account for the partial span flaps factor on
        the pitch moment coefficient (figure 8.105).

        :param taper_ratio: lifting surface taper ratio.
        :param eta_in: start of the control surface, in percent of the lifting surface span.
        :param eta_out: end of the control surface, in percent of the lifting surface span.
        :return k_p: partial span factor.
        """

        if taper_ratio != np.clip(taper_ratio, 0.25, 1.0):
            _LOGGER.warning("Taper ratio outside of the range in Roskam's book, value clipped")

        taper_ratio = np.clip(taper_ratio, 0.25, 1.0)
        eta_in = np.clip(eta_in, 0.0, 1.0)
        eta_out = np.clip(eta_out, 0.0, 1.0)

        k_p_in = _k_p(taper_ratio, eta_in)
        k_p_out = _k_p(taper_ratio, eta_out)

        return k_p_out - k_p_in

    @staticmethod
    @functools.lru_cache(maxsize=128)
    def pitch_to_reference_lift(thickness_ratio: float, chord_ratio: float) -> float:
        """
        Surrogate model based on Roskam data to account for the ratio between the pitch moment
        coefficient and the reference lift coefficient increment (figure 8.106).

        :param thickness_ratio: thickness to chord ratio of the lifting surface.
        :param chord_ratio: chord ratio of the control surface over the lifting surface.
        :return delta_cm_delta_cl_ref: ration between the pitching moment and the reference lift
        coefficient.
        """

        if chord_ratio != np.clip(chord_ratio, 0.05, 0.4):
            _LOGGER.warning("Chord ratio outside of the range in Roskam's book, value clipped")

        if thickness_ratio != np.clip(thickness_ratio, 0.03, 0.21):
            _LOGGER.warning(
                "Thickness to chord ratio outside of the range in Roskam's book, " "value clipped"
            )

        chord_ratio = np.clip(chord_ratio, 0.05, 0.4)
        thickness_ratio = np.clip(thickness_ratio, 0.03, 0.21)

        k = (
            -0.61953
            + 1.52483 * chord_ratio
            + 1.65991 * thickness_ratio
            + 9.16051 * thickness_ratio * chord_ratio ** 2.0
            - 7.73053 * thickness_ratio * chord_ratio
            - 0.55584 * chord_ratio ** 2.0
            - 2280.12581 * thickness_ratio ** 7.0
            + 3561.86293 * thickness_ratio ** 6.0 * chord_ratio
            + 282.52088 * thickness_ratio ** 4.0 * chord_ratio ** 3.0
            - 2.34977 * chord_ratio ** 3.0
            - 1822.09526 * thickness_ratio ** 5.0 * chord_ratio ** 2.0
            + 5.48933 * chord_ratio ** 6.0
        )

        return k

    @staticmethod
    @functools.lru_cache(maxsize=128)
    def k_delta_flaps(taper_ratio: float, eta_in: float, eta_out: float) -> float:
        """
        Roskam data to estimate the conversion factor which accounts for partial span flaps on a
        swept wing (c_prime/c = 1.0) (figure 8.107).

        :param taper_ratio: lifting surface taper ratio.
        :param eta_in: start of the control surface, in percent of the lifting surface span.
        :param eta_out: end of the control surface, in percent of the lifting surface span.
        :return delta_k: partial span factor.
        """

        file = pth.join(resources.__path__[0], K_DELTA)
        db = read_csv(file)

        x_10 = db["X_1_0"]
        y_10 = db["Y_1_0"]
        errors = np.logical_or(np.isnan(x_10), np.isnan(y_10))
        x_10 = x_10[np.logical_not(errors)].tolist()
        y_10 = y_10[np.logical_not(errors)].tolist()
        eta_in_1_0 = interpolate.interp1d(x_10, y_10)(np.clip(eta_in, min(x_10), max(x_10)))
        eta_out_1_0 = interpolate.interp1d(x_10, y_10)(np.clip(eta_out, min(x_10), max(x_10)))

        x_05 = db["X_0_5"]
        y_05 = db["Y_0_5"]
        errors = np.logical_or(np.isnan(x_05), np.isnan(y_05))
        x_05 = x_05[np.logical_not(errors)].tolist()
        y_05 = y_05[np.logical_not(errors)].tolist()
        eta_in_0_5 = interpolate.interp1d(x_05, y_05)(np.clip(eta_in, min(x_05), max(x_05)))
        eta_out_0_5 = interpolate.interp1d(x_05, y_05)(np.clip(eta_out, min(x_05), max(x_05)))

        x_0333 = db["X_0_333"]
        y_0333 = db["Y_0_333"]
        errors = np.logical_or(np.isnan(x_0333), np.isnan(y_0333))
        x_0333 = x_0333[np.logical_not(errors)].tolist()
        y_0333 = y_0333[np.logical_not(errors)].tolist()
        eta_in_0_333 = interpolate.interp1d(x_0333, y_0333)(
            np.clip(eta_in, min(x_0333), max(x_0333))
        )
        eta_out_0_333 = interpolate.interp1d(x_0333, y_0333)(
            np.clip(eta_out, min(x_0333), max(x_0333))
        )

        x_02 = db["X_0_2"]
        y_02 = db["Y_0_2"]
        errors = np.logical_or(np.isnan(x_02), np.isnan(y_02))
        x_02 = x_02[np.logical_not(errors)].tolist()
        y_02 = y_02[np.logical_not(errors)].tolist()
        eta_in_0_2 = interpolate.interp1d(x_02, y_02)(np.clip(eta_in, min(x_02), max(x_02)))
        eta_out_0_2 = interpolate.interp1d(x_02, y_02)(np.clip(eta_out, min(x_02), max(x_02)))

        taper_array = [0.2, 0.333, 0.5, 1.0]
        eta_in_array = [eta_in_0_2, eta_in_0_333, eta_in_0_5, eta_in_1_0]
        eta_out_array = [eta_out_0_2, eta_out_0_333, eta_out_0_5, eta_out_1_0]

        if taper_ratio != np.clip(taper_ratio, 0.2, 1.0):
            _LOGGER.warning("Taper ratio outside of the range in Roskam's book, value clipped")

        k_delta_in = interpolate.interp1d(taper_array, eta_in_array)(np.clip(taper_ratio, 0.2, 1.0))
        k_delta_out = interpolate.interp1d(taper_array, eta_out_array)(
            np.clip(taper_ratio, 0.2, 1.0)
        )

        k_delta = k_delta_out - k_delta_in

        return k_delta

    @staticmethod
    def k_ar_fuselage(taper_ratio, span, avg_fuselage_depth) -> float:
        """
        Roskam data to account for the effect of the fuselage on the VTP effective aspect ratio (
        figure  10.14).

        :param taper_ratio: lifting surface taper ratio.
        :param span: lifting surface span, in m.
        :param avg_fuselage_depth: average fuselage depth (diameter if fuselage considered
        circular), in m.
        :return k_ar_fuselage: correction factor to account for the end plate effect of the fuselage
         on effective VTP AR.
        """

        file = pth.join(resources.__path__[0], K_AR_FUSELAGE)
        db = read_csv(file)

        x_06 = db["X_06"]
        y_06 = db["Y_06"]
        errors = np.logical_or(np.isnan(x_06), np.isnan(y_06))
        x_06 = x_06[np.logical_not(errors)].tolist()
        y_06 = y_06[np.logical_not(errors)].tolist()

        x_10 = db["X_10"]
        y_10 = db["Y_10"]
        errors = np.logical_or(np.isnan(x_10), np.isnan(y_10))
        x_10 = x_10[np.logical_not(errors)].tolist()
        y_10 = y_10[np.logical_not(errors)].tolist()

        x_value = span / avg_fuselage_depth

        if x_value != np.clip(x_value, min(min(x_06), min(x_10)), max(max(x_06), max(x_10))):
            _LOGGER.warning(
                "Ratio of span on fuselage depth outside of the range in Roskam's book, "
                "value clipped"
            )

        y_value_06 = interpolate.interp1d(x_06, y_06)(np.clip(x_value, min(x_06), max(x_06)))
        y_value_10 = interpolate.interp1d(x_10, y_10)(np.clip(x_value, min(x_10), max(x_10)))

        if taper_ratio != np.clip(taper_ratio, 0.6, 1.0):
            _LOGGER.warning("Taper ratio outside of the range in Roskam's book, value clipped")

        k_ar_fuselage = interpolate.interp1d([0.6, 1.0], [float(y_value_06), float(y_value_10)])(
            np.clip(taper_ratio, 0.6, 1.0)
        )

        return k_ar_fuselage

    @staticmethod
    def k_vh(area_ratio) -> float:
        """
        Surrogate model based on Roskam data to estimate the impact of relative area ratio on
        the effective aspect ratio (figure 10.16).

        :param area_ratio: ratio of the horizontal tail area over the vertical tail area.
        :return k_vh: impact of area ratio on effective aspect ratio.
        """

        if float(area_ratio) != np.clip(float(area_ratio), 0.0, 2.0):
            _LOGGER.warning("Area ratio value outside of the range in Roskam's book, value clipped")

        area_ratio = np.clip(float(area_ratio), 0.0, 2.0)

        k_vh = (
            0.1604 * area_ratio ** 6.0
            - 1.0202 * area_ratio ** 5.0
            + 2.4252 * area_ratio ** 4.0
            - 2.4621 * area_ratio ** 3.0
            + 0.3819 * area_ratio ** 2.0
            + 1.4554 * area_ratio
            - 0.0137
        )

        return k_vh

    @staticmethod
    @functools.lru_cache(maxsize=128)
    def k_ch_alpha(thickness_ratio, airfoil_lift_coefficient, chord_ratio):
        """
        Surrogate model based on Roskam data to compute the correction factor to differentiate
        the 2D control surface hinge moment derivative due to AOA from the reference (figure 10.63).

        :param thickness_ratio: airfoil thickness ratio.
        :param airfoil_lift_coefficient: the lift coefficient of the airfoil, in rad**-1.
        :param chord_ratio: flap chord over wing chord ratio.
        :return k_ch_alpha: correction factor for 2D control surface hinge moment derivative due to
        AOA.
        """

        # Figure 10.64 b
        if thickness_ratio != np.clip(thickness_ratio, 0.0, 0.2):
            _LOGGER.warning(
                "Thickness ratio value outside of the range in Roskam's book, value clipped"
            )
        cl_alpha_th = 6.3 + np.clip(thickness_ratio, 0.0, 0.2) / 0.2 * (7.3 - 6.3)

        if float(airfoil_lift_coefficient / cl_alpha_th) != np.clip(
            float(airfoil_lift_coefficient / cl_alpha_th), 0.7, 1.0
        ):
            _LOGGER.warning(
                "Airfoil lift coefficient to theoretical lift coefficient ratio value outside of "
                "the range in Roskam's book, value clipped"
            )

        chord_ratio = np.clip(chord_ratio, 0.1, 0.4)
        k_cl_alpha = np.clip(float(airfoil_lift_coefficient / cl_alpha_th), 0.7, 1.0)

        k_ch_alpha = (
            10.0
            ** (
                0.30178
                + 0.94741 * np.log10(k_cl_alpha)
                - 116.71747 * np.log10(chord_ratio) * np.log10(k_cl_alpha) ** 3.0
                + 0.00137 * np.log10(chord_ratio)
                - 74.35653 * np.log10(chord_ratio) ** 2.0 * np.log10(k_cl_alpha) ** 3.0
                + 0.38937 * np.log10(chord_ratio) ** 2.0 * np.log10(k_cl_alpha)
                + 0.66420 * np.log10(chord_ratio) ** 2.0 * np.log10(k_cl_alpha) ** 2.0
                + 1.30055 * np.log10(k_cl_alpha) ** 2.0
                + 0.19129 * np.log10(chord_ratio) ** 4.0 * np.log10(k_cl_alpha)
            )
            - 1.0
        )

        return k_ch_alpha

    @staticmethod
    @functools.lru_cache(maxsize=128)
    def ch_alpha_th(thickness_ratio, chord_ratio):
        """
        Surrogate model based on Roskam data to compute the theoretical 2D control surface
        hinge moment derivative due to AOA (figure 10.63).

        :param thickness_ratio: airfoil thickness ratio.
        :param chord_ratio: flap chord over wing chord ratio.
        :return ch_alpha: theoretical hinge moment derivative due to AOA.
        """

        if chord_ratio != np.clip(chord_ratio, 0.1, 0.4):
            _LOGGER.warning(
                "Chord ratio value outside of the range in Roskam's book, value clipped"
            )

        if thickness_ratio != np.clip(thickness_ratio, 0.0, 0.16):
            _LOGGER.warning(
                "Thickness ratio value outside of the range in Roskam's book, value clipped"
            )

        chord_ratio = np.clip(chord_ratio, 0.1, 0.4)
        thickness_ratio = np.clip(thickness_ratio, 0.0, 0.16)

        ch_alpha_th = (
            -0.24651
            - 0.21079 * chord_ratio ** 3.0
            + 1.14770 * thickness_ratio
            - 1.04870 * chord_ratio ** 2.0 * thickness_ratio
            - 539.59855 * thickness_ratio ** 4.0
            + 159.55277 * thickness_ratio ** 3.0
            - 13.82122 * thickness_ratio ** 2.0
            - 0.08529 * chord_ratio ** 4.0
            - 0.50251 * chord_ratio ** 2.0
            - 1.01159 * chord_ratio
            - 0.44413 * chord_ratio ** 3.0 * thickness_ratio
        )

        return ch_alpha_th

    @staticmethod
    @functools.lru_cache(maxsize=128)
    def k_ch_delta(thickness_ratio, airfoil_lift_coefficient, chord_ratio):
        """
        Surrogate model based on Roskam data to compute the correction factor to differentiate
        the 2D control surface hinge moment derivative due to control surface deflection from
        the reference (figure 10.69 a).

        :param thickness_ratio: airfoil thickness ratio.
        :param airfoil_lift_coefficient: the lift coefficient of the airfoil, in rad**-1.
        :param chord_ratio: control surface chord over lifting surface chord ratio.
        :return k_ch_delta: hinge moment derivative due to control surface deflection correction
        factor.
        """

        # Figure 10.64 b
        if thickness_ratio != np.clip(thickness_ratio, 0.0, 0.2):
            _LOGGER.warning(
                "Thickness ratio value outside of the range in Roskam's book, value clipped"
            )

        cl_alpha_th = 6.3 + np.clip(thickness_ratio, 0.0, 0.2) / 0.2 * (7.3 - 6.3)

        if float(airfoil_lift_coefficient / cl_alpha_th) != np.clip(
            float(airfoil_lift_coefficient / cl_alpha_th), 0.6, 1.0
        ):
            _LOGGER.warning(
                "Airfoil lift coefficient to theoretical lift coefficient ratio value outside of "
                "the range in Roskam's book, value clipped"
            )

        if chord_ratio != np.clip(chord_ratio, 0.1, 0.4):
            _LOGGER.warning(
                "Chord ratio value outside of the range in Roskam's book, value clipped"
            )

        chord_ratio = np.clip(chord_ratio, 0.1, 0.4)
        k_cl_alpha = np.clip(float(airfoil_lift_coefficient / cl_alpha_th), 0.6, 1.0)

        k_ch_delta = (
            0.22461
            + 0.78058 * k_cl_alpha
            - 14.74883 * chord_ratio
            + 50.39894 * chord_ratio * k_cl_alpha
            - 57.80300 * chord_ratio * k_cl_alpha ** 2.0
            + 23.72360 * chord_ratio * k_cl_alpha ** 3.0
            - 24.57380 * chord_ratio ** 4.0
            + 26.55090 * chord_ratio ** 3.0 * k_cl_alpha
            - 10.07301 * chord_ratio ** 2.0 * k_cl_alpha ** 2.0
            - 0.08509 * k_cl_alpha ** 4.0
        )

        return k_ch_delta

    @staticmethod
    @functools.lru_cache(maxsize=128)
    def ch_delta_th(thickness_ratio, chord_ratio):
        """
        Surrogate model based on Roskam data to compute the theoretical 2D control surface hinge
        moment derivative due to control surface deflection (figure 10.69 b).

        :param thickness_ratio: airfoil thickness ratio.
        :param chord_ratio: flap chord over wing chord ratio.
        :return ch_delta: theoretical hinge moment derivative due to control surface deflection.
        """

        if float(thickness_ratio) != np.clip(float(thickness_ratio), 0.0, 0.15):
            _LOGGER.warning(
                "Thickness ratio value outside of the range in Roskam's book, value clipped"
            )

        if chord_ratio != np.clip(chord_ratio, 0.1, 0.4):
            _LOGGER.warning(
                "Chord ratio value outside of the range in Roskam's book, value clipped"
            )

        chord_ratio = np.clip(chord_ratio, 0.1, 0.4)
        thickness_ratio = np.clip(thickness_ratio, 0.0, 0.15)

        ch_delta_th = (
            -0.82987
            - 0.46421 * chord_ratio
            + 1.19892 * thickness_ratio
            - 1.98707 * chord_ratio * thickness_ratio
            + 4.58039 * thickness_ratio ** 2.0
            - 7.18363 * chord_ratio * thickness_ratio ** 2.0
        )

        return ch_delta_th

    @staticmethod
    def k_fus(root_quarter_chord_position_ratio) -> float:
        """
        Surrogate model based on Roskam data to estimate the empirical pitching moment factor
        K_fus (figure 16.14).

        :param root_quarter_chord_position_ratio: the position of the root quarter chord of the
        wing from the nose.
        divided by the total length of the fuselage.
        :return k_fus: the empirical pitching moment factor.
        """

        if float(root_quarter_chord_position_ratio) != np.clip(
            float(root_quarter_chord_position_ratio), 0.092, 0.59
        ):
            _LOGGER.warning(
                "Position of the root quarter-chord as percent of fuselage length is outside of "
                "the range in Roskam's book, value clipped"
            )

        root_quarter_chord_position_ratio = np.clip(
            float(root_quarter_chord_position_ratio), 0.092, 0.59
        )

        k_fus = (
            -5.175619 * root_quarter_chord_position_ratio ** 6.0
            + 11.59254 * root_quarter_chord_position_ratio ** 5.0
            - 10.45849 * root_quarter_chord_position_ratio ** 4.0
            + 4.844842 * root_quarter_chord_position_ratio ** 3.0
            - 1.209880 * root_quarter_chord_position_ratio ** 2.0
            + 0.1711324 * root_quarter_chord_position_ratio
            - 7.721572e-03
        )

        return k_fus

    @staticmethod
    def cl_beta_sweep_contribution(taper_ratio, aspect_ratio, sweep_50) -> float:
        """
        Roskam data to estimate the contribution to the roll moment of the sweep angle of the
        lifting surface. (figure 10.20)

        :param taper_ratio: the taper ratio of the lifting surface
        :param aspect_ratio: the aspect ratio of the lifting surface
        :param sweep_50: the sweep angle at 50 percent of the chord of the lifting surface, in deg
        :return cl_beta_lambda: the contribution to the roll moment of the sweep angle of the
        lifting surface.
        """

        file = pth.join(resources.__path__[0], CL_BETA_SWEEP)
        db = read_csv(file)

        taper_ratio_data = db["TAPER_RATIO"]
        aspect_ratio_data = db["ASPECT_RATIO"]
        sweep_50_data = db["SWEEP_50"]
        sweep_contribution = db["SWEEP_CONTRIBUTION"]
        errors = np.logical_or.reduce(
            (
                np.isnan(taper_ratio_data),
                np.isnan(aspect_ratio_data),
                np.isnan(sweep_50_data),
                np.isnan(sweep_contribution),
            )
        )
        taper_ratio_data = taper_ratio_data[np.logical_not(errors)].tolist()
        aspect_ratio_data = aspect_ratio_data[np.logical_not(errors)].tolist()
        sweep_50_data = sweep_50_data[np.logical_not(errors)].tolist()
        sweep_contribution = sweep_contribution[np.logical_not(errors)].tolist()

        if float(taper_ratio) != np.clip(
            float(taper_ratio), min(taper_ratio_data), max(taper_ratio_data)
        ):
            _LOGGER.warning("Taper ratio is outside of the range in Roskam's book, value clipped")
        if float(aspect_ratio) != np.clip(
            float(aspect_ratio), min(aspect_ratio_data), max(aspect_ratio_data)
        ):
            _LOGGER.warning("Aspect ratio is outside of the range in Roskam's book, value clipped")
        if float(sweep_50) != np.clip(float(sweep_50), min(sweep_50_data), max(sweep_50_data)):
            _LOGGER.warning(
                "Sweep at 50% chord is outside of the range in Roskam's book, " "value clipped"
            )

        # Linear interpolation is preferred but we put the nearest one as protection
        cl_beta_lambda = interpolate.griddata(
            (taper_ratio_data, aspect_ratio_data, sweep_50_data),
            sweep_contribution,
            np.array([taper_ratio, aspect_ratio, sweep_50]).T,
            method="linear",
        )
        if np.isnan(cl_beta_lambda):
            cl_beta_lambda = interpolate.griddata(
                (taper_ratio_data, aspect_ratio_data, sweep_50_data),
                sweep_contribution,
                np.array([taper_ratio, aspect_ratio, sweep_50]).T,
                method="nearest",
            )

        return float(cl_beta_lambda)

    @staticmethod
    def cl_beta_sweep_compressibility_correction(swept_aspect_ratio, swept_mach) -> float:
        """
        Roskam data to estimate the compressibility correction for the sweep angle. (figure 10.21)

        :param swept_aspect_ratio: the aspect ratio of the lifting surface divided by cos(sweep_50)
        :param swept_mach: mach number multiplied by cos(sweep_50)
        :return k_m_lambda: compressibility correction for the sweep angle.
        """

        file = pth.join(resources.__path__[0], K_M_LAMBDA)
        db = read_csv(file)

        swept_aspect_ratio_data = db["AR_SWEPT"]
        swept_mach_data = db["M_SWEPT"]
        k_m_lambda_data = db["SWEEP_COMPRESSIBILITY_CORRECTION"]
        errors = np.logical_or.reduce(
            (
                np.isnan(swept_aspect_ratio_data),
                np.isnan(swept_mach_data),
                np.isnan(k_m_lambda_data),
            )
        )
        swept_aspect_ratio_data = swept_aspect_ratio_data[np.logical_not(errors)].tolist()
        swept_mach_data = swept_mach_data[np.logical_not(errors)].tolist()
        k_m_lambda_data = k_m_lambda_data[np.logical_not(errors)].tolist()

        if float(swept_aspect_ratio) != np.clip(
            float(swept_aspect_ratio), min(swept_aspect_ratio_data), max(swept_aspect_ratio_data)
        ):
            _LOGGER.warning(
                "Swept aspect ratio is outside of the range in Roskam's book, value clipped"
            )
        if float(swept_mach) != np.clip(
            float(swept_mach), min(swept_mach_data), max(swept_mach_data)
        ):
            _LOGGER.warning(
                "Swept mach number is outside of the range in Roskam's book, value clipped"
            )

        k_m_lambda = interpolate.griddata(
            (swept_aspect_ratio_data, swept_mach_data),
            k_m_lambda_data,
            np.array([swept_aspect_ratio, swept_mach]).T,
            method="linear",
        )
        if np.isnan(k_m_lambda):
            k_m_lambda = interpolate.griddata(
                (swept_aspect_ratio_data, swept_mach_data),
                k_m_lambda_data,
                np.array([swept_aspect_ratio, swept_mach]).T,
                method="nearest",
            )

        return float(k_m_lambda)

    @staticmethod
    def cl_beta_fuselage_correction(swept_aspect_ratio, lf_to_b_ratio) -> float:
        """
        Surrogate model based on Roskam data to estimate the fuselage correction factor.
        (figure 10.22)

        :param swept_aspect_ratio: the aspect ratio of the lifting surface divided by cos(sweep_50)
        :param lf_to_b_ratio: ratio between the distance from nose to root half chord and the
        wing span
        :return k_fuselage: fuselage correction factor.
        """

        if float(swept_aspect_ratio) != np.clip(float(swept_aspect_ratio), 4.0, 8.0):
            _LOGGER.warning(
                "Swept aspect ratio is outside of the range in Roskam's book, value clipped"
            )

        if float(lf_to_b_ratio) != np.clip(float(lf_to_b_ratio), 0.0, 1.6):
            _LOGGER.warning(
                "Ratio between the distance from nose to root half chord and the wing span is "
                "outside of the range in Roskam's book, value clipped"
            )

        swept_aspect_ratio = np.clip(swept_aspect_ratio, 4.0, 8.0)
        lf_to_b_ratio = np.exp(np.clip(lf_to_b_ratio, 0.0, 1.6))
        # lf_to_b_ratio is scaled just to ease the surrogate modeling process

        k_fuselage = 10.0 ** (
            30.30177
            - 301.96004 * np.log10(swept_aspect_ratio) ** 4.0 * np.log10(lf_to_b_ratio) ** 2.0
            + 58.15281 * np.log10(lf_to_b_ratio)
            - 353.84075 * np.log10(swept_aspect_ratio) * np.log10(lf_to_b_ratio)
            + 1043.60669 * np.log10(swept_aspect_ratio) ** 3.0 * np.log10(lf_to_b_ratio) ** 2.0
            - 1329.51750 * np.log10(swept_aspect_ratio) ** 2.0 * np.log10(lf_to_b_ratio) ** 2.0
            - 1013.34698 * np.log10(swept_aspect_ratio) ** 3.0 * np.log10(lf_to_b_ratio)
            + 4.14333 * np.log10(lf_to_b_ratio) ** 6.0
            + 48.93432 * np.log10(swept_aspect_ratio) ** 6.0
            - 2.64603 * np.log10(swept_aspect_ratio) * np.log10(lf_to_b_ratio) ** 5.0
            - 6.71611 * np.log10(lf_to_b_ratio) ** 5.0
            + 78.89282 * np.log10(lf_to_b_ratio) ** 3.0
            + 630.15406 * np.log10(swept_aspect_ratio) ** 2.0
            - 316.17554 * np.log10(swept_aspect_ratio) ** 5.0
            - 216.58234 * np.log10(swept_aspect_ratio)
            - 140.70409 * np.log10(swept_aspect_ratio) ** 3.0 * np.log10(lf_to_b_ratio) ** 3.0
            + 850.53251 * np.log10(swept_aspect_ratio) ** 2.0 * np.log10(lf_to_b_ratio)
            - 143.22447 * np.log10(swept_aspect_ratio) ** 5.0 * np.log10(lf_to_b_ratio)
            + 738.95626 * np.log10(swept_aspect_ratio) * np.log10(lf_to_b_ratio) ** 2.0
            + 356.66910 * np.log10(swept_aspect_ratio) ** 2.0 * np.log10(lf_to_b_ratio) ** 3.0
            - 949.14714 * np.log10(swept_aspect_ratio) ** 3.0
            - 150.75002 * np.log10(lf_to_b_ratio) ** 2.0
            - 297.78529 * np.log10(swept_aspect_ratio) * np.log10(lf_to_b_ratio) ** 3.0
            + 601.45144 * np.log10(swept_aspect_ratio) ** 4.0 * np.log10(lf_to_b_ratio)
            - 15.02028 * np.log10(lf_to_b_ratio) ** 4.0
            - 32.43379 * np.log10(swept_aspect_ratio) ** 2.0 * np.log10(lf_to_b_ratio) ** 4.0
            + 54.09680 * np.log10(swept_aspect_ratio) * np.log10(lf_to_b_ratio) ** 4.0
            + 772.47283 * np.log10(swept_aspect_ratio) ** 4.0
        )

        return float(k_fuselage)

    @staticmethod
    def cl_beta_ar_contribution(taper_ratio, aspect_ratio) -> float:
        """
        Roskam data to estimate the contribution to the roll moment of the aspect ratio of the
        lifting surface. (figure 10.23)

        :param taper_ratio: the taper ratio of the lifting surface
        :param aspect_ratio: the aspect ratio of the lifting surface
        :return cl_beta_ar: the contribution to the roll moment of the aspect ratio of the
        lifting surface.
        """

        file = pth.join(resources.__path__[0], CL_BETA_AR)
        db = read_csv(file)

        taper_ratio_data = db["TAPER_RATIO"]
        aspect_ratio_data = db["ASPECT_RATIO"]
        ar_contribution = db["ASPECT_RATIO_CONTRIBUTION"]
        errors = np.logical_or.reduce(
            (
                np.isnan(taper_ratio_data),
                np.isnan(aspect_ratio_data),
                np.isnan(ar_contribution),
            )
        )
        taper_ratio_data = taper_ratio_data[np.logical_not(errors)].tolist()
        aspect_ratio_data = aspect_ratio_data[np.logical_not(errors)].tolist()
        ar_contribution = ar_contribution[np.logical_not(errors)].tolist()

        if float(taper_ratio) != np.clip(
            float(taper_ratio), min(taper_ratio_data), max(taper_ratio_data)
        ):
            _LOGGER.warning("Taper ratio is outside of the range in Roskam's book, value clipped")
        if float(aspect_ratio) != np.clip(
            float(aspect_ratio), min(aspect_ratio_data), max(aspect_ratio_data)
        ):
            _LOGGER.warning("Aspect ratio is outside of the range in Roskam's book, value clipped")

        # Linear interpolation is preferred but we put the nearest one as protection
        cl_beta_ar = interpolate.griddata(
            (taper_ratio_data, aspect_ratio_data),
            ar_contribution,
            np.array([taper_ratio, aspect_ratio]).T,
            method="linear",
        )
        if np.isnan(cl_beta_ar):
            cl_beta_ar = interpolate.griddata(
                (taper_ratio_data, aspect_ratio_data),
                ar_contribution,
                np.array([taper_ratio, aspect_ratio]).T,
                method="nearest",
            )

        return float(cl_beta_ar)

    @staticmethod
    def cl_beta_dihedral_contribution(taper_ratio, aspect_ratio, sweep_50) -> float:
        """
        Roskam data to estimate the contribution to the roll moment of the dihedral angle of the
        lifting surface. (figure 10.24)

        :param taper_ratio: the taper ratio of the lifting surface
        :param aspect_ratio: the aspect ratio of the lifting surface
        :param sweep_50: the sweep angle at 50 percent of the chord of the lifting surface, in deg
        :return cl_beta_gamma: the contribution to the roll moment of the dihedral angle of the
        lifting surface.
        """

        # For this graph, only the absolute value of the sweep angle is necessary
        sweep_50 = np.abs(sweep_50)

        file = pth.join(resources.__path__[0], CL_BETA_GAMMA)
        db = read_csv(file)

        taper_ratio_data = db["TAPER_RATIO"]
        aspect_ratio_data = db["ASPECT_RATIO"]
        sweep_50_data = db["SWEEP_50"]
        dihedral_contribution = db["DIHEDRAL_CONTRIBUTION"]
        errors = np.logical_or.reduce(
            (
                np.isnan(taper_ratio_data),
                np.isnan(aspect_ratio_data),
                np.isnan(sweep_50_data),
                np.isnan(dihedral_contribution),
            )
        )
        taper_ratio_data = taper_ratio_data[np.logical_not(errors)].tolist()
        aspect_ratio_data = aspect_ratio_data[np.logical_not(errors)].tolist()
        sweep_50_data = sweep_50_data[np.logical_not(errors)].tolist()
        dihedral_contribution = dihedral_contribution[np.logical_not(errors)].tolist()

        if float(taper_ratio) != np.clip(
            float(taper_ratio), min(taper_ratio_data), max(taper_ratio_data)
        ):
            _LOGGER.warning("Taper ratio is outside of the range in Roskam's book, value clipped")
        if float(aspect_ratio) != np.clip(
            float(aspect_ratio), min(aspect_ratio_data), max(aspect_ratio_data)
        ):
            _LOGGER.warning("Aspect ratio is outside of the range in Roskam's book, value clipped")
        if float(sweep_50) != np.clip(float(sweep_50), min(sweep_50_data), max(sweep_50_data)):
            _LOGGER.warning(
                "Sweep at 50% chord is outside of the range in Roskam's book, " "value clipped"
            )

        # Linear interpolation is preferred but we put the nearest one as protection
        cl_beta_gamma = interpolate.griddata(
            (taper_ratio_data, aspect_ratio_data, sweep_50_data),
            dihedral_contribution,
            np.array([taper_ratio, aspect_ratio, sweep_50]).T,
            method="linear",
        )
        if np.isnan(cl_beta_gamma):
            cl_beta_gamma = interpolate.griddata(
                (taper_ratio_data, aspect_ratio_data, sweep_50_data),
                dihedral_contribution,
                np.array([taper_ratio, aspect_ratio, sweep_50]).T,
                method="nearest",
            )

        return float(cl_beta_gamma)

    @staticmethod
    def cl_beta_dihedral_compressibility_correction(swept_aspect_ratio, swept_mach) -> float:
        """
        Surrogate model based on Roskam data to estimate the compressibility correction for
        the dihedral angle. (figure 10.25)

        :param swept_aspect_ratio: the aspect ratio of the lifting surface divided by cos(sweep_50)
        :param swept_mach: mach number multiplied by cos(sweep_50)
        :return k_m_gamma: compressibility correction for the dihedral angle.
        """

        if float(swept_aspect_ratio) != np.clip(float(swept_aspect_ratio), 2.0, 10.0):
            _LOGGER.warning(
                "Swept aspect ratio is outside of the range in Roskam's book, value clipped"
            )
        if float(swept_mach) != np.clip(float(swept_mach), 0.0, 0.957):
            _LOGGER.warning(
                "Swept mach number is outside of the range in Roskam's book, value clipped"
            )

        swept_aspect_ratio = np.clip(swept_aspect_ratio, 2.0, 10.0)
        swept_mach = np.clip(swept_mach, 0.0, 0.957)

        k_m_gamma = (
            1.00011
            + 0.45331 * swept_aspect_ratio * swept_mach ** 3.0
            - 0.01004 * swept_mach ** 3.0
            - 0.00030 * swept_aspect_ratio ** 3.0 * swept_mach ** 3.0
            + 0.01663 * swept_aspect_ratio ** 2.0 * swept_mach ** 4.0
            - 0.04659 * swept_aspect_ratio * swept_mach ** 2.0
            - 0.00973 * swept_aspect_ratio ** 2.0 * swept_mach ** 3.0
            - 0.03766 * swept_mach
            + 0.00969 * swept_aspect_ratio * swept_mach
            + 0.30939 * swept_aspect_ratio * swept_mach ** 5.0
            - 0.68111 * swept_aspect_ratio * swept_mach ** 4.0
        )

        return float(k_m_gamma)

    @staticmethod
    def cl_beta_twist_correction(taper_ratio, aspect_ratio) -> float:
        """
        Surrogate model based on Roskam data to estimate the correction due to the twist of
        the lifting surface. (figure 10.26)

        :param taper_ratio: the taper ratio of the lifting surface
        :param aspect_ratio: the aspect ratio of the lifting surface
        :return k_epsilon: the factor to take into account the twist of the lifting surface for
        the computation of the rolling moment
        """

        if float(taper_ratio) != np.clip(float(taper_ratio), 0.0, 0.6):
            _LOGGER.warning("Taper ratio is outside of the range in Roskam's book, value clipped")

        if float(aspect_ratio) != np.clip(float(aspect_ratio), 3.0, 11.51):
            _LOGGER.warning("Aspect ratio is outside of the range in Roskam's book, value clipped")

        taper_ratio = np.exp(np.clip(taper_ratio, 0.0, 0.6))
        aspect_ratio = np.clip(aspect_ratio, 3.0, 11.51) * 0.01 + 0.1
        # aspect_ratio and taper_ratio are translated and scaled just to ease the surrogate modeling process

        k_epsilon = (
            -1.0
            / 10.0 ** 3.0
            * np.log(
                10.0
                ** (
                    4414.31434
                    - 4010.35916 * np.log10(aspect_ratio) ** 8.0
                    - 27160.56784 * np.log10(taper_ratio)
                    - 145795.79645 * np.log10(taper_ratio) * np.log10(aspect_ratio) ** 7.0
                    - 807996.27290 * np.log10(taper_ratio) * np.log10(aspect_ratio) ** 6.0
                    - 243317.77067 * np.log10(taper_ratio) * np.log10(aspect_ratio)
                    + 38416.60761 * np.log10(aspect_ratio)
                    + 141923.45735 * np.log10(aspect_ratio) ** 2.0
                    + 0.82681 * np.log10(taper_ratio) ** 2.0
                    + 2.32588 * np.log10(taper_ratio) ** 2.0 * np.log10(aspect_ratio) ** 6.0
                    + 1.61461 * np.log10(taper_ratio) ** 2.0 * np.log10(aspect_ratio) ** 5.0
                    + 287040.71354 * np.log10(aspect_ratio) ** 3.0
                    + 339465.71034 * np.log10(aspect_ratio) ** 4.0
                    - 932087.67523 * np.log10(taper_ratio) * np.log10(aspect_ratio) ** 2.0
                    + 227920.16703 * np.log10(aspect_ratio) ** 5.0
                    - 1915028.95980 * np.log10(taper_ratio) * np.log10(aspect_ratio) ** 5.0
                    - 1979243.81244 * np.log10(taper_ratio) * np.log10(aspect_ratio) ** 3.0
                    + 71584.19258 * np.log10(aspect_ratio) ** 6.0
                    - 2516140.72497 * np.log10(taper_ratio) * np.log10(aspect_ratio) ** 4.0
                    + 1.21555 * np.log10(taper_ratio) ** 2.0 * np.log10(aspect_ratio)
                )
            )
        )

        return float(k_epsilon)

    @staticmethod
    def cl_p_roll_damping_parameter(taper_ratio, aspect_ratio, mach, sweep_25, k) -> float:
        """
        Roskam data to estimate the contribution to the roll moment of the roll damping parameter
        (figure 10.35).

        :param taper_ratio: the taper ratio of the lifting surface
        :param aspect_ratio: the aspect ratio of the lifting surface
        :param mach: the mach number
        :param sweep_25: the sweep angle at 25 percent of the chord of the lifting surface, in deg
        :param k: the ratio between the airfoil slope and 2*np.pi
        :return k_roll_damping: the roll damping parameter
        """

        beta = np.sqrt(1.0 - mach ** 2.0)

        corrected_ar = aspect_ratio * beta / k
        corrected_sweep = np.arctan(np.tan(sweep_25) / beta)
        file = pth.join(resources.__path__[0], K_ROLL_DAMPING)
        db = read_csv(file)

        taper_ratio_data = db["TAPER_RATIO"]
        correct_ar_data = db["CORRECTED_AR"]
        corrected_sweep_data = db["CORRECTED_SWEEP"]
        roll_damping_data = db["ROLL_DAMPING_PARAMETER"]
        errors = np.logical_or.reduce(
            (
                np.isnan(taper_ratio_data),
                np.isnan(correct_ar_data),
                np.isnan(corrected_sweep_data),
                np.isnan(roll_damping_data),
            )
        )
        taper_ratio_data = taper_ratio_data[np.logical_not(errors)].tolist()
        correct_ar_data = correct_ar_data[np.logical_not(errors)].tolist()
        corrected_sweep_data = corrected_sweep_data[np.logical_not(errors)].tolist()
        roll_damping_data = roll_damping_data[np.logical_not(errors)].tolist()

        if float(taper_ratio) != np.clip(
            float(taper_ratio), min(taper_ratio_data), max(taper_ratio_data)
        ):
            _LOGGER.warning("Taper ratio is outside of the range in Roskam's book, value clipped")
        if float(corrected_ar) != np.clip(
            float(corrected_ar), min(correct_ar_data), max(correct_ar_data)
        ):
            _LOGGER.warning(
                "Corrected Aspect ratio is outside of the range in Roskam's book, value clipped"
            )
        if float(corrected_sweep) != np.clip(
            float(corrected_sweep), min(corrected_sweep_data), max(corrected_sweep_data)
        ):
            _LOGGER.warning(
                "Corrected Sweep is outside of the range in Roskam's book, value clipped"
            )

        # Linear interpolation is preferred but we put the nearest one as protection
        k_roll_damping = interpolate.griddata(
            (taper_ratio_data, correct_ar_data, corrected_sweep_data),
            roll_damping_data,
            np.array([taper_ratio, corrected_ar, corrected_sweep]).T,
            method="linear",
        )
        if np.isnan(k_roll_damping):
            k_roll_damping = interpolate.griddata(
                (taper_ratio_data, correct_ar_data, corrected_sweep_data),
                roll_damping_data,
                np.array([taper_ratio, corrected_ar, corrected_sweep]).T,
                method="nearest",
            )

        return float(k_roll_damping)

    @staticmethod
    def cl_p_cdi_roll_damping(sweep_25, aspect_ratio) -> float:
        """
        Surrogate model based on Roskam data to estimate the contribution to the roll moment damping
        of the drag-due-to-lift (figure 10.36)

        :param sweep_25: the sweep angle at 25% of the chord of the lifting surface
        :param aspect_ratio: the aspect ratio of the lifting surface
        :return k_cdi_roll_damping: the contribution to the roll moment of the aspect ratio of the
        lifting surface.
        """

        if float(sweep_25) != np.clip(float(sweep_25), 0.0, 60.0):
            _LOGGER.warning(
                "Sweep at 25% of the chord is outside of the range in Roskam's book, value clipped"
            )
        if float(aspect_ratio) != np.clip(float(aspect_ratio), 1.0, 10.0):
            _LOGGER.warning("Aspect ratio is outside of the range in Roskam's book, value clipped")

        sweep_25 = np.clip(sweep_25, 0.0, 60.0) * 0.01
        aspect_ratio = np.clip(aspect_ratio, 1.0, 10.0) * 0.1
        # sweep25 and aspect ratio are scaled to ease the surrogate modeling process

        k_cdi_roll_damping = (
            -0.08737
            - 3.18085 * sweep_25 ** 4.0
            + 35.84855 * sweep_25 ** 4.0 * aspect_ratio
            - 157.04948 * sweep_25 ** 4.0 * aspect_ratio ** 2.0
            + 0.63981 * aspect_ratio
            + 373.85899 * sweep_25 ** 4.0 * aspect_ratio ** 3.0
            - 2.19618 * aspect_ratio ** 2.0
            - 494.24239 * sweep_25 ** 4.0 * aspect_ratio ** 4.0
            + 3.74221 * aspect_ratio ** 3.0
            + 338.79305 * sweep_25 ** 4.0 * aspect_ratio ** 5.0
            - 3.07844 * aspect_ratio ** 4.0
            + 0.97357 * aspect_ratio ** 5.0
            - 93.68557 * sweep_25 ** 4.0 * aspect_ratio ** 6.0
            - 0.07282 * sweep_25
            + 5.46093 * sweep_25 ** 9.0 * aspect_ratio
            + 0.07748 * sweep_25 * aspect_ratio
            - 2.71505 * sweep_25 ** 6.0
            - 0.01909 * sweep_25 * aspect_ratio ** 2.0
        )

        return float(k_cdi_roll_damping)

    @staticmethod
    def cl_r_lifting_effect(aspect_ratio, taper_ratio, sweep_25):
        """
        Roskam data to estimate the slope of the rolling moment due to yaw rate (figure 10.41).
        The figure is separated into two parts (a and b).

        :param aspect_ratio: wing aspect ratio
        :param taper_ratio: wing taper ratio
        :param sweep_25: wing sweep angle at quarter-taper point line in radians
        :return cl_r_lift: slope of the rolling moment due to yaw rate
        """

        sweep_25 = sweep_25 * 180.0 / np.pi  # radians to degrees

        # Reading data from the first part (a) relative to the wing taper ratio
        file = pth.join(resources.__path__[0], CL_R_LIFT_PART_A)
        db = read_csv(file)

        x_0 = db["TAPER_RATIO_0_X"]
        y_0 = db["TAPER_RATIO_0_Y"]
        errors = np.logical_or(np.isnan(x_0), np.isnan(y_0))
        x_0 = x_0[np.logical_not(errors)].tolist()
        y_0 = y_0[np.logical_not(errors)].tolist()
        x_0.sort()
        y_0.sort()

        x_0_25 = db["TAPER_RATIO_025_X"]
        y_0_25 = db["TAPER_RATIO_025_Y"]
        errors = np.logical_or(np.isnan(x_0_25), np.isnan(y_0_25))
        x_0_25 = x_0_25[np.logical_not(errors)].tolist()
        y_0_25 = y_0_25[np.logical_not(errors)].tolist()
        x_0_25.sort()
        y_0_25.sort()

        x_0_5 = db["TAPER_RATIO_05_X"]
        y_0_5 = db["TAPER_RATIO_05_Y"]
        errors = np.logical_or(np.isnan(x_0_5), np.isnan(y_0_5))
        x_0_5 = x_0_5[np.logical_not(errors)].tolist()
        y_0_5 = y_0_5[np.logical_not(errors)].tolist()
        x_0_5.sort()
        y_0_5.sort()

        x_1_0 = db["TAPER_RATIO_1_X"]
        y_1_0 = db["TAPER_RATIO_1_Y"]
        errors = np.logical_or(np.isnan(x_1_0), np.isnan(y_1_0))
        x_1_0 = x_1_0[np.logical_not(errors)].tolist()
        y_1_0 = y_1_0[np.logical_not(errors)].tolist()
        x_1_0.sort()
        y_1_0.sort()

        k_taper0 = interpolate.interp1d(x_0, y_0)
        k_taper025 = interpolate.interp1d(x_0_25, y_0_25)
        k_taper05 = interpolate.interp1d(x_0_5, y_0_5)
        k_taper1 = interpolate.interp1d(x_1_0, y_1_0)

        if (
            (aspect_ratio != np.clip(aspect_ratio, min(x_1_0), max(x_1_0)))
            or (aspect_ratio != np.clip(aspect_ratio, min(x_0_5), max(x_0_5)))
            or (aspect_ratio != np.clip(aspect_ratio, min(x_0_25), max(x_0_25)))
            or (aspect_ratio != np.clip(aspect_ratio, min(x_0), max(x_0)))
        ):
            _LOGGER.warning(
                "Aspect ratio value outside of the range in Roskam's book, value clipped"
            )

        k_taper = [
            float(k_taper0(np.clip(aspect_ratio, min(x_0), max(x_0)))),
            float(k_taper025(np.clip(aspect_ratio, min(x_0_25), max(x_0_25)))),
            float(k_taper05(np.clip(aspect_ratio, min(x_0_5), max(x_0_5)))),
            float(k_taper1(np.clip(aspect_ratio, min(x_1_0), max(x_1_0)))),
        ]

        if taper_ratio != np.clip(taper_ratio, 0.0, 1.0):
            _LOGGER.warning(
                "Taper ratio value outside of the range in Roskam's book, value clipped"
            )

        k_intermediate = float(
            interpolate.interp1d([0.0, 0.25, 0.5, 1.0], k_taper)(np.clip(taper_ratio, 0.0, 1.0))
        )

        # Reading the second part of the figure (b) relative to the different wing sweep angles.
        file = pth.join(resources.__path__[0], CL_R_LIFT_PART_B)
        db = read_csv(file)

        x_sw_0 = db["SWEEP_25_0_X"]
        y_sw_0 = db["SWEEP_25_0_Y"]
        errors = np.logical_or(np.isnan(x_sw_0), np.isnan(y_sw_0))
        x_sw_0 = x_sw_0[np.logical_not(errors)].tolist()
        y_sw_0 = y_sw_0[np.logical_not(errors)].tolist()
        x_sw_0.sort()
        y_sw_0.sort()

        x_sw_15 = db["SWEEP_25_15_X"]
        y_sw_15 = db["SWEEP_25_15_Y"]
        errors = np.logical_or(np.isnan(x_sw_15), np.isnan(y_sw_15))
        x_sw_15 = x_sw_15[np.logical_not(errors)].tolist()
        y_sw_15 = y_sw_15[np.logical_not(errors)].tolist()
        x_sw_15.sort()
        y_sw_15.sort()

        x_sw_30 = db["SWEEP_25_30_X"]
        y_sw_30 = db["SWEEP_25_30_Y"]
        errors = np.logical_or(np.isnan(x_sw_30), np.isnan(y_sw_30))
        x_sw_30 = x_sw_30[np.logical_not(errors)].tolist()
        y_sw_30 = y_sw_30[np.logical_not(errors)].tolist()
        x_sw_30.sort()
        y_sw_30.sort()

        x_sw_45 = db["SWEEP_25_45_X"]
        y_sw_45 = db["SWEEP_25_45_Y"]
        errors = np.logical_or(np.isnan(x_sw_45), np.isnan(y_sw_45))
        x_sw_45 = x_sw_45[np.logical_not(errors)].tolist()
        y_sw_45 = y_sw_45[np.logical_not(errors)].tolist()
        x_sw_45.sort()
        y_sw_45.sort()

        x_sw_60 = db["SWEEP_25_60_X"]
        y_sw_60 = db["SWEEP_25_60_Y"]
        errors = np.logical_or(np.isnan(x_sw_60), np.isnan(y_sw_60))
        x_sw_60 = x_sw_60[np.logical_not(errors)].tolist()
        y_sw_60 = y_sw_60[np.logical_not(errors)].tolist()
        x_sw_60.sort()
        y_sw_60.sort()

        k_sweep0 = interpolate.interp1d(x_sw_0, y_sw_0)
        k_sweep15 = interpolate.interp1d(x_sw_15, y_sw_15)
        k_sweep30 = interpolate.interp1d(x_sw_30, y_sw_30)
        k_sweep45 = interpolate.interp1d(x_sw_45, y_sw_45)
        k_sweep60 = interpolate.interp1d(x_sw_60, y_sw_60)

        if (
            (k_intermediate != np.clip(k_intermediate, min(x_sw_45), max(x_sw_45)))
            or (k_intermediate != np.clip(k_intermediate, min(x_sw_30), max(x_sw_30)))
            or (k_intermediate != np.clip(k_intermediate, min(x_sw_15), max(x_sw_15)))
            or (k_intermediate != np.clip(k_intermediate, min(x_sw_0), max(x_sw_0)))
            or (k_intermediate != np.clip(k_intermediate, min(x_sw_60), max(x_sw_60)))
        ):
            _LOGGER.warning(
                "Intermediate value outside of the range in Roskam's book, value clipped"
            )

        k_sweep = [
            float(k_sweep0(np.clip(k_intermediate, min(x_sw_0), max(x_sw_0)))),
            float(k_sweep15(np.clip(k_intermediate, min(x_sw_15), max(x_sw_15)))),
            float(k_sweep30(np.clip(k_intermediate, min(x_sw_30), max(x_sw_30)))),
            float(k_sweep45(np.clip(k_intermediate, min(x_sw_45), max(x_sw_45)))),
            float(k_sweep60(np.clip(k_intermediate, min(x_sw_60), max(x_sw_60)))),
        ]

        if sweep_25 != np.clip(sweep_25, 0.0, 60.0):
            _LOGGER.warning(
                "Sweep angle value outside of the range in Roskam's book, value clipped"
            )

        cl_r_lift = float(
            interpolate.interp1d([0.0, 15.0, 30.0, 45.0, 60.0], k_sweep)(
                np.clip(sweep_25, 0.0, 60.0)
            )
        )

        return cl_r_lift

    @staticmethod
    def cl_r_twist_effect(taper_ratio, aspect_ratio) -> float:
        """
        Surrogate model based on Roskam data to estimate the contribution to the roll moment
        coefficient of the twist. (figure 10.42)

        :param taper_ratio: the taper ratio of the lifting surface
        :param aspect_ratio: the aspect ratio of the lifting surface
        :return k_twist: contribution to the roll moment coefficient of the twist.
        """

        if float(taper_ratio) != np.clip(float(taper_ratio), 0.0, 1.0):
            _LOGGER.warning("Taper ratio is outside of the range in Roskam's book, value clipped")

        if float(aspect_ratio) != np.clip(float(aspect_ratio), 2.0, 10.0):
            _LOGGER.warning("Aspect ratio is outside of the range in Roskam's book, value clipped")

        taper_ratio = np.exp(np.clip(taper_ratio, 0.0, 1.0))
        aspect_ratio = np.clip(aspect_ratio, 2.0, 10.0)
        # taper_ratio is scaled just to ease the surrogate modeling process

        k_twist = np.log(
            10.0
            ** (
                -0.05691
                + 0.75788 * np.log10(aspect_ratio)
                + 2.29272 * np.log10(taper_ratio) * np.log10(aspect_ratio) ** 2.0
                + 11.77360 * np.log10(taper_ratio) ** 2.0 * np.log10(aspect_ratio) ** 2.0
                - 111.82216 * np.log10(taper_ratio) ** 3.0 * np.log10(aspect_ratio) ** 2.0
                - 6.68663 * np.log10(taper_ratio) * np.log10(aspect_ratio) ** 3.0
                - 23.22959 * np.log10(taper_ratio) ** 2.0 * np.log10(aspect_ratio) ** 4.0
                + 0.00449 * np.log10(taper_ratio)
                - 5.64798 * np.log10(taper_ratio) ** 2.0 * np.log10(aspect_ratio)
                - 3.75020 * np.log10(aspect_ratio) ** 2.0
                - 14.53987 * np.log10(taper_ratio) ** 2.0 * np.log10(aspect_ratio) ** 7.0
                + 25.31895 * np.log10(taper_ratio) ** 2.0 * np.log10(aspect_ratio) ** 6.0
                + 6.07351 * np.log10(taper_ratio) ** 2.0 * np.log10(aspect_ratio) ** 5.0
                + 96.06766 * np.log10(taper_ratio) ** 6.0 * np.log10(aspect_ratio) ** 3.0
                - 0.27059 * np.log10(taper_ratio) * np.log10(aspect_ratio)
                - 9.57656 * np.log10(aspect_ratio) ** 9.0
                + 44.24848 * np.log10(aspect_ratio) ** 8.0
                - 3.73715 * np.log10(taper_ratio) * np.log10(aspect_ratio) ** 6.0
                - 35.73339 * np.log10(aspect_ratio) ** 5.0
                - 45.72376 * np.log10(taper_ratio) ** 5.0 * np.log10(aspect_ratio) ** 3.0
                + 50.82538 * np.log10(taper_ratio) ** 3.0 * np.log10(aspect_ratio) ** 6.0
                + 213.22572 * np.log10(taper_ratio) ** 5.0
                + 357.73718 * np.log10(taper_ratio) ** 4.0 * np.log10(aspect_ratio) ** 4.0
                + 217.08369 * np.log10(taper_ratio) ** 4.0 * np.log10(aspect_ratio) ** 2.0
                - 63.55425 * np.log10(taper_ratio) ** 4.0 * np.log10(aspect_ratio)
                - 509.87340 * np.log10(taper_ratio) ** 6.0
                - 87.09365 * np.log10(taper_ratio) ** 4.0 * np.log10(aspect_ratio) ** 5.0
                + 1.07977 * np.log10(taper_ratio) * np.log10(aspect_ratio) ** 8.0
                - 83.31625 * np.log10(aspect_ratio) ** 7.0
                + 79.51372 * np.log10(aspect_ratio) ** 6.0
                + 7.91769 * np.log10(aspect_ratio) ** 3.0
                - 404.33108 * np.log10(taper_ratio) ** 4.0 * np.log10(aspect_ratio) ** 3.0
                + 37.25993 * np.log10(taper_ratio) ** 3.0 * np.log10(aspect_ratio)
                + 523.81998 * np.log10(taper_ratio) ** 7.0
                - 55.92062 * np.log10(taper_ratio) ** 5.0 * np.log10(aspect_ratio) ** 4.0
                + 32.37203 * np.log10(taper_ratio) ** 5.0 * np.log10(aspect_ratio) ** 2.0
                - 263.48485 * np.log10(taper_ratio) ** 9.0
                + 7.31962 * np.log10(taper_ratio) * np.log10(aspect_ratio) ** 4.0
                - 37.83321 * np.log10(taper_ratio) ** 4.0
                + 0.63774 * np.log10(taper_ratio) ** 2.0
                - 55.92877 * np.log10(taper_ratio) ** 7.0 * np.log10(aspect_ratio) ** 2.0
                - 113.89145 * np.log10(taper_ratio) ** 3.0 * np.log10(aspect_ratio) ** 5.0
                + 136.44476 * np.log10(taper_ratio) ** 3.0 * np.log10(aspect_ratio) ** 3.0
            )
        )

        return float(k_twist)

    @staticmethod
    def cn_delta_a_correlation_constant(taper_ratio, aspect_ratio, eta_i) -> float:
        """
        Roskam data to estimate the correlation constant for the computation of the yaw moment
        due to aileron. (figure 10.48)

        :param taper_ratio: the taper ratio of the lifting surface
        :param aspect_ratio: the aspect ratio of the lifting surface
        :param eta_i: aileron inboard span location, as a ratio of the span
        :return k_a: the correlation constant for the computation of the yaw moment
        due to aileron
        """

        file = pth.join(resources.__path__[0], CN_DELTA_A_K_A)
        db = read_csv(file)

        taper_ratio_data = db["TAPER_RATIO"]
        aspect_ratio_data = db["ASPECT_RATIO"]
        eta_i_data = db["SPAN_RATIO"]
        correlation_constant = db["CORRELATION_CONSTANT"]
        errors = np.logical_or.reduce(
            (
                np.isnan(taper_ratio_data),
                np.isnan(aspect_ratio_data),
                np.isnan(eta_i_data),
                np.isnan(correlation_constant),
            )
        )
        taper_ratio_data = taper_ratio_data[np.logical_not(errors)].tolist()
        aspect_ratio_data = aspect_ratio_data[np.logical_not(errors)].tolist()
        eta_i_data = eta_i_data[np.logical_not(errors)].tolist()
        correlation_constant = correlation_constant[np.logical_not(errors)].tolist()

        if float(taper_ratio) != np.clip(
            float(taper_ratio), min(taper_ratio_data), max(taper_ratio_data)
        ):
            _LOGGER.warning("Taper ratio is outside of the range in Roskam's book, value clipped")
        if float(aspect_ratio) != np.clip(
            float(aspect_ratio), min(aspect_ratio_data), max(aspect_ratio_data)
        ):
            _LOGGER.warning("Aspect ratio is outside of the range in Roskam's book, value clipped")
        if float(eta_i) != np.clip(
            float(eta_i), min(correlation_constant), max(correlation_constant)
        ):
            _LOGGER.warning(
                "Aileron inboard location is outside of the range in Roskam's book, value clipped"
            )

        # Linear interpolation is preferred but we put the nearest one as protection
        k_a = interpolate.griddata(
            (taper_ratio_data, aspect_ratio_data, eta_i_data),
            correlation_constant,
            np.array([taper_ratio, aspect_ratio, eta_i]).T,
            method="linear",
        )
        if np.isnan(k_a):
            k_a = interpolate.griddata(
                (taper_ratio_data, aspect_ratio_data, eta_i_data),
                correlation_constant,
                np.array([taper_ratio, aspect_ratio, eta_i]).T,
                method="nearest",
            )

        return float(k_a)

    @staticmethod
    def cn_p_twist_contribution(taper_ratio, aspect_ratio) -> float:
        """
        Surrogate model based on Roskam data to estimate the contribution to the yaw moment of
        the twist of the lifting surface. (figure 10.37)

        :param taper_ratio: the taper ratio of the lifting surface
        :param aspect_ratio: the aspect ratio of the lifting surface
        :return cn_p_twist: the contribution to the yaw moment of the twist of the
        lifting surface.
        """

        if float(taper_ratio) != np.clip(float(taper_ratio), 0.0, 1.0):
            _LOGGER.warning("Taper ratio is outside of the range in Roskam's book, value clipped")
        if float(aspect_ratio) != np.clip(float(aspect_ratio), 2.0, 12.0):
            _LOGGER.warning("Aspect ratio is outside of the range in Roskam's book, value clipped")

        taper_ratio = np.clip(taper_ratio, 0.0, 1.0)
        aspect_ratio = np.clip(aspect_ratio, 2.0, 12.0) * 0.1
        # aspect ratio is scaled by 0.01 to ease the surrogate modeling process

        cn_p_twist = (
            -0.000449
            + 0.00255 * aspect_ratio
            - 0.00584 * taper_ratio
            - 0.00252 * aspect_ratio ** 2.0
            - 0.00768 * taper_ratio ** 5.0 * aspect_ratio ** 3.0
            + 0.01296 * taper_ratio ** 2.0
            + 0.00148 * taper_ratio * aspect_ratio ** 4.0
            + 0.03829 * taper_ratio ** 4.0 * aspect_ratio ** 3.0
            + 0.01168 * taper_ratio * aspect_ratio
            + 0.02099 * taper_ratio ** 3.0 * aspect_ratio
            + 0.00150 * aspect_ratio ** 3.0
            + 0.01936 * taper_ratio ** 6.0 * aspect_ratio ** 2.0
            + 0.02307 * taper_ratio ** 2.0 * aspect_ratio ** 2.0
            - 0.01259 * taper_ratio ** 3.0
            - 0.03098 * taper_ratio ** 3.0 * aspect_ratio ** 3.0
            - 0.03142 * taper_ratio ** 2.0 * aspect_ratio
            - 0.00792 * taper_ratio * aspect_ratio ** 2.0
            - 0.00029 * taper_ratio ** 8.0
            - 0.00325 * taper_ratio ** 2.0 * aspect_ratio ** 4.0
            + 0.01092 * taper_ratio ** 3.0 * aspect_ratio ** 4.0
            + 0.00475 * taper_ratio ** 4.0
            - 0.03463 * taper_ratio ** 5.0 * aspect_ratio ** 2.0
            - 0.00085 * taper_ratio ** 7.0 * aspect_ratio
            - 0.00024 * aspect_ratio ** 5.0
            - 0.00891 * taper_ratio ** 4.0 * aspect_ratio ** 4.0
        )

        return float(cn_p_twist)

    @staticmethod
    def cn_r_lift_effect(static_margin, sweep_25, aspect_ratio, taper_ratio) -> float:
        """
        Roskam data to estimate the effect of lift for the computation of the yaw moment
        due yaw rate (yaw damping). (figure 10.48)

        :param static_margin: distance between aft cg and aircraft aerodynamic center divided by MAC
        :param sweep_25: the sweep at 25% of the lifting surface
        :param aspect_ratio: the aspect ratio of the lifting surface
        :param taper_ratio: the taper ratio of the lifting surface
        :return lift_effect: the effect of lift fot the computation of the yaw moment due to yaw
        rate
        """

        # Only absolute value counts for this coefficient
        sweep_25 = abs(sweep_25)

        file = pth.join(resources.__path__[0], CN_R_LIFT_EFFECT)
        db = read_csv(file)

        static_margin_data = db["STATIC_MARGIN"]
        sweep_25_data = db["SWEEP_25"]
        aspect_ratio_data = db["ASPECT_RATIO"]
        intermediate_coeff_data = db["INTERMEDIATE_COEFF"]
        errors = np.logical_or.reduce(
            (
                np.isnan(static_margin_data),
                np.isnan(sweep_25_data),
                np.isnan(aspect_ratio_data),
                np.isnan(intermediate_coeff_data),
            )
        )
        static_margin_data = static_margin_data[np.logical_not(errors)].tolist()
        sweep_25_data = sweep_25_data[np.logical_not(errors)].tolist()
        aspect_ratio_data = aspect_ratio_data[np.logical_not(errors)].tolist()
        intermediate_coeff_data = intermediate_coeff_data[np.logical_not(errors)].tolist()

        if float(static_margin) != np.clip(
            float(static_margin), min(static_margin_data), max(static_margin_data)
        ):
            _LOGGER.warning("Static margin is outside of the range in Roskam's book, value clipped")
        if float(sweep_25) != np.clip(float(sweep_25), min(sweep_25_data), max(sweep_25_data)):
            _LOGGER.warning(
                "Sweep at 25% chord is outside of the range in Roskam's book, value clipped"
            )
        if float(aspect_ratio) != np.clip(
            float(aspect_ratio), min(aspect_ratio_data), max(aspect_ratio_data)
        ):
            _LOGGER.warning("Aspect ratio is outside of the range in Roskam's book, value clipped")

        # Linear interpolation is preferred but we put the nearest one as protection
        mid_coeff = interpolate.griddata(
            (static_margin_data, sweep_25_data, aspect_ratio_data),
            intermediate_coeff_data,
            np.array([static_margin, sweep_25, aspect_ratio]).T,
            method="linear",
        )
        if np.isnan(mid_coeff):
            mid_coeff = interpolate.griddata(
                (static_margin_data, sweep_25_data, aspect_ratio_data),
                intermediate_coeff_data,
                np.array([static_margin, sweep_25, aspect_ratio]).T,
                method="nearest",
            )

        lift_effect = 1.0 / 20.0 * (mid_coeff - 2.7 - 0.3 * taper_ratio)

        return float(lift_effect)

    @staticmethod
    def cn_r_drag_effect(static_margin, sweep_25, aspect_ratio) -> float:
        """
        Roskam data to estimate the effect of drag for the computation of the yaw moment
        due yaw rate (yaw damping). (figure 10.48)

        :param static_margin: distance between aft cg and aircraft aerodynamic center divided by MAC
        :param sweep_25: the sweep at 25% of the lifting surface
        :param aspect_ratio: the aspect ratio of the lifting surface
        :return drag_effect: the effect of drag for the computation of the yaw moment due to yaw
        rate
        """

        # Only absolute value counts for this coefficient
        sweep_25 = abs(sweep_25)

        file = pth.join(resources.__path__[0], CN_R_DRAG_EFFECT)
        db = read_csv(file)

        static_margin_data = db["STATIC_MARGIN"]
        sweep_25_data = db["SWEEP_25"]
        aspect_ratio_data = db["ASPECT_RATIO"]
        drag_effect_data = db["DRAG_EFFECT"]
        errors = np.logical_or.reduce(
            (
                np.isnan(static_margin_data),
                np.isnan(sweep_25_data),
                np.isnan(aspect_ratio_data),
                np.isnan(drag_effect_data),
            )
        )
        static_margin_data = static_margin_data[np.logical_not(errors)].tolist()
        sweep_25_data = sweep_25_data[np.logical_not(errors)].tolist()
        aspect_ratio_data = aspect_ratio_data[np.logical_not(errors)].tolist()
        drag_effect_data = drag_effect_data[np.logical_not(errors)].tolist()

        if float(static_margin) != np.clip(
            float(static_margin), min(static_margin_data), max(static_margin_data)
        ):
            _LOGGER.warning("Static margin is outside of the range in Roskam's book, value clipped")
        if float(sweep_25) != np.clip(float(sweep_25), min(sweep_25_data), max(sweep_25_data)):
            _LOGGER.warning(
                "Sweep at 25% chord is outside of the range in Roskam's book, value clipped"
            )
        if float(aspect_ratio) != np.clip(
            float(aspect_ratio), min(aspect_ratio_data), max(aspect_ratio_data)
        ):
            _LOGGER.warning("Aspect ratio is outside of the range in Roskam's book, value clipped")

        # Linear interpolation is preferred but we put the nearest one as protection
        drag_effect = interpolate.griddata(
            (static_margin_data, sweep_25_data, aspect_ratio_data),
            drag_effect_data,
            np.array([static_margin, sweep_25, aspect_ratio]).T,
            method="linear",
        )
        if np.isnan(drag_effect):
            drag_effect = interpolate.griddata(
                (static_margin_data, sweep_25_data, aspect_ratio_data),
                drag_effect_data,
                np.array([static_margin, sweep_25, aspect_ratio]).T,
                method="nearest",
            )

        return float(drag_effect)

    @staticmethod
    def interpolate_database(database, tag_x: str, tag_y: str, input_x: float):

        database_x = database[tag_x]
        database_y = database[tag_y]
        errors = np.logical_or(np.isnan(database_x), np.isnan(database_x))
        database_x = database_x[np.logical_not(errors)].tolist()
        database_y = database_y[np.logical_not(errors)].tolist()

        output_y = interpolate.interp1d(database_x, database_y)(
            np.clip(input_x, min(database_x), max(database_x))
        )

        return output_y


@functools.lru_cache(maxsize=128)
def _k_p(taper_ratio, eta):
    """
    Model based on Roskam data to account for the partial span flaps factor on the pitch
    moment coefficient (figure 8.105).

    :param taper_ratio: lifting surface taper ratio.
    :param eta: start or end of the control surface, in percent of the lifting surface span.
    :return k_p: partial span factor.
    """

    k_p = (
        0.00602
        + 3.47583 * eta
        - 0.11155 * taper_ratio
        - 2.54352 * eta ** 2.0
        + 0.09743 * taper_ratio ** 2.0
        + 0.18193 * taper_ratio ** 2.0 * eta ** 6.0
        + 1.36086 * taper_ratio ** 2.0 * eta
        - 1.36165 * taper_ratio ** 2.0 * eta ** 2.0
        + 0.30816 * eta ** 5.0
        - 0.87442 * taper_ratio * eta ** 6.0
        - 3.80301 * taper_ratio * eta
        + 3.89409 * taper_ratio * eta ** 2.0
        + 0.38675 * taper_ratio * eta ** 7.0
    )

    return k_p


@functools.lru_cache(maxsize=128)
def _k_b(taper_ratio, eta):
    """
    Model based on Roskam data to estimate the flap span factor Kb (figure 8.52).

    :param eta: position along the wing span of the start or end of the flaps divided by span.
    :param taper_ratio: taper ration of the surface.
    :return: kb factor contribution to 3D lift.
    """

    k_b = (
        10.0
        ** (
            0.00082
            + 1.64509 * np.log10(eta)
            + 3.09929 * np.log10(eta) ** 3.0
            - 479.17688 * np.log10(eta) ** 8.0
            - 2.72758 * np.log10(eta) ** 2.0
            + 52.73155 * np.log10(taper_ratio) ** 4.0 * np.log10(eta)
            - 269.97611 * np.log10(taper_ratio) ** 4.0 * np.log10(eta) ** 2.0
            + 29.22030 * np.log10(taper_ratio) ** 8.0
            - 9.19705 * np.log10(taper_ratio) ** 2.0 * np.log10(eta)
            + 51.73734 * np.log10(taper_ratio) ** 2.0 * np.log10(eta) ** 2.0
            - 68.23377 * np.log10(taper_ratio) ** 2.0 * np.log10(eta) ** 3.0
            + 953.14945 * np.log10(taper_ratio) ** 4.0 * np.log10(eta) ** 4.0
            - 0.00474 * np.log10(taper_ratio)
        )
        - 1.0
    )

    return k_b
