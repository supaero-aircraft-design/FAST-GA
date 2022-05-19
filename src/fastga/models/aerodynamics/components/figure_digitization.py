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
import os.path as pth

import numpy as np
import openmdao.api as om
from pandas import read_csv
from scipy import interpolate

from . import resources

DELTA_CD_PLAIN_FLAP = "delta_drag_plain_flap.csv"
K_PLAIN_FLAP = "k_plain_flap.csv"
CL_DELTA_TH_PLAIN_FLAP = "cl_delta_th_plain_flap.csv"
K_CL_DELTA_PLAIN_FLAP = "k_cl_delta_plain_flap.csv"
K_SINGLE_SLOT = "k_single_slot.csv"
BASE_INCREMENT_CL_MAX = "base_increment.csv"
K1 = "k1.csv"
K2 = "k2.csv"
K3 = "k3.csv"
KB_FLAPS = "kb_flaps.csv"
A_DELTA_AIRFOIL = "a_delta_airfoil.csv"
K_A_DELTA = "k_a_delta.csv"
K_P_FLAPS = "k_p.csv"
DELTA_CM_DELTA_CL_REF = "delta_cm_delta_cl_ref.csv"
K_DELTA = "k_delta.csv"
K_AR_FUSELAGE = "k_ar_fuselage.csv"
K_VH = "k_vh.csv"
K_CH_ALPHA = "k_ch_alpha.csv"
CH_ALPHA_TH = "ch_alpha_th.csv"
K_CH_DELTA = "k_ch_delta.csv"
CH_DELTA_TH = "ch_delta_th.csv"

K_FUS = "k_fus.csv"

_LOGGER = logging.getLogger(__name__)


class FigureDigitization(om.ExplicitComponent):
    """Provides lift and drag increments due to high-lift devices."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.phase = None

    @staticmethod
    def delta_cd_plain_flap(chord_ratio, control_deflection) -> float:
        """
        Roskam data to account for the profile drag increment due to the deployment of plain flap
        (figure 4.44).

        :param chord_ratio: control surface over lifting surface ratio.
        :param control_deflection: control surface deflection, in deg.
        :return delta_cd_flap: profile drag increment due to the deployment of flaps.
        """

        file = pth.join(resources.__path__[0], DELTA_CD_PLAIN_FLAP)
        db = read_csv(file)

        x_15 = db["DELTA_F_15_X"]
        y_15 = db["DELTA_F_15_Y"]
        errors = np.logical_or(np.isnan(x_15), np.isnan(y_15))
        x_15 = x_15[np.logical_not(errors)].tolist()
        y_15 = y_15[np.logical_not(errors)].tolist()

        x_60 = db["DELTA_F_60_X"]
        y_60 = db["DELTA_F_60_Y"]
        errors = np.logical_or(np.isnan(x_60), np.isnan(y_60))
        x_60 = x_60[np.logical_not(errors)].tolist()
        y_60 = y_60[np.logical_not(errors)].tolist()

        if chord_ratio != np.clip(
            chord_ratio, min(min(x_15), min(x_60)), max(max(x_15), max(x_60))
        ):
            _LOGGER.warning("Chord ratio outside of the range in Roskam's book, value clipped")

        x_value_00 = 0.0
        x_value_15 = interpolate.interp1d(x_15, y_15)(
            np.clip(float(chord_ratio), min(x_15), max(x_15))
        )
        x_value_60 = interpolate.interp1d(x_60, y_60)(
            np.clip(float(chord_ratio), min(x_60), max(x_60))
        )

        if control_deflection != np.clip(control_deflection, 0.0, 60.0):
            _LOGGER.warning(
                "Control surface deflection outside of the range in Roskam's book, value clipped"
            )

        delta_cd_flap = interpolate.interp1d(
            [0.0, 15.0, 60.0], [x_value_00, x_value_15, x_value_60], kind="quadratic"
        )(np.clip(control_deflection, 0.0, 60.0))

        return delta_cd_flap

    @staticmethod
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
    def cl_delta_theory_plain_flap(thickness, chord_ratio):
        """
        Roskam data to estimate the theoretical airfoil lift effectiveness of a plain flap (
        figure 8.14).

        :param thickness: the airfoil thickness.
        :param chord_ratio: flap chord over wing chord ratio.
        :return cl_delta: theoretical airfoil lift effectiveness of the plain flap.
        """

        file = pth.join(resources.__path__[0], CL_DELTA_TH_PLAIN_FLAP)
        db = read_csv(file)

        x_0 = db["X_0"]
        y_0 = db["Y_0"]
        errors = np.logical_or(np.isnan(x_0), np.isnan(y_0))
        x_0 = x_0[np.logical_not(errors)].tolist()
        y_0 = y_0[np.logical_not(errors)].tolist()
        x_04 = db["X_04"]
        y_04 = db["Y_04"]
        errors = np.logical_or(np.isnan(x_04), np.isnan(y_04))
        x_04 = x_04[np.logical_not(errors)].tolist()
        y_04 = y_04[np.logical_not(errors)].tolist()
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
        cld_thk0 = interpolate.interp1d(x_0, y_0)
        cld_thk04 = interpolate.interp1d(x_04, y_04)
        cld_thk10 = interpolate.interp1d(x_10, y_10)
        cld_thk15 = interpolate.interp1d(x_15, y_15)

        if (
            (chord_ratio != np.clip(chord_ratio, min(x_0), max(x_0)))
            or (chord_ratio != np.clip(chord_ratio, min(x_04), max(x_04)))
            or (chord_ratio != np.clip(chord_ratio, min(x_10), max(x_10)))
            or (chord_ratio != np.clip(chord_ratio, min(x_15), max(x_15)))
        ):
            _LOGGER.warning(
                "Chord ratio value outside of the range in Roskam's book, value clipped"
            )

        cld_t = [
            float(cld_thk0(np.clip(chord_ratio, min(x_0), max(x_0)))),
            float(cld_thk04(np.clip(chord_ratio, min(x_04), max(x_04)))),
            float(cld_thk10(np.clip(chord_ratio, min(x_10), max(x_10)))),
            float(cld_thk15(np.clip(chord_ratio, min(x_15), max(x_15)))),
        ]

        if thickness != np.clip(thickness, 0.0, 0.15):
            _LOGGER.warning(
                "Thickness ratio value outside of the range in Roskam's book, value clipped"
            )

        cl_delta_th = interpolate.interp1d([0.0, 0.04, 0.1, 0.15], cld_t)(
            np.clip(thickness, 0.0, 0.15)
        )

        return cl_delta_th

    @staticmethod
    def k_cl_delta_plain_flap(thickness_ratio, airfoil_lift_coefficient, chord_ratio):
        """
        Roskam data to estimate the correction factor to estimate difference from theoretical
        plain flap lift (figure 8.15).

        :param thickness_ratio: airfoil thickness ratio.
        :param airfoil_lift_coefficient: the lift coefficient of the airfoil, in rad**-1.
        :param chord_ratio: flap chord over wing chord ratio.
        :return k_cl_delta: correction factor to account for difference from theoretical plain
        flap lift.
        """

        file = pth.join(resources.__path__[0], K_CL_DELTA_PLAIN_FLAP)
        db = read_csv(file)

        # Figure 10.64 b
        cl_alpha_th = 6.3 + np.clip(thickness_ratio, 0.0, 0.2) / 0.2 * (7.3 - 6.3)

        k_cl_alpha_data = db["K_CL_ALPHA"]
        errors = np.isnan(k_cl_alpha_data)
        k_cl_alpha_data = k_cl_alpha_data[np.logical_not(errors)].tolist()
        k_cl_delta_min_data = db["K_CL_DELTA_MIN"]
        errors = np.isnan(k_cl_delta_min_data)
        k_cl_delta_min_data = k_cl_delta_min_data[np.logical_not(errors)].tolist()
        k_cl_delta_max_data = db["K_CL_DELTA_MAX"]
        errors = np.isnan(k_cl_delta_max_data)
        k_cl_delta_max_data = k_cl_delta_max_data[np.logical_not(errors)].tolist()

        if float(airfoil_lift_coefficient / cl_alpha_th) != np.clip(
            float(airfoil_lift_coefficient / cl_alpha_th),
            min(k_cl_alpha_data),
            max(k_cl_alpha_data),
        ):
            _LOGGER.warning(
                "Airfoil lift slope ratio value outside of the range in Roskam's book, "
                "value clipped"
            )

        k_cl_alpha = np.clip(
            float(airfoil_lift_coefficient / cl_alpha_th),
            min(k_cl_alpha_data),
            max(k_cl_alpha_data),
        )

        k_cl_delta_min = interpolate.interp1d(k_cl_alpha_data, k_cl_delta_min_data)(k_cl_alpha)
        k_cl_delta_max = interpolate.interp1d(k_cl_alpha_data, k_cl_delta_max_data)(k_cl_alpha)

        if chord_ratio != np.clip(chord_ratio, 0.05, 0.5):
            _LOGGER.warning(
                "Chord ratio value outside of the range in Roskam's book, value clipped"
            )

        chord_ratio = np.clip(chord_ratio, 0.05, 0.5)
        k_cl_delta = interpolate.interp1d([0.05, 0.5], [k_cl_delta_min, k_cl_delta_max])(
            chord_ratio
        )

        return k_cl_delta

    @staticmethod
    def k_prime_single_slotted(flap_angle, chord_ratio):
        """
        Roskam data to estimate the lift effectiveness of a single slotted flap (figure 8.17),
        noted here k_prime to match the notation of the plain flap but is written alpha_delta in
        the book.

        :param flap_angle: the control surface deflection angle angle (in °).
        :param chord_ratio: control surface chord over lifting surface chord ratio.
        :return k_prime: lift effectiveness factor of a single slotted flap.
        """

        file = pth.join(resources.__path__[0], K_SINGLE_SLOT)
        db = read_csv(file)

        x_15 = db["X_15"]
        y_15 = db["Y_15"]
        errors = np.logical_or(np.isnan(x_15), np.isnan(y_15))
        x_15 = x_15[np.logical_not(errors)].tolist()
        y_15 = y_15[np.logical_not(errors)].tolist()
        k_chord_15 = interpolate.interp1d(x_15, y_15)

        x_20 = db["X_20"]
        y_20 = db["Y_20"]
        errors = np.logical_or(np.isnan(x_20), np.isnan(y_20))
        x_20 = x_20[np.logical_not(errors)].tolist()
        y_20 = y_20[np.logical_not(errors)].tolist()
        k_chord_20 = interpolate.interp1d(x_20, y_20)

        x_25 = db["X_25"]
        y_25 = db["Y_25"]
        errors = np.logical_or(np.isnan(x_25), np.isnan(y_25))
        x_25 = x_25[np.logical_not(errors)].tolist()
        y_25 = y_25[np.logical_not(errors)].tolist()
        k_chord_25 = interpolate.interp1d(x_25, y_25)

        x_30 = db["X_30"]
        y_30 = db["Y_30"]
        errors = np.logical_or(np.isnan(x_30), np.isnan(y_30))
        x_30 = x_30[np.logical_not(errors)].tolist()
        y_30 = y_30[np.logical_not(errors)].tolist()
        k_chord_30 = interpolate.interp1d(x_30, y_30)

        x_40 = db["X_40"]
        y_40 = db["Y_40"]
        errors = np.logical_or(np.isnan(x_40), np.isnan(y_40))
        x_40 = x_40[np.logical_not(errors)].tolist()
        y_40 = y_40[np.logical_not(errors)].tolist()
        k_chord_40 = interpolate.interp1d(x_40, y_40)

        if (
            (float(flap_angle) != np.clip(float(flap_angle), min(x_15), max(x_15)))
            or (float(flap_angle) != np.clip(float(flap_angle), min(x_20), max(x_20)))
            or (float(flap_angle) != np.clip(float(flap_angle), min(x_25), max(x_25)))
            or (float(flap_angle) != np.clip(float(flap_angle), min(x_30), max(x_30)))
            or (float(flap_angle) != np.clip(float(flap_angle), min(x_40), max(x_40)))
        ):
            _LOGGER.warning("Flap angle value outside of the range in Roskam's book, value clipped")

        k_chord = [
            k_chord_15(np.clip(float(flap_angle), min(x_15), max(x_15))),
            k_chord_20(np.clip(float(flap_angle), min(x_20), max(x_20))),
            k_chord_25(np.clip(float(flap_angle), min(x_25), max(x_25))),
            k_chord_30(np.clip(float(flap_angle), min(x_30), max(x_30))),
            k_chord_40(np.clip(float(flap_angle), min(x_40), max(x_40))),
        ]

        if float(chord_ratio) != np.clip(float(chord_ratio), 0.15, 0.4):
            _LOGGER.warning(
                "Chord ratio value outside of the range in Roskam's book, value clipped"
            )

        k_prime = float(
            interpolate.interp1d([0.15, 0.20, 0.25, 0.3, 0.4], k_chord)(
                np.clip(float(chord_ratio), 0.15, 0.4)
            )
        )

        return k_prime

    @staticmethod
    def base_max_lift_increment(thickness_ratio: float, flap_type: float) -> float:
        """
        Roskam data to estimate base lift increment used in the computation of flap delta_cl_max
        (figure 8.31).

        :param thickness_ratio: thickness ratio f the lifting surface, in %.
        :param flap_type: type of flap used as described in Roskam, for now can be 0.0 for plain,
        1.0 for single_slot.
        :return: delta_cl_base.
        """

        file = pth.join(resources.__path__[0], BASE_INCREMENT_CL_MAX)
        db = read_csv(file)

        x_plain = db["X_PLAIN_FLAP"]
        y_plain = db["Y_PLAIN_FLAP"]
        errors = np.logical_or(np.isnan(x_plain), np.isnan(y_plain))
        x_plain = x_plain[np.logical_not(errors)].tolist()
        y_plain = y_plain[np.logical_not(errors)].tolist()

        x_single_slot = db["X_SINGLE_SLOT"]
        y_single_slot = db["Y_SINGLE_SLOT"]
        errors = np.logical_or(np.isnan(x_single_slot), np.isnan(y_single_slot))
        x_single_slot = x_single_slot[np.logical_not(errors)].tolist()
        y_single_slot = y_single_slot[np.logical_not(errors)].tolist()

        if flap_type == 0.0:
            base_increment = interpolate.interp1d(x_plain, y_plain)
            if thickness_ratio != np.clip(thickness_ratio, min(x_plain), max(x_plain)):
                _LOGGER.warning(
                    "Thickness ratio value outside of the range in Roskam's book, value clipped"
                )
            delta_cl_max_base = base_increment(np.clip(thickness_ratio, min(x_plain), max(x_plain)))
        elif flap_type == 1.0:
            base_increment = interpolate.interp1d(x_single_slot, y_single_slot)
            if thickness_ratio != np.clip(thickness_ratio, min(x_single_slot), max(x_single_slot)):
                _LOGGER.warning(
                    "Thickness ratio value outside of the range in Roskam's book, value clipped"
                )
            delta_cl_max_base = base_increment(
                np.clip(thickness_ratio, min(x_single_slot), max(x_single_slot))
            )
        else:
            _LOGGER.warning("Flap type not recognized, used plain flap instead")
            base_increment = interpolate.interp1d(x_plain, y_plain)
            if thickness_ratio != np.clip(thickness_ratio, min(x_plain), max(x_plain)):
                _LOGGER.warning(
                    "Thickness ratio value outside of the range in Roskam's book, value clipped"
                )
            delta_cl_max_base = base_increment(np.clip(thickness_ratio, min(x_plain), max(x_plain)))

        return delta_cl_max_base

    @staticmethod
    def k1_max_lift(chord_ratio, flap_type) -> float:
        """
        Roskam data to correct the base lift increment to account for chord ratio difference wrt
        to the reference flap configuration (figure 8.32).

        :param chord_ratio: ration of the chord of the control surface over that of the whole
        surface, in %.
        :param flap_type: type of flap used as described in Roskam, for now can be 0.0 for plain,
        1.0 for single_slot.

        :return k1: correction factor to account for chord ratio difference wrt the reference
        configuration.
        """

        file = pth.join(resources.__path__[0], K1)
        db = read_csv(file)

        if flap_type == 1.0 or flap_type == 0.0:
            x = db["X_PLAIN_SINGLE_SPLIT"]
            y = db["Y_PLAIN_SINGLE_SPLIT"]
            errors = np.logical_or(np.isnan(x), np.isnan(y))
            x = x[np.logical_not(errors)].tolist()
            y = y[np.logical_not(errors)].tolist()
        else:
            _LOGGER.warning("Flap type not recognized, used plain flap instead")
            x = db["X_PLAIN_SINGLE_SPLIT"]
            y = db["Y_PLAIN_SINGLE_SPLIT"]
            errors = np.logical_or(np.isnan(x), np.isnan(y))
            x = x[np.logical_not(errors)].tolist()
            y = y[np.logical_not(errors)].tolist()

        if float(chord_ratio) != np.clip(float(chord_ratio), min(x), max(x)):
            _LOGGER.warning(
                "Chord ratio value outside of the range in Roskam's book, value clipped"
            )

        k1 = interpolate.interp1d(x, y)(np.clip(float(chord_ratio), min(x), max(x)))

        return k1

    @staticmethod
    def k2_max_lift(angle, flap_type) -> float:
        """
        Roskam data to correct the base lift increment to account for the control surface
        deflection angle difference wrt to the reference flap configuration (figure 8.33).

        :param angle: control surface deflection angle, in °.
        :param flap_type: type of flap used as described in Roskam, for now can be 0.0 for plain,
        1.0 for single_slot.
        :return k2: correction factor to account for the control surface deflection angle wrt the
        reference configuration.
        """

        file = pth.join(resources.__path__[0], K2)
        db = read_csv(file)

        x_plain = db["X_PLAIN_FLAP"]
        y_plain = db["Y_PLAIN_FLAP"]
        errors = np.logical_or(np.isnan(x_plain), np.isnan(y_plain))
        x_plain = x_plain[np.logical_not(errors)].tolist()
        y_plain = y_plain[np.logical_not(errors)].tolist()

        x_single_slot = db["X_SINGLE_SLOT"]
        y_single_slot = db["Y_SINGLE_SLOT"]
        errors = np.logical_or(np.isnan(x_single_slot), np.isnan(y_single_slot))
        x_single_slot = x_single_slot[np.logical_not(errors)].tolist()
        y_single_slot = y_single_slot[np.logical_not(errors)].tolist()

        if flap_type == 0.0:
            k2_interp = interpolate.interp1d(x_plain, y_plain)
            if angle != np.clip(angle, min(x_plain), max(x_plain)):
                _LOGGER.warning(
                    "Control surface deflection value outside of the range in Roskam's book, "
                    "value clipped"
                )
            k2 = k2_interp(np.clip(angle, min(x_plain), max(x_plain)))
        elif flap_type == 1.0:
            k2_interp = interpolate.interp1d(x_single_slot, y_single_slot)
            if angle != np.clip(angle, min(x_single_slot), max(x_single_slot)):
                _LOGGER.warning(
                    "Control surface deflection value outside of the range in Roskam's book, "
                    "value clipped"
                )
            k2 = k2_interp(np.clip(angle, min(x_single_slot), max(x_single_slot)))
        else:
            _LOGGER.warning("Flap type not recognized, used plain flap instead")
            k2_interp = interpolate.interp1d(x_plain, y_plain)
            if angle != np.clip(angle, min(x_plain), max(x_plain)):
                _LOGGER.warning(
                    "Control surface deflection value outside of the range in Roskam's book, "
                    "value clipped"
                )
            k2 = k2_interp(np.clip(angle, min(x_plain), max(x_plain)))

        return k2

    @staticmethod
    def k3_max_lift(angle, flap_type) -> float:
        """
        Roskam data for flap motion correction factor (figure 8.34).

        :param angle: control surface deflection angle, in °.
        :param flap_type: type of flap used as described in Roskam, for now can be 0.0 for plain,
        1.0 for single_slot.
        :return k3: correction factor to account flap motion correction.
        """

        file = pth.join(resources.__path__[0], K3)
        db = read_csv(file)

        if flap_type == 0.0:
            k3 = 1.0
        elif flap_type == 1.0:
            x = db["X_SINGLE_SLOT"]
            y = db["Y_SINGLE_SLOT"]
            errors = np.logical_or(np.isnan(x), np.isnan(y))
            x = x[np.logical_not(errors)].tolist()
            y = y[np.logical_not(errors)].tolist()
            reference_angle = 45.0
            if float(angle / reference_angle) != np.clip(
                float(angle / reference_angle), min(x), max(x)
            ):
                _LOGGER.warning(
                    "Control surface deflection value outside of the range in Roskam's book, "
                    "value clipped, reference value is %f",
                    reference_angle,
                )
            k3 = interpolate.interp1d(x, y)(np.clip(float(angle / reference_angle), min(x), max(x)))
        else:
            _LOGGER.warning("Flap type not recognized, used plain flap instead")
            k3 = 1.0

        return k3

    @staticmethod
    def k_b_flaps(eta_in: float, eta_out: float, taper_ratio: float) -> float:
        """
        Roskam data to estimate the flap span factor Kb (figure 8.52) This factor accounts for a
        finite flap contribution to the 3D lift increase, depending on its position and size and
        the taper ratio of the wing.

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

        taper_ratio = np.clip(taper_ratio, 0.0, 1.0)
        file = pth.join(resources.__path__[0], KB_FLAPS)
        db = read_csv(file)

        x_0 = db["X_0"]
        y_0 = db["Y_0"]
        errors = np.logical_or(np.isnan(x_0), np.isnan(y_0))
        x_0 = x_0[np.logical_not(errors)].tolist()
        y_0 = y_0[np.logical_not(errors)].tolist()
        x_05 = db["X_0.5"]
        y_05 = db["Y_0.5"]
        errors = np.logical_or(np.isnan(x_05), np.isnan(y_05))
        x_05 = x_05[np.logical_not(errors)].tolist()
        y_05 = y_05[np.logical_not(errors)].tolist()
        x_1 = db["X_1"]
        y_1 = db["Y_1"]
        errors = np.logical_or(np.isnan(x_1), np.isnan(y_1))
        x_1 = x_1[np.logical_not(errors)].tolist()
        y_1 = y_1[np.logical_not(errors)].tolist()
        k_taper0 = interpolate.interp1d(x_0, y_0)
        k_taper05 = interpolate.interp1d(x_05, y_05)
        k_taper1 = interpolate.interp1d(x_1, y_1)

        if (
            (eta_in != np.clip(eta_in, min(x_0), max(x_0)))
            or (eta_in != np.clip(eta_in, min(x_05), max(x_05)))
            or (eta_in != np.clip(eta_in, min(x_1), max(x_1)))
        ):
            _LOGGER.warning(
                "Flap inward position ratio value outside of the range in Roskam's book, "
                "value clipped"
            )

        k_eta = [
            float(k_taper0(np.clip(eta_in, min(x_0), max(x_0)))),
            float(k_taper05(np.clip(eta_in, min(x_05), max(x_05)))),
            float(k_taper1(np.clip(eta_in, min(x_1), max(x_1)))),
        ]
        kb_in = interpolate.interp1d([0.0, 0.5, 1.0], k_eta)(taper_ratio)

        if (
            (eta_out != np.clip(eta_out, min(x_0), max(x_0)))
            or (eta_out != np.clip(eta_out, min(x_05), max(x_05)))
            or (eta_out != np.clip(eta_out, min(x_1), max(x_1)))
        ):
            _LOGGER.warning(
                "Flap inward position ratio value outside of the range in Roskam's book, "
                "value clipped"
            )

        k_eta = [
            float(k_taper0(np.clip(eta_out, min(x_0), max(x_0)))),
            float(k_taper05(np.clip(eta_out, min(x_05), max(x_05)))),
            float(k_taper1(np.clip(eta_out, min(x_1), max(x_1)))),
        ]
        kb_out = interpolate.interp1d([0.0, 0.5, 1.0], k_eta)(taper_ratio)

        return float(kb_out - kb_in)

    @staticmethod
    def a_delta_airfoil(chord_ratio) -> float:
        """
        Roskam data to estimate the two-dimensional flap effectiveness factor (figure 8.53a) This
        factor can be then used in the computation of the 3D flap effectiveness factor which is
        often the coefficient of interest.

        :param chord_ratio: ration of the chord of the control surface over that of the whole
        surface.
        :return: kb factor contribution to 3D lift.
        """

        file = pth.join(resources.__path__[0], A_DELTA_AIRFOIL)
        db = read_csv(file)

        x = db["X"]
        y = db["Y"]
        errors = np.logical_or(np.isnan(x), np.isnan(y))
        x = x[np.logical_not(errors)].tolist()
        y = y[np.logical_not(errors)].tolist()

        if chord_ratio != np.clip(chord_ratio, 0.0, 1.0):
            _LOGGER.warning(
                "Chord ratio value outside of the range in Roskam's book, value clipped"
            )

        a_delta = interpolate.interp1d(x, y)(np.clip(float(chord_ratio), 0.0, 1.0))

        return a_delta

    @staticmethod
    def k_a_delta(a_delta_airfoil, aspect_ratio) -> float:
        """
        Roskam data to estimate the two dimensional to three dimensional control surface lift
        effectiveness parameter (figure 8.53b).

        :param a_delta_airfoil: control surface two-dimensional flap effectiveness factor.
        :param aspect_ratio: aspect ratio of the fixed surface.
        :return k_a_delta: two dimensional to three dimensional control surface lift effectiveness
        parameter.
        """

        file = pth.join(resources.__path__[0], K_A_DELTA)
        db = read_csv(file)

        if float(aspect_ratio) != np.clip(float(aspect_ratio), 0.0, 10.0):
            _LOGGER.warning(
                "Aspect ratio value outside of the range in Roskam's book, value clipped"
            )

        x_01 = db["X_01"]
        y_01 = db["Y_01"]
        errors = np.logical_or(np.isnan(x_01), np.isnan(y_01))
        x_01 = x_01[np.logical_not(errors)].tolist()
        y_01 = y_01[np.logical_not(errors)].tolist()
        y1 = interpolate.interp1d(x_01, y_01)(np.clip(float(aspect_ratio), min(x_01), max(x_01)))

        x_02 = db["X_02"]
        y_02 = db["Y_02"]
        errors = np.logical_or(np.isnan(x_02), np.isnan(y_02))
        x_02 = x_02[np.logical_not(errors)].tolist()
        y_02 = y_02[np.logical_not(errors)].tolist()
        y2 = interpolate.interp1d(x_02, y_02)(np.clip(float(aspect_ratio), min(x_02), max(x_02)))
        x_03 = db["X_03"]
        y_03 = db["Y_03"]
        errors = np.logical_or(np.isnan(x_03), np.isnan(y_03))
        x_03 = x_03[np.logical_not(errors)].tolist()
        y_03 = y_03[np.logical_not(errors)].tolist()
        y3 = interpolate.interp1d(x_03, y_03)(np.clip(float(aspect_ratio), min(x_03), max(x_03)))

        x_04 = db["X_04"]
        y_04 = db["Y_04"]
        errors = np.logical_or(np.isnan(x_04), np.isnan(y_04))
        x_04 = x_04[np.logical_not(errors)].tolist()
        y_04 = y_04[np.logical_not(errors)].tolist()
        y4 = interpolate.interp1d(x_04, y_04)(np.clip(float(aspect_ratio), min(x_04), max(x_04)))

        x_05 = db["X_05"]
        y_05 = db["Y_05"]
        errors = np.logical_or(np.isnan(x_05), np.isnan(y_05))
        x_05 = x_05[np.logical_not(errors)].tolist()
        y_05 = y_05[np.logical_not(errors)].tolist()
        y5 = interpolate.interp1d(x_05, y_05)(np.clip(float(aspect_ratio), min(x_05), max(x_05)))

        x_06 = db["X_06"]
        y_06 = db["Y_06"]
        errors = np.logical_or(np.isnan(x_06), np.isnan(y_06))
        x_06 = x_06[np.logical_not(errors)].tolist()
        y_06 = y_06[np.logical_not(errors)].tolist()
        y6 = interpolate.interp1d(x_06, y_06)(np.clip(float(aspect_ratio), min(x_06), max(x_06)))

        x_07 = db["X_07"]
        y_07 = db["Y_07"]
        errors = np.logical_or(np.isnan(x_07), np.isnan(y_07))
        x_07 = x_07[np.logical_not(errors)].tolist()
        y_07 = y_07[np.logical_not(errors)].tolist()
        y7 = interpolate.interp1d(x_07, y_07)(np.clip(float(aspect_ratio), min(x_07), max(x_07)))

        x_08 = db["X_08"]
        y_08 = db["Y_08"]
        errors = np.logical_or(np.isnan(x_08), np.isnan(y_08))
        x_08 = x_08[np.logical_not(errors)].tolist()
        y_08 = y_08[np.logical_not(errors)].tolist()
        y8 = interpolate.interp1d(x_08, y_08)(np.clip(float(aspect_ratio), min(x_08), max(x_08)))

        x_09 = db["X_09"]
        y_09 = db["Y_09"]
        errors = np.logical_or(np.isnan(x_09), np.isnan(y_09))
        x_09 = x_09[np.logical_not(errors)].tolist()
        y_09 = y_09[np.logical_not(errors)].tolist()
        y9 = interpolate.interp1d(x_09, y_09)(np.clip(float(aspect_ratio), min(x_09), max(x_09)))

        x_10 = db["X_10"]
        y_10 = db["Y_10"]
        errors = np.logical_or(np.isnan(x_10), np.isnan(y_10))
        x_10 = x_10[np.logical_not(errors)].tolist()
        y_10 = y_10[np.logical_not(errors)].tolist()
        y10 = interpolate.interp1d(x_10, y_10)(np.clip(float(aspect_ratio), min(x_10), max(x_10)))

        x = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        y = [y1, y2, y3, y4, y5, y6, y7, y8, y9, y10]

        if float(a_delta_airfoil) != np.clip(float(a_delta_airfoil), 0.0, 1.0):
            _LOGGER.warning(
                "Control surface effectiveness ratio value outside of the range in "
                "Roskam's book, value clipped"
            )

        k_a_delta = interpolate.interp1d(x, y)(np.clip(float(a_delta_airfoil), 0.1, 1.0))

        return k_a_delta

    @staticmethod
    def x_cp_c_prime(flap_chord_ratio: float) -> float:
        """
        Roskam data to estimate the location of the center of pressure due to Incremental Flap
        Load (figure 8.91).

        :param flap_chord_ratio: ratio of the control surface chord over the lifting surface chord.
        :return x_cp_c_prime: location of center of pressure due to flap deployment.
        """

        # Graph is simple so no csv is read, rather, a direct formula is used.

        if flap_chord_ratio != np.clip(flap_chord_ratio, 0.0, 1.0):
            _LOGGER.warning("Chord ratio outside of the range in Roskam's book, value clipped")

        x_cp_c_prime = interpolate.interp1d([0.0, 1.0], [0.5, 0.25])(
            np.clip(flap_chord_ratio, 0.0, 1.0)
        )

        return x_cp_c_prime

    @staticmethod
    def k_p_flaps(taper_ratio, eta_in, eta_out) -> float:
        """
        Roskam data to account for the partial span flaps factor on the pitch moment coefficient
        (figure 8.105).

        :param taper_ratio: lifting surface taper ratio.
        :param eta_in: start of the control surface, in percent of the lifting surface span.
        :param eta_out: end of the control surface, in percent of the lifting surface span.
        :return k_p: partial span factor.
        """

        file = pth.join(resources.__path__[0], K_P_FLAPS)
        db = read_csv(file)

        eta_in_1_0 = FigureDigitization.interpolate_database(
            db, "taper_1_0_X", "taper_1_0_Y", eta_in
        )
        eta_out_1_0 = FigureDigitization.interpolate_database(
            db, "taper_1_0_X", "taper_1_0_Y", eta_out
        )

        eta_in_0_5 = FigureDigitization.interpolate_database(
            db, "taper_0_5_X", "taper_0_5_Y", eta_in
        )
        eta_out_0_5 = FigureDigitization.interpolate_database(
            db, "taper_0_5_X", "taper_0_5_Y", eta_out
        )

        eta_in_0_333 = FigureDigitization.interpolate_database(
            db, "taper_0_333_X", "taper_0_333_Y", eta_in
        )
        eta_out_0_333 = FigureDigitization.interpolate_database(
            db, "taper_0_333_X", "taper_0_333_Y", eta_out
        )

        eta_in_0_25 = FigureDigitization.interpolate_database(
            db, "taper_0_25_X", "taper_0_25_Y", eta_in
        )
        eta_out_0_25 = FigureDigitization.interpolate_database(
            db, "taper_0_25_X", "taper_0_25_Y", eta_out
        )

        taper_array = [0.25, 0.333, 0.5, 1.0]
        eta_in_array = [eta_in_0_25, eta_in_0_333, eta_in_0_5, eta_in_1_0]
        eta_out_array = [eta_out_0_25, eta_out_0_333, eta_out_0_5, eta_out_1_0]

        if taper_ratio != np.clip(taper_ratio, 0.25, 1.0):
            _LOGGER.warning("Taper ratio outside of the range in Roskam's book, value clipped")

        k_p = interpolate.interp1d(taper_array, eta_out_array)(
            np.clip(taper_ratio, 0.25, 1.0)
        ) - interpolate.interp1d(taper_array, eta_in_array)(np.clip(taper_ratio, 0.25, 1.0))

        return k_p

    @staticmethod
    def pitch_to_reference_lift(thickness_ratio: float, chord_ratio: float) -> float:
        """
        Roskam data to account for the ratio between the pitch moment coefficient and the
        reference lift coefficient increment (figure 8.106).

        :param thickness_ratio: thickness to chord ratio of the lifting surface.
        :param chord_ratio: chord ratio of the control surface over the lifting surface.
        :return delta_cm_delta_cl_ref: ration between the pitching moment and the reference lift
        coefficient.
        """

        if chord_ratio != np.clip(chord_ratio, 0.05, 0.4):
            _LOGGER.warning("Chord ratio outside of the range in Roskam's book, value clipped")

        file = pth.join(resources.__path__[0], DELTA_CM_DELTA_CL_REF)
        db = read_csv(file)

        x_21 = db["TOC_21_X"]
        y_21 = db["TOC_21_Y"]
        errors = np.logical_or(np.isnan(x_21), np.isnan(y_21))
        x_21 = x_21[np.logical_not(errors)].tolist()
        y_21 = y_21[np.logical_not(errors)].tolist()
        k_21 = interpolate.interp1d(x_21, y_21)(np.clip(chord_ratio, min(x_21), max(x_21)))

        x_18 = db["TOC_18_X"]
        y_18 = db["TOC_18_Y"]
        errors = np.logical_or(np.isnan(x_18), np.isnan(y_18))
        x_18 = x_18[np.logical_not(errors)].tolist()
        y_18 = y_18[np.logical_not(errors)].tolist()
        k_18 = interpolate.interp1d(x_18, y_18)(np.clip(chord_ratio, min(x_18), max(x_18)))

        x_15 = db["TOC_15_X"]
        y_15 = db["TOC_15_Y"]
        errors = np.logical_or(np.isnan(x_15), np.isnan(y_15))
        x_15 = x_15[np.logical_not(errors)].tolist()
        y_15 = y_15[np.logical_not(errors)].tolist()
        k_15 = interpolate.interp1d(x_15, y_15)(np.clip(chord_ratio, min(x_15), max(x_15)))

        x_12 = db["TOC_12_X"]
        y_12 = db["TOC_12_Y"]
        errors = np.logical_or(np.isnan(x_12), np.isnan(y_12))
        x_12 = x_12[np.logical_not(errors)].tolist()
        y_12 = y_12[np.logical_not(errors)].tolist()
        k_12 = interpolate.interp1d(x_12, y_12)(np.clip(chord_ratio, min(x_12), max(x_12)))

        x_09 = db["TOC_09_X"]
        y_09 = db["TOC_09_Y"]
        errors = np.logical_or(np.isnan(x_09), np.isnan(y_09))
        x_09 = x_09[np.logical_not(errors)].tolist()
        y_09 = y_09[np.logical_not(errors)].tolist()
        k_09 = interpolate.interp1d(x_09, y_09)(np.clip(chord_ratio, min(x_09), max(x_09)))

        x_06 = db["TOC_06_X"]
        y_06 = db["TOC_06_Y"]
        errors = np.logical_or(np.isnan(x_06), np.isnan(y_06))
        x_06 = x_06[np.logical_not(errors)].tolist()
        y_06 = y_06[np.logical_not(errors)].tolist()
        k_06 = interpolate.interp1d(x_06, y_06)(np.clip(chord_ratio, min(x_06), max(x_06)))

        x_03 = db["TOC_03_X"]
        y_03 = db["TOC_03_Y"]
        errors = np.logical_or(np.isnan(x_03), np.isnan(y_03))
        x_03 = x_03[np.logical_not(errors)].tolist()
        y_03 = y_03[np.logical_not(errors)].tolist()
        k_03 = interpolate.interp1d(x_03, y_03)(np.clip(chord_ratio, min(x_03), max(x_03)))

        toc_array = [0.03, 0.06, 0.09, 0.12, 0.15, 0.18, 0.21]
        k_array = [k_03, k_06, k_09, k_12, k_15, k_18, k_21]

        if thickness_ratio != np.clip(thickness_ratio, 0.03, 0.4):
            _LOGGER.warning(
                "Thickness to chord ratio outside of the range in Roskam's book, " "value clipped"
            )

        k = interpolate.interp1d(toc_array, k_array)(np.clip(thickness_ratio, 0.03, 0.4))

        return k

    @staticmethod
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
        Roskam data to estimate the impact of relative area ratio on the effective aspect ratio (
        figure 10.16).

        :param area_ratio: ratio of the horizontal tail area over the vertical tail area.
        :return k_vh: impact of area ratio on effective aspect ratio.
        """

        file = pth.join(resources.__path__[0], K_VH)
        db = read_csv(file)

        x = db["X"]
        y = db["Y"]
        errors = np.logical_or(np.isnan(x), np.isnan(y))
        x = x[np.logical_not(errors)].tolist()
        y = y[np.logical_not(errors)].tolist()

        if float(area_ratio) != np.clip(float(area_ratio), min(x), max(x)):
            _LOGGER.warning("Area ratio value outside of the range in Roskam's book, value clipped")

        k_vh = interpolate.interp1d(x, y)(np.clip(float(area_ratio), min(x), max(x)))

        return k_vh

    @staticmethod
    def k_ch_alpha(thickness_ratio, airfoil_lift_coefficient, chord_ratio):
        """
        Roskam data to compute the correction factor to differentiate the 2D control surface hinge
        moment derivative.
        due to AOA from the reference (figure 10.63).

        :param thickness_ratio: airfoil thickness ratio.
        :param airfoil_lift_coefficient: the lift coefficient of the airfoil, in rad**-1.
        :param chord_ratio: flap chord over wing chord ratio.
        :return k_ch_alpha: correction factor for 2D control surface hinge moment derivative due to
        AOA.
        """

        file = pth.join(resources.__path__[0], K_CH_ALPHA)
        db = read_csv(file)

        # Figure 10.64 b
        if thickness_ratio != np.clip(thickness_ratio, 0.0, 0.2):
            _LOGGER.warning(
                "Thickness ratio value outside of the range in Roskam's book, value clipped"
            )
        cl_alpha_th = 6.3 + np.clip(thickness_ratio, 0.0, 0.2) / 0.2 * (7.3 - 6.3)

        k_cl_alpha_data = db["K_CL_ALPHA"]
        errors = np.isnan(k_cl_alpha_data)
        k_cl_alpha_data = k_cl_alpha_data[np.logical_not(errors)].tolist()
        k_ch_alpha_min_data = db["K_CH_ALPHA_MIN"]
        errors = np.isnan(k_ch_alpha_min_data)
        k_ch_alpha_min_data = k_ch_alpha_min_data[np.logical_not(errors)].tolist()
        k_ch_alpha_max_data = db["K_CH_ALPHA_MAX"]
        errors = np.isnan(k_ch_alpha_max_data)
        k_ch_alpha_max_data = k_ch_alpha_max_data[np.logical_not(errors)].tolist()

        if float(airfoil_lift_coefficient / cl_alpha_th) != np.clip(
            float(airfoil_lift_coefficient / cl_alpha_th),
            min(k_cl_alpha_data),
            max(k_cl_alpha_data),
        ):
            _LOGGER.warning(
                "Airfoil lift coefficient to theoretical lift coefficient ratio value outside of "
                "the range in Roskam's book, value clipped"
            )

        k_cl_alpha = np.clip(
            float(airfoil_lift_coefficient / cl_alpha_th),
            min(k_cl_alpha_data),
            max(k_cl_alpha_data),
        )

        k_ch_alpha_min = interpolate.interp1d(k_cl_alpha_data, k_ch_alpha_min_data)(k_cl_alpha)
        k_ch_alpha_max = interpolate.interp1d(k_cl_alpha_data, k_ch_alpha_max_data)(k_cl_alpha)

        chord_ratio = np.clip(chord_ratio, 0.1, 0.4)
        k_ch_alpha = interpolate.interp1d([0.1, 0.4], [k_ch_alpha_min, k_ch_alpha_max])(chord_ratio)

        return k_ch_alpha

    @staticmethod
    def ch_alpha_th(thickness_ratio, chord_ratio):
        """
        Roskam data to compute the theoretical 2D control surface hinge moment derivative due to
        AOA (figure 10.63).

        :param thickness_ratio: airfoil thickness ratio.
        :param chord_ratio: flap chord over wing chord ratio.
        :return ch_alpha: theoretical hinge moment derivative due to AOA.
        """

        file = pth.join(resources.__path__[0], CH_ALPHA_TH)
        db = read_csv(file)

        thickness_ratio_data = db["THICKNESS_RATIO"]
        errors = np.isnan(thickness_ratio_data)
        thickness_ratio_data = thickness_ratio_data[np.logical_not(errors)].tolist()
        ch_alpha_min_data = db["CH_ALPHA_MIN"]
        errors = np.isnan(ch_alpha_min_data)
        ch_alpha_min_data = ch_alpha_min_data[np.logical_not(errors)].tolist()
        ch_alpha_max_data = db["CH_ALPHA_MAX"]
        errors = np.isnan(ch_alpha_max_data)
        ch_alpha_max_data = ch_alpha_max_data[np.logical_not(errors)].tolist()

        if float(thickness_ratio) != np.clip(
            float(thickness_ratio), min(thickness_ratio_data), max(thickness_ratio_data)
        ):
            _LOGGER.warning(
                "Thickness ratio value outside of the range in Roskam's book, value clipped"
            )

        ch_alpha_min = interpolate.interp1d(thickness_ratio_data, ch_alpha_min_data)(
            np.clip(float(thickness_ratio), min(thickness_ratio_data), max(thickness_ratio_data))
        )
        ch_alpha_max = interpolate.interp1d(thickness_ratio_data, ch_alpha_max_data)(
            np.clip(float(thickness_ratio), min(thickness_ratio_data), max(thickness_ratio_data))
        )

        if chord_ratio != np.clip(chord_ratio, 0.1, 0.4):
            _LOGGER.warning(
                "Chord ratio value outside of the range in Roskam's book, value clipped"
            )

        chord_ratio = np.clip(chord_ratio, 0.1, 0.4)
        ch_alpha_th = interpolate.interp1d([0.1, 0.4], [ch_alpha_min, ch_alpha_max])(chord_ratio)

        return ch_alpha_th

    @staticmethod
    def k_ch_delta(thickness_ratio, airfoil_lift_coefficient, chord_ratio):
        """
        Roskam data to compute the correction factor to differentiate the 2D control surface
        hinge moment derivative due to control surface deflection from the reference (figure
        10.69 a).

        :param thickness_ratio: airfoil thickness ratio.
        :param airfoil_lift_coefficient: the lift coefficient of the airfoil, in rad**-1.
        :param chord_ratio: control surface chord over lifting surface chord ratio.
        :return k_ch_delta: hinge moment derivative due to control surface deflection correction
        factor.
        """

        file = pth.join(resources.__path__[0], K_CH_DELTA)
        db = read_csv(file)

        # Figure 10.64 b
        if thickness_ratio != np.clip(thickness_ratio, 0.0, 0.2):
            _LOGGER.warning(
                "Thickness ratio value outside of the range in Roskam's book, value clipped"
            )

        cl_alpha_th = 6.3 + np.clip(thickness_ratio, 0.0, 0.2) / 0.2 * (7.3 - 6.3)

        k_cl_alpha_data = db["K_CL_ALPHA"]
        errors = np.isnan(k_cl_alpha_data)
        k_cl_alpha_data = k_cl_alpha_data[np.logical_not(errors)].tolist()
        k_ch_delta_min_data = db["K_CH_DELTA_MIN"]
        errors = np.isnan(k_ch_delta_min_data)
        k_ch_delta_min_data = k_ch_delta_min_data[np.logical_not(errors)].tolist()
        k_ch_delta_avg_data = db["K_CH_DELTA_AVG"]
        errors = np.isnan(k_ch_delta_avg_data)
        k_ch_delta_avg_data = k_ch_delta_avg_data[np.logical_not(errors)].tolist()
        k_ch_delta_max_data = db["K_CH_DELTA_MAX"]
        errors = np.isnan(k_ch_delta_max_data)
        k_ch_delta_max_data = k_ch_delta_max_data[np.logical_not(errors)].tolist()

        if float(airfoil_lift_coefficient / cl_alpha_th) != np.clip(
            float(airfoil_lift_coefficient / cl_alpha_th),
            min(k_cl_alpha_data),
            max(k_cl_alpha_data),
        ):
            _LOGGER.warning(
                "Airfoil lift coefficient to theoretical lift coefficient ratio value outside of "
                "the range in Roskam's book, value clipped"
            )

        k_cl_alpha = np.clip(
            float(airfoil_lift_coefficient / cl_alpha_th),
            min(k_cl_alpha_data),
            max(k_cl_alpha_data),
        )

        k_ch_delta_min = interpolate.interp1d(k_cl_alpha_data, k_ch_delta_min_data)(k_cl_alpha)
        k_ch_delta_avg = interpolate.interp1d(k_cl_alpha_data, k_ch_delta_avg_data)(k_cl_alpha)
        k_ch_delta_max = interpolate.interp1d(k_cl_alpha_data, k_ch_delta_max_data)(k_cl_alpha)

        if chord_ratio != np.clip(chord_ratio, 0.1, 0.4):
            _LOGGER.warning(
                "Chord ratio value outside of the range in Roskam's book, value clipped"
            )

        chord_ratio = np.clip(chord_ratio, 0.1, 0.4)
        k_ch_delta = interpolate.interp1d(
            [0.1, 0.25, 0.4], [k_ch_delta_min, k_ch_delta_avg, k_ch_delta_max]
        )(chord_ratio)

        return k_ch_delta

    @staticmethod
    def ch_delta_th(thickness_ratio, chord_ratio):
        """
        Roskam data to compute the theoretical 2D control surface hinge moment derivative due to
        control surface deflection (figure 10.69 b).

        :param thickness_ratio: airfoil thickness ratio.
        :param chord_ratio: flap chord over wing chord ratio.
        :return ch_delta: theoretical hinge moment derivative due to control surface deflection.
        """

        file = pth.join(resources.__path__[0], CH_DELTA_TH)
        db = read_csv(file)

        thickness_ratio_data = db["THICKNESS_RATIO"]
        errors = np.isnan(thickness_ratio_data)
        thickness_ratio_data = thickness_ratio_data[np.logical_not(errors)].tolist()
        ch_delta_min_data = db["CH_DELTA_MIN"]
        errors = np.isnan(ch_delta_min_data)
        ch_delta_min_data = ch_delta_min_data[np.logical_not(errors)].tolist()
        ch_delta_max_data = db["CH_DELTA_MAX"]
        errors = np.isnan(ch_delta_max_data)
        ch_delta_max_data = ch_delta_max_data[np.logical_not(errors)].tolist()

        if float(thickness_ratio) != np.clip(
            float(thickness_ratio), min(thickness_ratio_data), max(thickness_ratio_data)
        ):
            _LOGGER.warning(
                "Thickness ratio value outside of the range in Roskam's book, value clipped"
            )

        ch_delta_min = interpolate.interp1d(thickness_ratio_data, ch_delta_min_data)(
            np.clip(float(thickness_ratio), min(thickness_ratio_data), max(thickness_ratio_data))
        )
        ch_delta_max = interpolate.interp1d(thickness_ratio_data, ch_delta_max_data)(
            np.clip(float(thickness_ratio), min(thickness_ratio_data), max(thickness_ratio_data))
        )

        if chord_ratio != np.clip(chord_ratio, 0.1, 0.4):
            _LOGGER.warning(
                "Chord ratio value outside of the range in Roskam's book, value clipped"
            )

        chord_ratio = np.clip(chord_ratio, 0.1, 0.4)
        ch_delta_th = interpolate.interp1d([0.1, 0.4], [ch_delta_min, ch_delta_max])(chord_ratio)

        return ch_delta_th

    @staticmethod
    def k_fus(root_quarter_chord_position_ratio) -> float:
        """
        Raymer data to estimate the empirical pitching moment factor K_fus (figure 16.14).

        :param root_quarter_chord_position_ratio: the position of the root quarter chord of the
        wing from the nose.
        divided by the total length of the fuselage.
        :return k_fus: the empirical pitching moment factor.
        """

        file = pth.join(resources.__path__[0], K_FUS)
        db = read_csv(file)

        x = db["X_0_25_RATIO"]
        y = db["K_FUS"]
        errors = np.logical_or(np.isnan(x), np.isnan(y))
        x = x[np.logical_not(errors)].tolist()
        y = y[np.logical_not(errors)].tolist()

        if float(root_quarter_chord_position_ratio) != np.clip(
            float(root_quarter_chord_position_ratio), min(x), max(x)
        ):
            _LOGGER.warning(
                "Position of the root quarter-chord as percent of fuselage length is outside of "
                "the range in Roskam's book, value clipped"
            )

        k_fus = interpolate.interp1d(x, y)(
            np.clip(float(root_quarter_chord_position_ratio), min(x), max(x))
        )

        return k_fus

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
