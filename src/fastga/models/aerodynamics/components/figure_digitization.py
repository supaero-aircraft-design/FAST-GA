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
from typing import List

import numpy as np
import openmdao.api as om
import pandas as pd
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
CL_BETA_SWEEP = "cl_beta_sweep_contribution.csv"
K_M_LAMBDA = "sweep_compressibility_correction.csv"
K_FUSELAGE = "cl_beta_fuselage_correction.csv"
CL_BETA_AR = "cl_beta_ar_contribution.csv"
CL_BETA_GAMMA = "cl_beta_dihedral_contribution.csv"
K_M_GAMMA = "dihedral_compressibility_correction.csv"
K_TWIST = "twist_correction.csv"
K_ROLL_DAMPING = "cl_p_roll_damping_parameter.csv"
K_CDI_ROLL_DAMPING = "cl_p_cdi_roll_damping.csv"
CL_R_LIFT_PART_A = "cl_r_lift_effect_part_a.csv"
CL_R_LIFT_PART_B = "cl_r_lift_effect_part_b.csv"
CL_R_TWIST_EFFECT = "cl_r_twist_effect.csv"
CN_DELTA_A_K_A = "cn_delta_a_correlation_cst.csv"
CN_P_TWIST = "cn_p_twist_contribution.csv"
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
        Roskam data to account for the profile drag increment due to the deployment of plain flap
        (figure 4.44).

        :param chord_ratio: control surface over lifting surface ratio.
        :param control_deflection: control surface deflection, in deg.
        :return delta_cd_flap: profile drag increment due to the deployment of flaps.
        """

        file = pth.join(resources.__path__[0], DELTA_CD_PLAIN_FLAP)
        db = pd.read_csv(file)

        x_15, y_15 = filter_nans(db, ["DELTA_F_15_X", "DELTA_F_15_Y"])
        x_60, y_60 = filter_nans(db, ["DELTA_F_60_X", "DELTA_F_60_Y"])

        if chord_ratio != np.clip(
            chord_ratio, min(min(x_15), min(x_60)), max(max(x_15), max(x_60))
        ):
            _LOGGER.warning("Chord ratio outside of the range in Roskam's book, value clipped")

        x_value_00 = 0.0
        x_value_15 = np.interp(np.clip(float(chord_ratio), min(x_15), max(x_15)), x_15, y_15)
        x_value_60 = np.interp(np.clip(float(chord_ratio), min(x_60), max(x_60)), x_60, y_60)

        if control_deflection != np.clip(control_deflection, 0.0, 60.0):
            _LOGGER.warning(
                "Control surface deflection outside of the range in Roskam's book, value clipped"
            )

        delta_cd_flap = float(
            np.polyval(
                np.polyfit([0.0, 15.0, 60.0], [x_value_00, x_value_15, x_value_60], 2),
                np.clip(control_deflection, 0.0, 60.0),
            )
        )

        return delta_cd_flap

    @staticmethod
    @functools.lru_cache(maxsize=128)
    def k_prime_plain_flap(flap_angle, chord_ratio):
        """
        Roskam data to estimate the correction factor to estimate non-linear lift behaviour of
        plain flap (figure 8.13).

        :param flap_angle: the flap angle (in °).
        :param chord_ratio: flap chord over wing chord ratio.
        :return k_prime: correction factor to estimate non-linear lift behaviour of plain flap.
        """

        file = pth.join(resources.__path__[0], K_PLAIN_FLAP)
        db = pd.read_csv(file)

        x_10, y_10 = filter_nans(db, ["X_10", "Y_10"])
        x_15, y_15 = filter_nans(db, ["X_15", "Y_15"])
        x_25, y_25 = filter_nans(db, ["X_25", "Y_25"])
        x_30, y_30 = filter_nans(db, ["X_30", "Y_30"])
        x_40, y_40 = filter_nans(db, ["X_40", "Y_40"])
        x_50, y_50 = filter_nans(db, ["X_50", "Y_50"])

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
            float(np.interp(np.clip(flap_angle, min(x_10), max(x_10)), x_10, y_10)),
            float(np.interp(np.clip(flap_angle, min(x_15), max(x_15)), x_15, y_15)),
            float(np.interp(np.clip(flap_angle, min(x_25), max(x_25)), x_25, y_25)),
            float(np.interp(np.clip(flap_angle, min(x_30), max(x_30)), x_30, y_30)),
            float(np.interp(np.clip(flap_angle, min(x_40), max(x_40)), x_40, y_40)),
            float(np.interp(np.clip(flap_angle, min(x_50), max(x_50)), x_50, y_50)),
        ]

        if chord_ratio != np.clip(chord_ratio, 0.1, 0.5):
            _LOGGER.warning(
                "Chord ratio value outside of the range in Roskam's book, value clipped"
            )

        k_prime = float(
            np.interp(np.clip(chord_ratio, 0.1, 0.5), [0.1, 0.15, 0.25, 0.3, 0.4, 0.5], k_chord)
        )

        return k_prime

    @staticmethod
    @functools.lru_cache(maxsize=128)
    def cl_delta_theory_plain_flap(thickness, chord_ratio):
        """
        Roskam data to estimate the theoretical airfoil lift effectiveness of a plain flap (
        figure 8.14).

        :param thickness: the airfoil thickness.
        :param chord_ratio: flap chord over wing chord ratio.
        :return cl_delta: theoretical airfoil lift effectiveness of the plain flap.
        """

        file = pth.join(resources.__path__[0], CL_DELTA_TH_PLAIN_FLAP)
        db = pd.read_csv(file)

        x_0, y_0 = filter_nans(db, ["X_0", "Y_0"])
        x_04, y_04 = filter_nans(db, ["X_04", "Y_04"])
        x_10, y_10 = filter_nans(db, ["X_10", "Y_10"])
        x_15, y_15 = filter_nans(db, ["X_15", "Y_15"])

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
            float(np.interp(np.clip(chord_ratio, min(x_0), max(x_0)), x_0, y_0)),
            float(np.interp(np.clip(chord_ratio, min(x_04), max(x_04)), x_04, y_04)),
            float(np.interp(np.clip(chord_ratio, min(x_10), max(x_10)), x_10, y_10)),
            float(np.interp(np.clip(chord_ratio, min(x_15), max(x_15)), x_15, y_15)),
        ]

        if thickness != np.clip(thickness, 0.0, 0.15):
            _LOGGER.warning(
                "Thickness ratio value outside of the range in Roskam's book, value clipped"
            )

        cl_delta_th = np.interp(np.clip(thickness, 0.0, 0.15), [0.0, 0.04, 0.1, 0.15], cld_t)

        return cl_delta_th

    @staticmethod
    @functools.lru_cache(maxsize=128)
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
        db = pd.read_csv(file)

        # Figure 10.64 b
        cl_alpha_th = 6.3 + np.clip(thickness_ratio, 0.0, 0.2) / 0.2 * (7.3 - 6.3)

        k_cl_alpha_data = filter_nans(db, ["K_CL_ALPHA"])[0]
        k_cl_delta_min_data = filter_nans(db, ["K_CL_DELTA_MIN"])[0]
        k_cl_delta_max_data = filter_nans(db, ["K_CL_DELTA_MAX"])[0]

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

        k_cl_delta_min = np.interp(k_cl_alpha, k_cl_alpha_data, k_cl_delta_min_data)
        k_cl_delta_max = np.interp(k_cl_alpha, k_cl_alpha_data, k_cl_delta_max_data)

        if chord_ratio != np.clip(chord_ratio, 0.05, 0.5):
            _LOGGER.warning(
                "Chord ratio value outside of the range in Roskam's book, value clipped"
            )

        chord_ratio = np.clip(chord_ratio, 0.05, 0.5)
        k_cl_delta = np.interp(chord_ratio, [0.05, 0.5], [k_cl_delta_min, k_cl_delta_max])

        return k_cl_delta

    @staticmethod
    @functools.lru_cache(maxsize=128)
    def k_prime_single_slotted(flap_angle, chord_ratio):
        """
        Roskam data to estimate the lift effectiveness of a single slotted flap (figure 8.17),
        noted here k_prime to match the notation of the plain flap but is written alpha_delta in
        the book.

        :param flap_angle: the control surface deflection angle (in °).
        :param chord_ratio: control surface chord over lifting surface chord ratio.
        :return k_prime: lift effectiveness factor of a single slotted flap.
        """

        file = pth.join(resources.__path__[0], K_SINGLE_SLOT)
        db = pd.read_csv(file)

        x_15, y_15 = filter_nans(db, ["X_15", "Y_15"])
        x_20, y_20 = filter_nans(db, ["X_20", "Y_20"])
        x_25, y_25 = filter_nans(db, ["X_25", "Y_25"])
        x_30, y_30 = filter_nans(db, ["X_30", "Y_30"])
        x_40, y_40 = filter_nans(db, ["X_40", "Y_40"])

        if (
            (float(flap_angle) != np.clip(float(flap_angle), min(x_15), max(x_15)))
            or (float(flap_angle) != np.clip(float(flap_angle), min(x_20), max(x_20)))
            or (float(flap_angle) != np.clip(float(flap_angle), min(x_25), max(x_25)))
            or (float(flap_angle) != np.clip(float(flap_angle), min(x_30), max(x_30)))
            or (float(flap_angle) != np.clip(float(flap_angle), min(x_40), max(x_40)))
        ):
            _LOGGER.warning("Flap angle value outside of the range in Roskam's book, value clipped")

        k_chord = [
            np.interp(np.clip(float(flap_angle), min(x_15), max(x_15)), x_15, y_15),
            np.interp(np.clip(float(flap_angle), min(x_20), max(x_15)), x_20, y_20),
            np.interp(np.clip(float(flap_angle), min(x_25), max(x_25)), x_25, y_25),
            np.interp(np.clip(float(flap_angle), min(x_30), max(x_30)), x_30, y_30),
            np.interp(np.clip(float(flap_angle), min(x_40), max(x_40)), x_40, y_40),
        ]

        if float(chord_ratio) != np.clip(float(chord_ratio), 0.15, 0.4):
            _LOGGER.warning(
                "Chord ratio value outside of the range in Roskam's book, value clipped"
            )

        k_prime = float(
            np.interp(np.clip(float(chord_ratio), 0.15, 0.4), [0.15, 0.20, 0.25, 0.3, 0.4], k_chord)
        )

        return k_prime

    @staticmethod
    @functools.lru_cache(maxsize=128)
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
        db = pd.read_csv(file)

        x_plain, y_plain = filter_nans(db, ["X_PLAIN_FLAP", "Y_PLAIN_FLAP"])
        x_single_slot, y_single_slot = filter_nans(db, ["X_SINGLE_SLOT", "Y_SINGLE_SLOT"])

        if flap_type == 0.0:
            if thickness_ratio != np.clip(thickness_ratio, min(x_plain), max(x_plain)):
                _LOGGER.warning(
                    "Thickness ratio value outside of the range in Roskam's book, value clipped"
                )
            delta_cl_max_base = float(
                np.interp(np.clip(thickness_ratio, min(x_plain), max(x_plain)), x_plain, y_plain)
            )
        elif flap_type == 1.0:
            if thickness_ratio != np.clip(thickness_ratio, min(x_single_slot), max(x_single_slot)):
                _LOGGER.warning(
                    "Thickness ratio value outside of the range in Roskam's book, value clipped"
                )
            delta_cl_max_base = float(
                np.interp(
                    np.clip(thickness_ratio, min(x_single_slot), max(x_single_slot)),
                    x_single_slot,
                    y_single_slot,
                )
            )
        else:
            _LOGGER.warning("Flap type not recognized, used plain flap instead")
            if thickness_ratio != np.clip(thickness_ratio, min(x_plain), max(x_plain)):
                _LOGGER.warning(
                    "Thickness ratio value outside of the range in Roskam's book, value clipped"
                )
            delta_cl_max_base = float(
                np.interp(np.clip(thickness_ratio, min(x_plain), max(x_plain)), x_plain, y_plain)
            )

        return delta_cl_max_base

    @staticmethod
    @functools.lru_cache(maxsize=128)
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
        db = pd.read_csv(file)

        if flap_type == 1.0 or flap_type == 0.0:
            x, y = filter_nans(db, ["X_PLAIN_SINGLE_SPLIT", "Y_PLAIN_SINGLE_SPLIT"])
        else:
            _LOGGER.warning("Flap type not recognized, used plain flap instead")
            x, y = filter_nans(db, ["X_PLAIN_SINGLE_SPLIT", "Y_PLAIN_SINGLE_SPLIT"])

        if float(chord_ratio) != np.clip(float(chord_ratio), min(x), max(x)):
            _LOGGER.warning(
                "Chord ratio value outside of the range in Roskam's book, value clipped"
            )

        k1 = float(np.interp(np.clip(float(chord_ratio), min(x), max(x)), x, y))

        return k1

    @staticmethod
    @functools.lru_cache(maxsize=128)
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
        db = pd.read_csv(file)

        x_plain, y_plain = filter_nans(db, ["X_PLAIN_FLAP", "Y_PLAIN_FLAP"])
        x_single_slot, y_single_slot = filter_nans(db, ["X_SINGLE_SLOT", "Y_SINGLE_SLOT"])

        if flap_type == 0.0:
            if angle != np.clip(angle, min(x_plain), max(x_plain)):
                _LOGGER.warning(
                    "Control surface deflection value outside of the range in Roskam's book, "
                    "value clipped"
                )
            k2 = float(np.interp(np.clip(angle, min(x_plain), max(x_plain)), x_plain, y_plain))
        elif flap_type == 1.0:
            if angle != np.clip(angle, min(x_single_slot), max(x_single_slot)):
                _LOGGER.warning(
                    "Control surface deflection value outside of the range in Roskam's book, "
                    "value clipped"
                )
            k2 = float(
                np.interp(
                    np.clip(angle, min(x_single_slot), max(x_single_slot)),
                    x_single_slot,
                    y_single_slot,
                )
            )
        else:
            _LOGGER.warning("Flap type not recognized, used plain flap instead")
            if angle != np.clip(angle, min(x_plain), max(x_plain)):
                _LOGGER.warning(
                    "Control surface deflection value outside of the range in Roskam's book, "
                    "value clipped"
                )
            k2 = float(np.interp(np.clip(angle, min(x_plain), max(x_plain)), x_plain, y_plain))

        return k2

    @staticmethod
    @functools.lru_cache(maxsize=128)
    def k3_max_lift(angle, flap_type) -> float:
        """
        Roskam data for flap motion correction factor (figure 8.34).

        :param angle: control surface deflection angle, in °.
        :param flap_type: type of flap used as described in Roskam, for now can be 0.0 for plain,
        1.0 for single_slot.
        :return k3: correction factor to account flap motion correction.
        """

        file = pth.join(resources.__path__[0], K3)
        db = pd.read_csv(file)

        if flap_type == 0.0:
            k3 = 1.0
        elif flap_type == 1.0:
            x, y = filter_nans(db, ["X_SINGLE_SLOT", "Y_SINGLE_SLOT"])
            reference_angle = 45.0
            if float(angle / reference_angle) != np.clip(
                float(angle / reference_angle), min(x), max(x)
            ):
                _LOGGER.warning(
                    "Control surface deflection value outside of the range in Roskam's book, "
                    "value clipped, reference value is %f",
                    reference_angle,
                )
            k3 = float(np.interp(np.clip(float(angle / reference_angle), min(x), max(x)), x, y))
        else:
            _LOGGER.warning("Flap type not recognized, used plain flap instead")
            k3 = 1.0

        return k3

    @staticmethod
    @functools.lru_cache(maxsize=128)
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
        db = pd.read_csv(file)

        x_0, y_0 = filter_nans(db, ["X_0", "Y_0"])
        x_05, y_05 = filter_nans(db, ["X_0.5", "Y_0.5"])
        x_1, y_1 = filter_nans(db, ["X_1", "Y_1"])

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
            float(np.interp(np.clip(eta_in, min(x_0), max(x_0)), x_0, y_0)),
            float(np.interp(np.clip(eta_in, min(x_05), max(x_05)), x_05, y_05)),
            float(np.interp(np.clip(eta_in, min(x_1), max(x_1)), x_1, y_1)),
        ]
        kb_in = np.interp(taper_ratio, [0.0, 0.5, 1.0], k_eta)

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
            float(np.interp(np.clip(eta_out, min(x_0), max(x_0)), x_0, y_0)),
            float(np.interp(np.clip(eta_out, min(x_05), max(x_05)), x_05, y_05)),
            float(np.interp(np.clip(eta_out, min(x_1), max(x_1)), x_1, y_1)),
        ]
        kb_out = np.interp(taper_ratio, [0.0, 0.5, 1.0], k_eta)

        return float(kb_out - kb_in)

    @staticmethod
    @functools.lru_cache(maxsize=128)
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
        db = pd.read_csv(file)

        a_delta = interpolate_database(db, "X", "Y", chord_ratio)

        if chord_ratio != np.clip(chord_ratio, 0.0, 1.0):
            _LOGGER.warning(
                "Chord ratio value outside of the range in Roskam's book, value clipped"
            )

        return a_delta

    @staticmethod
    @functools.lru_cache(maxsize=128)
    def k_a_delta(a_delta_airfoil, aspect_ratio) -> float:
        """
        Roskam data to estimate the two-dimensional to three-dimensional control surface lift
        effectiveness parameter (figure 8.53b).

        :param a_delta_airfoil: control surface two-dimensional flap effectiveness factor.
        :param aspect_ratio: aspect ratio of the fixed surface.
        :return k_a_delta: two-dimensional to three-dimensional control surface lift effectiveness
        parameter.
        """

        file = pth.join(resources.__path__[0], K_A_DELTA)
        db = pd.read_csv(file)

        if float(aspect_ratio) != np.clip(float(aspect_ratio), 0.0, 10.0):
            _LOGGER.warning(
                "Aspect ratio value outside of the range in Roskam's book, value clipped"
            )

        y1 = interpolate_database(db, "X_01", "Y_01", aspect_ratio)
        y2 = interpolate_database(db, "X_02", "Y_02", aspect_ratio)
        y3 = interpolate_database(db, "X_03", "Y_03", aspect_ratio)
        y4 = interpolate_database(db, "X_04", "Y_04", aspect_ratio)
        y5 = interpolate_database(db, "X_05", "Y_05", aspect_ratio)
        y6 = interpolate_database(db, "X_06", "Y_06", aspect_ratio)
        y7 = interpolate_database(db, "X_07", "Y_07", aspect_ratio)
        y8 = interpolate_database(db, "X_08", "Y_08", aspect_ratio)
        y9 = interpolate_database(db, "X_09", "Y_09", aspect_ratio)
        y10 = interpolate_database(db, "X_10", "Y_10", aspect_ratio)

        x = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        y = [y1, y2, y3, y4, y5, y6, y7, y8, y9, y10]

        if a_delta_airfoil != np.clip(a_delta_airfoil, 0.0, 1.0):
            _LOGGER.warning(
                "Control surface effectiveness ratio value outside of the range in "
                "Roskam's book, value clipped"
            )

        k_a_delta = float(np.interp(np.clip(a_delta_airfoil, 0.1, 1.0), x, y))

        return k_a_delta

    @staticmethod
    @functools.lru_cache(maxsize=128)
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

        x_cp_c_prime = float(
            np.interp(np.clip(flap_chord_ratio, 0.0, 1.0), [0.0, 1.0], [0.5, 0.25])
        )

        return x_cp_c_prime

    @staticmethod
    @functools.lru_cache(maxsize=128)
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
        db = pd.read_csv(file)

        eta_in_1_0 = interpolate_database(db, "taper_1_0_X", "taper_1_0_Y", eta_in)
        eta_out_1_0 = interpolate_database(db, "taper_1_0_X", "taper_1_0_Y", eta_out)

        eta_in_0_5 = interpolate_database(db, "taper_0_5_X", "taper_0_5_Y", eta_in)
        eta_out_0_5 = interpolate_database(db, "taper_0_5_X", "taper_0_5_Y", eta_out)

        eta_in_0_333 = interpolate_database(db, "taper_0_333_X", "taper_0_333_Y", eta_in)
        eta_out_0_333 = interpolate_database(db, "taper_0_333_X", "taper_0_333_Y", eta_out)

        eta_in_0_25 = interpolate_database(db, "taper_0_25_X", "taper_0_25_Y", eta_in)
        eta_out_0_25 = interpolate_database(db, "taper_0_25_X", "taper_0_25_Y", eta_out)

        taper_array = [0.25, 0.333, 0.5, 1.0]
        eta_in_array = [eta_in_0_25, eta_in_0_333, eta_in_0_5, eta_in_1_0]
        eta_out_array = [eta_out_0_25, eta_out_0_333, eta_out_0_5, eta_out_1_0]

        if taper_ratio != np.clip(taper_ratio, 0.25, 1.0):
            _LOGGER.warning("Taper ratio outside of the range in Roskam's book, value clipped")

        k_p = float(
            np.interp(np.clip(taper_ratio, 0.25, 1.0), taper_array, eta_out_array)
            - np.interp(np.clip(taper_ratio, 0.25, 1.0), taper_array, eta_in_array)
        )

        return k_p

    @staticmethod
    @functools.lru_cache(maxsize=128)
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
        db = pd.read_csv(file)

        k_21 = interpolate_database(db, "TOC_21_X", "TOC_21_Y", chord_ratio)
        k_18 = interpolate_database(db, "TOC_18_X", "TOC_18_Y", chord_ratio)
        k_15 = interpolate_database(db, "TOC_15_X", "TOC_15_Y", chord_ratio)
        k_12 = interpolate_database(db, "TOC_12_X", "TOC_12_Y", chord_ratio)
        k_09 = interpolate_database(db, "TOC_09_X", "TOC_09_Y", chord_ratio)
        k_06 = interpolate_database(db, "TOC_06_X", "TOC_06_Y", chord_ratio)
        k_03 = interpolate_database(db, "TOC_03_X", "TOC_03_Y", chord_ratio)

        toc_array = [0.03, 0.06, 0.09, 0.12, 0.15, 0.18, 0.21]
        k_array = [k_03, k_06, k_09, k_12, k_15, k_18, k_21]

        if thickness_ratio != np.clip(thickness_ratio, 0.03, 0.4):
            _LOGGER.warning(
                "Thickness to chord ratio outside of the range in Roskam's book, " "value clipped"
            )

        k = float(np.interp(np.clip(thickness_ratio, 0.03, 0.4), toc_array, k_array))

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
        db = pd.read_csv(file)

        eta_in_1_0 = interpolate_database(db, "X_1_0", "Y_1_0", eta_in)
        eta_in_0_5 = interpolate_database(db, "X_0_5", "Y_0_5", eta_in)
        eta_in_0_333 = interpolate_database(db, "X_0_333", "Y_0_333", eta_in)
        eta_in_0_2 = interpolate_database(db, "X_0_2", "Y_0_2", eta_in)

        eta_out_1_0 = interpolate_database(db, "X_1_0", "Y_1_0", eta_out)
        eta_out_0_5 = interpolate_database(db, "X_0_5", "Y_0_5", eta_out)
        eta_out_0_333 = interpolate_database(db, "X_0_333", "Y_0_333", eta_out)
        eta_out_0_2 = interpolate_database(db, "X_0_2", "Y_0_2", eta_out)

        taper_array = [0.2, 0.333, 0.5, 1.0]
        eta_in_array = [eta_in_0_2, eta_in_0_333, eta_in_0_5, eta_in_1_0]
        eta_out_array = [eta_out_0_2, eta_out_0_333, eta_out_0_5, eta_out_1_0]

        if taper_ratio != np.clip(taper_ratio, 0.2, 1.0):
            _LOGGER.warning("Taper ratio outside of the range in Roskam's book, value clipped")

        k_delta_in = np.interp(np.clip(taper_ratio, 0.2, 1.0), taper_array, eta_in_array)
        k_delta_out = np.interp(np.clip(taper_ratio, 0.2, 1.0), taper_array, eta_out_array)

        k_delta = float(k_delta_out - k_delta_in)

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
        db = pd.read_csv(file)

        x_06, y_06 = filter_nans(db, ["X_06", "Y_06"])
        x_10, y_10 = filter_nans(db, ["X_10", "Y_10"])

        x_value = span / avg_fuselage_depth

        if x_value != np.clip(x_value, min(min(x_06), min(x_10)), max(max(x_06), max(x_10))):
            _LOGGER.warning(
                "Ratio of span on fuselage depth outside of the range in Roskam's book, "
                "value clipped"
            )

        y_value_06 = np.interp(np.clip(x_value, min(x_06), max(x_06)), x_06, y_06)
        y_value_10 = np.interp(np.clip(x_value, min(x_10), max(x_10)), x_10, y_10)

        if taper_ratio != np.clip(taper_ratio, 0.6, 1.0):
            _LOGGER.warning("Taper ratio outside of the range in Roskam's book, value clipped")

        k_ar_fuselage = float(
            np.interp(
                np.clip(taper_ratio, 0.6, 1.0), [0.6, 1.0], [float(y_value_06), float(y_value_10)]
            )
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
        db = pd.read_csv(file)

        x, y = filter_nans(db, ["X", "Y"])

        if float(area_ratio) != np.clip(float(area_ratio), min(x), max(x)):
            _LOGGER.warning("Area ratio value outside of the range in Roskam's book, value clipped")

        k_vh = float(np.interp(np.clip(float(area_ratio), min(x), max(x)), x, y))

        return k_vh

    @staticmethod
    @functools.lru_cache(maxsize=128)
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
        db = pd.read_csv(file)

        # Figure 10.64 b
        if thickness_ratio != np.clip(thickness_ratio, 0.0, 0.2):
            _LOGGER.warning(
                "Thickness ratio value outside of the range in Roskam's book, value clipped"
            )
        cl_alpha_th = 6.3 + np.clip(thickness_ratio, 0.0, 0.2) / 0.2 * (7.3 - 6.3)

        k_cl_alpha_data = filter_nans(db, ["K_CL_ALPHA"])[0]
        k_ch_alpha_min_data = filter_nans(db, ["K_CH_ALPHA_MIN"])[0]
        k_ch_alpha_max_data = filter_nans(db, ["K_CH_ALPHA_MAX"])[0]

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

        k_ch_alpha_min = np.interp(k_cl_alpha, k_cl_alpha_data, k_ch_alpha_min_data)
        k_ch_alpha_max = np.interp(k_cl_alpha, k_cl_alpha_data, k_ch_alpha_max_data)

        chord_ratio = np.clip(chord_ratio, 0.1, 0.4)
        k_ch_alpha = float(np.interp(chord_ratio, [0.1, 0.4], [k_ch_alpha_min, k_ch_alpha_max]))

        return k_ch_alpha

    @staticmethod
    @functools.lru_cache(maxsize=128)
    def ch_alpha_th(thickness_ratio, chord_ratio):
        """
        Roskam data to compute the theoretical 2D control surface hinge moment derivative due to
        AOA (figure 10.63).

        :param thickness_ratio: airfoil thickness ratio.
        :param chord_ratio: flap chord over wing chord ratio.
        :return ch_alpha: theoretical hinge moment derivative due to AOA.
        """

        file = pth.join(resources.__path__[0], CH_ALPHA_TH)
        db = pd.read_csv(file)

        thickness_ratio_data = filter_nans(db, ["THICKNESS_RATIO"])[0]
        ch_alpha_min_data = filter_nans(db, ["CH_ALPHA_MIN"])[0]
        ch_alpha_max_data = filter_nans(db, ["CH_ALPHA_MAX"])[0]

        if float(thickness_ratio) != np.clip(
            float(thickness_ratio), min(thickness_ratio_data), max(thickness_ratio_data)
        ):
            _LOGGER.warning(
                "Thickness ratio value outside of the range in Roskam's book, value clipped"
            )

        ch_alpha_min = np.interp(
            np.clip(float(thickness_ratio), min(thickness_ratio_data), max(thickness_ratio_data)),
            thickness_ratio_data,
            ch_alpha_min_data,
        )
        ch_alpha_max = np.interp(
            np.clip(float(thickness_ratio), min(thickness_ratio_data), max(thickness_ratio_data)),
            thickness_ratio_data,
            ch_alpha_max_data,
        )
        if chord_ratio != np.clip(chord_ratio, 0.1, 0.4):
            _LOGGER.warning(
                "Chord ratio value outside of the range in Roskam's book, value clipped"
            )

        chord_ratio = np.clip(chord_ratio, 0.1, 0.4)
        ch_alpha_th = float(np.interp(chord_ratio, [0.1, 0.4], [ch_alpha_min, ch_alpha_max]))

        return ch_alpha_th

    @staticmethod
    @functools.lru_cache(maxsize=128)
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
        db = pd.read_csv(file)

        # Figure 10.64 b
        if thickness_ratio != np.clip(thickness_ratio, 0.0, 0.2):
            _LOGGER.warning(
                "Thickness ratio value outside of the range in Roskam's book, value clipped"
            )

        cl_alpha_th = 6.3 + np.clip(thickness_ratio, 0.0, 0.2) / 0.2 * (7.3 - 6.3)

        k_cl_alpha_data = filter_nans(db, ["K_CL_ALPHA"])[0]
        k_ch_delta_min_data = filter_nans(db, ["K_CH_DELTA_MIN"])[0]
        k_ch_delta_avg_data = filter_nans(db, ["K_CH_DELTA_AVG"])[0]
        k_ch_delta_max_data = filter_nans(db, ["K_CH_DELTA_MAX"])[0]

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

        k_ch_delta_min = float(np.interp(k_cl_alpha, k_cl_alpha_data, k_ch_delta_min_data))
        k_ch_delta_avg = float(np.interp(k_cl_alpha, k_cl_alpha_data, k_ch_delta_avg_data))
        k_ch_delta_max = float(np.interp(k_cl_alpha, k_cl_alpha_data, k_ch_delta_max_data))

        if chord_ratio != np.clip(chord_ratio, 0.1, 0.4):
            _LOGGER.warning(
                "Chord ratio value outside of the range in Roskam's book, value clipped"
            )

        chord_ratio = np.clip(chord_ratio, 0.1, 0.4)
        k_ch_delta = float(
            np.interp(
                chord_ratio, [0.1, 0.25, 0.4], [k_ch_delta_min, k_ch_delta_avg, k_ch_delta_max]
            )
        )

        return k_ch_delta

    @staticmethod
    @functools.lru_cache(maxsize=128)
    def ch_delta_th(thickness_ratio, chord_ratio):
        """
        Roskam data to compute the theoretical 2D control surface hinge moment derivative due to
        control surface deflection (figure 10.69 b).

        :param thickness_ratio: airfoil thickness ratio.
        :param chord_ratio: flap chord over wing chord ratio.
        :return ch_delta: theoretical hinge moment derivative due to control surface deflection.
        """

        file = pth.join(resources.__path__[0], CH_DELTA_TH)
        db = pd.read_csv(file)

        thickness_ratio_data = filter_nans(db, ["THICKNESS_RATIO"])[0]

        if float(thickness_ratio) != np.clip(
            float(thickness_ratio), min(thickness_ratio_data), max(thickness_ratio_data)
        ):
            _LOGGER.warning(
                "Thickness ratio value outside of the range in Roskam's book, value clipped"
            )

        ch_delta_min = interpolate_database(db, "THICKNESS_RATIO", "CH_DELTA_MIN", thickness_ratio)
        ch_delta_max = interpolate_database(db, "THICKNESS_RATIO", "CH_DELTA_MAX", thickness_ratio)

        if chord_ratio != np.clip(chord_ratio, 0.1, 0.4):
            _LOGGER.warning(
                "Chord ratio value outside of the range in Roskam's book, value clipped"
            )

        chord_ratio = np.clip(chord_ratio, 0.1, 0.4)
        ch_delta_th = float(np.interp(chord_ratio, [0.1, 0.4], [ch_delta_min, ch_delta_max]))

        return ch_delta_th

    @staticmethod
    def k_fus(root_quarter_chord_position_ratio) -> float:
        """
        Roskam data to estimate the empirical pitching moment factor K_fus (figure 16.14).

        :param root_quarter_chord_position_ratio: the position of the root quarter chord of the
        wing from the nose.
        divided by the total length of the fuselage.
        :return k_fus: the empirical pitching moment factor.
        """

        file = pth.join(resources.__path__[0], K_FUS)
        db = pd.read_csv(file)

        x, y = filter_nans(db, ["X_0_25_RATIO", "K_FUS"])

        if float(root_quarter_chord_position_ratio) != np.clip(
            float(root_quarter_chord_position_ratio), min(x), max(x)
        ):
            _LOGGER.warning(
                "Position of the root quarter-chord as percent of fuselage length is outside of "
                "the range in Roskam's book, value clipped"
            )

        k_fus = float(np.interp(np.clip(root_quarter_chord_position_ratio, min(x), max(x)), x, y))

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
        db = pd.read_csv(file)

        taper_ratio_data, aspect_ratio_data, sweep_50_data, sweep_contribution = filter_nans(
            db, ["TAPER_RATIO", "ASPECT_RATIO", "SWEEP_50", "SWEEP_CONTRIBUTION"]
        )

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

        # Linear interpolation is preferred, but we put the nearest one as protection
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
        db = pd.read_csv(file)

        swept_aspect_ratio_data, swept_mach_data, k_m_lambda_data = filter_nans(
            db, ["AR_SWEPT", "M_SWEPT", "SWEEP_COMPRESSIBILITY_CORRECTION"]
        )

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
        Roskam data to estimate the fuselage correction factor. (figure 10.22)

        :param swept_aspect_ratio: the aspect ratio of the lifting surface divided by cos(sweep_50)
        :param lf_to_b_ratio: ratio between the distance from nose to root half chord and the
        wing span
        :return k_fuselage: fuselage correction factor.
        """

        file = pth.join(resources.__path__[0], K_FUSELAGE)
        db = pd.read_csv(file)

        swept_aspect_ratio_data, lf_to_b_data, k_fuselage_data = filter_nans(
            db, ["AR_SWEPT", "LF_TO_B_RATIO", "K_FUSELAGE"]
        )

        if float(swept_aspect_ratio) != np.clip(
            float(swept_aspect_ratio), min(swept_aspect_ratio_data), max(swept_aspect_ratio_data)
        ):
            _LOGGER.warning(
                "Swept aspect ratio is outside of the range in Roskam's book, value clipped"
            )
        if float(lf_to_b_ratio) != np.clip(
            float(lf_to_b_ratio), min(lf_to_b_data), max(lf_to_b_data)
        ):
            _LOGGER.warning(
                "Ratio between the distance from nose to root half chord and the wing span is "
                "outside of the range in Roskam's book, value clipped"
            )

        k_fuselage = interpolate.griddata(
            (swept_aspect_ratio_data, lf_to_b_data),
            k_fuselage_data,
            np.array([swept_aspect_ratio, lf_to_b_ratio]).T,
            method="linear",
        )
        if np.isnan(k_fuselage):
            k_fuselage = interpolate.griddata(
                (swept_aspect_ratio_data, lf_to_b_data),
                k_fuselage_data,
                np.array([swept_aspect_ratio, lf_to_b_ratio]).T,
                method="nearest",
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
        db = pd.read_csv(file)

        taper_ratio_data, aspect_ratio_data, ar_contribution = filter_nans(
            db, ["TAPER_RATIO", "ASPECT_RATIO", "ASPECT_RATIO_CONTRIBUTION"]
        )

        if float(taper_ratio) != np.clip(
            float(taper_ratio), min(taper_ratio_data), max(taper_ratio_data)
        ):
            _LOGGER.warning("Taper ratio is outside of the range in Roskam's book, value clipped")
        if float(aspect_ratio) != np.clip(
            float(aspect_ratio), min(aspect_ratio_data), max(aspect_ratio_data)
        ):
            _LOGGER.warning("Aspect ratio is outside of the range in Roskam's book, value clipped")

        # Linear interpolation is preferred, but we put the nearest one as protection
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
        db = pd.read_csv(file)

        taper_ratio_data, aspect_ratio_data, sweep_50_data, dihedral_contribution = filter_nans(
            db, ["TAPER_RATIO", "ASPECT_RATIO", "SWEEP_50", "DIHEDRAL_CONTRIBUTION"]
        )

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

        # Linear interpolation is preferred, but we put the nearest one as protection
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
        Roskam data to estimate the compressibility correction for the dihedral angle. (figure
        10.25)

        :param swept_aspect_ratio: the aspect ratio of the lifting surface divided by cos(sweep_50)
        :param swept_mach: mach number multiplied by cos(sweep_50)
        :return k_m_gamma: compressibility correction for the dihedral angle.
        """

        file = pth.join(resources.__path__[0], K_M_GAMMA)
        db = pd.read_csv(file)

        swept_aspect_ratio_data, swept_mach_data, k_m_gamma_data = filter_nans(
            db, ["AR_SWEPT", "M_SWEPT", "DIHEDRAL_COMPRESSIBILITY_CORRECTION"]
        )

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

        k_m_gamma = interpolate.griddata(
            (swept_aspect_ratio_data, swept_mach_data),
            k_m_gamma_data,
            np.array([swept_aspect_ratio, swept_mach]).T,
            method="linear",
        )
        if np.isnan(k_m_gamma):
            k_m_gamma = interpolate.griddata(
                (swept_aspect_ratio_data, swept_mach_data),
                k_m_gamma_data,
                np.array([swept_aspect_ratio, swept_mach]).T,
                method="nearest",
            )

        return float(k_m_gamma)

    @staticmethod
    def cl_beta_twist_correction(taper_ratio, aspect_ratio) -> float:
        """
        Roskam data to estimate the correction due to the twist of the lifting surface. (figure
        10.26)

        :param taper_ratio: the taper ratio of the lifting surface
        :param aspect_ratio: the aspect ratio of the lifting surface
        :return k_epsilon: the factor to take into account the twist of the lifting surface for
        the computation of the rolling moment
        """

        file = pth.join(resources.__path__[0], K_TWIST)
        db = pd.read_csv(file)

        taper_ratio_data, aspect_ratio_data, twist_correction = filter_nans(
            db, ["TAPER_RATIO", "ASPECT_RATIO", "TWIST_CORRECTION"]
        )

        if float(taper_ratio) != np.clip(
            float(taper_ratio), min(taper_ratio_data), max(taper_ratio_data)
        ):
            _LOGGER.warning("Taper ratio is outside of the range in Roskam's book, value clipped")
        if float(aspect_ratio) != np.clip(
            float(aspect_ratio), min(aspect_ratio_data), max(aspect_ratio_data)
        ):
            _LOGGER.warning("Aspect ratio is outside of the range in Roskam's book, value clipped")

        # Linear interpolation is preferred, but we put the nearest one as protection
        k_epsilon = interpolate.griddata(
            (taper_ratio_data, aspect_ratio_data),
            twist_correction,
            np.array([taper_ratio, aspect_ratio]).T,
            method="linear",
        )
        if np.isnan(k_epsilon):
            k_epsilon = interpolate.griddata(
                (taper_ratio_data, aspect_ratio_data),
                twist_correction,
                np.array([taper_ratio, aspect_ratio]).T,
                method="nearest",
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

        beta = np.sqrt(1.0 - mach**2.0)

        corrected_ar = aspect_ratio * beta / k
        corrected_sweep = np.arctan(np.tan(sweep_25) / beta)
        file = pth.join(resources.__path__[0], K_ROLL_DAMPING)
        db = pd.read_csv(file)

        taper_ratio_data, correct_ar_data, corrected_sweep_data, roll_damping_data = filter_nans(
            db, ["TAPER_RATIO", "CORRECTED_AR", "CORRECTED_SWEEP", "ROLL_DAMPING_PARAMETER"]
        )

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

        # Linear interpolation is preferred, but we put the nearest one as protection
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
        Roskam data to estimate the contribution to the roll moment damping of the
        drag-due-to-lift (figure 10.36)

        :param sweep_25: the sweep angle at 25% of the chord of the lifting surface
        :param aspect_ratio: the aspect ratio of the lifting surface
        :return k_cdi_roll_damping: the contribution to the roll moment of the aspect ratio of the
        lifting surface.
        """

        file = pth.join(resources.__path__[0], K_CDI_ROLL_DAMPING)
        db = pd.read_csv(file)

        sweep_25_data, aspect_ratio_data, cdi_roll_damping_data = filter_nans(
            db, ["SWEEP_25", "ASPECT_RATIO", "CDI_ROLL_DAMPING_PARAMETER"]
        )

        if float(sweep_25) != np.clip(float(sweep_25), min(sweep_25_data), max(sweep_25_data)):
            _LOGGER.warning(
                "Sweep at 25% of the chord is outside of the range in Roskam's book, value clipped"
            )
        if float(aspect_ratio) != np.clip(
            float(aspect_ratio), min(aspect_ratio_data), max(aspect_ratio_data)
        ):
            _LOGGER.warning("Aspect ratio is outside of the range in Roskam's book, value clipped")

        # Linear interpolation is preferred, but we put the nearest one as protection
        k_cdi_roll_damping = interpolate.griddata(
            (sweep_25_data, aspect_ratio_data),
            cdi_roll_damping_data,
            np.array([sweep_25, aspect_ratio]).T,
            method="linear",
        )
        if np.isnan(k_cdi_roll_damping):
            k_cdi_roll_damping = interpolate.griddata(
                (sweep_25_data, aspect_ratio_data),
                cdi_roll_damping_data,
                np.array([sweep_25, aspect_ratio]).T,
                method="nearest",
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
        db = pd.read_csv(file)

        x_0, y_0 = filter_nans(db, ["TAPER_RATIO_0_X", "TAPER_RATIO_0_Y"])
        x_0.sort()
        y_0.sort()

        x_0_25, y_0_25 = filter_nans(db, ["TAPER_RATIO_025_X", "TAPER_RATIO_025_Y"])
        x_0_25.sort()
        y_0_25.sort()

        x_0_5, y_0_5 = filter_nans(db, ["TAPER_RATIO_05_X", "TAPER_RATIO_05_Y"])
        x_0_5.sort()
        y_0_5.sort()

        x_1_0, y_1_0 = filter_nans(db, ["TAPER_RATIO_1_X", "TAPER_RATIO_1_Y"])
        x_1_0.sort()
        y_1_0.sort()

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
            float(np.interp(np.clip(aspect_ratio, min(x_0), max(x_0)), x_0, y_0)),
            float(np.interp(np.clip(aspect_ratio, min(x_0_25), max(x_0_25)), x_0_25, y_0_25)),
            float(np.interp(np.clip(aspect_ratio, min(x_0_5), max(x_0_5)), x_0_5, y_0_5)),
            float(np.interp(np.clip(aspect_ratio, min(x_1_0), max(x_1_0)), x_1_0, y_1_0)),
        ]

        if taper_ratio != np.clip(taper_ratio, 0.0, 1.0):
            _LOGGER.warning(
                "Taper ratio value outside of the range in Roskam's book, value clipped"
            )

        k_intermediate = float(
            np.interp(np.clip(taper_ratio, 0.0, 1.0), [0.0, 0.25, 0.5, 1.0], k_taper)
        )

        # Reading the second part of the figure (b) relative to the different wing sweep angles.
        file = pth.join(resources.__path__[0], CL_R_LIFT_PART_B)
        db = pd.read_csv(file)

        x_sw_0, y_sw_0 = filter_nans(db, ["SWEEP_25_0_X", "SWEEP_25_0_Y"])
        x_sw_0.sort()
        y_sw_0.sort()

        x_sw_15, y_sw_15 = filter_nans(db, ["SWEEP_25_15_X", "SWEEP_25_15_Y"])
        x_sw_15.sort()
        y_sw_15.sort()

        x_sw_30, y_sw_30 = filter_nans(db, ["SWEEP_25_30_X", "SWEEP_25_30_Y"])
        x_sw_30.sort()
        y_sw_30.sort()

        x_sw_45, y_sw_45 = filter_nans(db, ["SWEEP_25_45_X", "SWEEP_25_45_Y"])
        x_sw_45.sort()
        y_sw_45.sort()

        x_sw_60, y_sw_60 = filter_nans(db, ["SWEEP_25_60_X", "SWEEP_25_60_Y"])
        x_sw_60.sort()
        y_sw_60.sort()

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
            float(np.interp(np.clip(k_intermediate, min(x_sw_0), max(x_sw_0)), x_sw_0, y_sw_0)),
            float(np.interp(np.clip(k_intermediate, min(x_sw_15), max(x_sw_15)), x_sw_15, y_sw_15)),
            float(np.interp(np.clip(k_intermediate, min(x_sw_30), max(x_sw_30)), x_sw_30, y_sw_30)),
            float(np.interp(np.clip(k_intermediate, min(x_sw_45), max(x_sw_45)), x_sw_45, y_sw_45)),
            float(np.interp(np.clip(k_intermediate, min(x_sw_60), max(x_sw_60)), x_sw_60, y_sw_60)),
        ]

        if sweep_25 != np.clip(sweep_25, 0.0, 60.0):
            _LOGGER.warning(
                "Sweep angle value outside of the range in Roskam's book, value clipped"
            )

        cl_r_lift = float(
            np.interp(np.clip(sweep_25, 0.0, 60.0), [0.0, 15.0, 30.0, 45.0, 60.0], k_sweep)
        )

        return cl_r_lift

    @staticmethod
    def cl_r_twist_effect(taper_ratio, aspect_ratio) -> float:
        """
        Roskam data to estimate the contribution to the roll moment coefficient of the twist.
        (figure 10.42)

        :param taper_ratio: the taper ratio of the lifting surface
        :param aspect_ratio: the aspect ratio of the lifting surface
        :return k_twist: contribution to the roll moment coefficient of the twist.
        """

        file = pth.join(resources.__path__[0], CL_R_TWIST_EFFECT)
        db = pd.read_csv(file)

        taper_ratio_data, aspect_ratio_data, twist_effect_data = filter_nans(
            db, ["TAPER_RATIO", "ASPECT_RATIO", "TWIST_EFFECT"]
        )

        if float(taper_ratio) != np.clip(
            float(taper_ratio), min(taper_ratio_data), max(taper_ratio_data)
        ):
            _LOGGER.warning("Taper ratio is outside of the range in Roskam's book, value clipped")
        if float(aspect_ratio) != np.clip(
            float(aspect_ratio), min(aspect_ratio_data), max(aspect_ratio_data)
        ):
            _LOGGER.warning("Aspect ratio is outside of the range in Roskam's book, value clipped")

        # Linear interpolation is preferred, but we put the nearest one as protection
        k_twist = interpolate.griddata(
            (taper_ratio_data, aspect_ratio_data),
            twist_effect_data,
            np.array([taper_ratio, aspect_ratio]).T,
            method="linear",
        )
        if np.isnan(k_twist):
            k_twist = interpolate.griddata(
                (taper_ratio_data, aspect_ratio_data),
                twist_effect_data,
                np.array([taper_ratio, aspect_ratio]).T,
                method="nearest",
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
        db = pd.read_csv(file)

        taper_ratio_data, aspect_ratio_data, eta_i_data, correlation_constant = filter_nans(
            db, ["TAPER_RATIO", "ASPECT_RATIO", "SPAN_RATIO", "CORRELATION_CONSTANT"]
        )

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

        # Linear interpolation is preferred, but we put the nearest one as protection
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
        Roskam data to estimate the contribution to the yaw moment of the twist of the
        lifting surface. (figure 10.37)

        :param taper_ratio: the taper ratio of the lifting surface
        :param aspect_ratio: the aspect ratio of the lifting surface
        :return cn_p_twist: the contribution to the yaw moment of the twist of the
        lifting surface.
        """

        file = pth.join(resources.__path__[0], CN_P_TWIST)
        db = pd.read_csv(file)

        taper_ratio_data, aspect_ratio_data, twist_contribution = filter_nans(
            db, ["TAPER_RATIO", "ASPECT_RATIO", "TWIST_CONTRIBUTION"]
        )

        if float(taper_ratio) != np.clip(
            float(taper_ratio), min(taper_ratio_data), max(taper_ratio_data)
        ):
            _LOGGER.warning("Taper ratio is outside of the range in Roskam's book, value clipped")
        if float(aspect_ratio) != np.clip(
            float(aspect_ratio), min(aspect_ratio_data), max(aspect_ratio_data)
        ):
            _LOGGER.warning("Aspect ratio is outside of the range in Roskam's book, value clipped")

        # Linear interpolation is preferred, but we put the nearest one as protection
        cn_p_twist = interpolate.griddata(
            (taper_ratio_data, aspect_ratio_data),
            twist_contribution,
            np.array([taper_ratio, aspect_ratio]).T,
            method="linear",
        )
        if np.isnan(cn_p_twist):
            cn_p_twist = interpolate.griddata(
                (taper_ratio_data, aspect_ratio_data),
                twist_contribution,
                np.array([taper_ratio, aspect_ratio]).T,
                method="nearest",
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
        db = pd.read_csv(file)

        static_margin_data, sweep_25_data, aspect_ratio_data, intermediate_coeff_data = filter_nans(
            db, ["STATIC_MARGIN", "SWEEP_25", "ASPECT_RATIO", "INTERMEDIATE_COEFF"]
        )

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

        # Linear interpolation is preferred, but we put the nearest one as protection
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
        db = pd.read_csv(file)

        static_margin_data, sweep_25_data, aspect_ratio_data, drag_effect_data = filter_nans(
            db, ["STATIC_MARGIN", "SWEEP_25", "ASPECT_RATIO", "DRAG_EFFECT"]
        )

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

        # Linear interpolation is preferred, but we put the nearest one as protection
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


def interpolate_database(database, tag_x: str, tag_y: str, input_x: float):
    """
    Utility to interpolate the data csv.
    """
    database_x, database_y = filter_nans(database, [tag_x, tag_y])

    output_y = float(
        np.interp(np.clip(input_x, min(database_x), max(database_x)), database_x, database_y)
    )

    return output_y


def filter_nans(database: pd.DataFrame, tags: List[str]) -> List[np.ndarray]:
    """
    Utility function to jointly filter out NaN in the database with the selected tags.
    """

    filtered_db = database[tags].dropna(axis=0, subset=tags)

    database_columns = []
    for tag in tags:
        database_columns.append(filtered_db[tag].to_numpy())

    return database_columns
