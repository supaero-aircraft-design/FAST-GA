"""Test module for aerodynamics groups."""
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

from platform import system

import numpy as np
import pytest

from .dummy_engines import ENGINE_WRAPPER_BE76 as ENGINE_WRAPPER
from .test_functions import (
    xfoil_path,
    compute_reynolds,
    cd0_high_speed,
    cd0_low_speed,
    polar,
    airfoil_slope_wt_xfoil,
    airfoil_slope_xfoil,
    comp_high_speed,
    comp_low_speed,
    hinge_moment_2d,
    hinge_moment_3d,
    hinge_moments,
    high_lift,
    extreme_cl,
    wing_extreme_cl_clean,
    htp_extreme_cl_clean,
    l_d_max,
    cnbeta,
    slipstream_openvsp_cruise,
    slipstream_openvsp_low_speed,
    compute_mach_interpolation_roskam,
    cl_alpha_vt,
    cy_delta_r,
    effective_efficiency,
    cm_alpha_fus,
    high_speed_connection,
    low_speed_connection,
    v_n_diagram,
    load_factor,
    propeller,
    non_equilibrated_cl_cd_polar,
    equilibrated_cl_cd_polar,
    elevator,
    cy_beta_fus,
    downwash_gradient,
    lift_aoa_rate_derivative,
    lift_pitch_velocity_derivative_ht,
    lift_pitch_velocity_derivative_wing,
    lift_pitch_velocity_derivative_aircraft,
    side_force_sideslip_derivative_wing,
    side_force_sideslip_derivative_vt,
    side_force_sideslip_aircraft,
    side_force_yaw_rate_aircraft,
    side_force_roll_rate_aircraft,
    roll_moment_side_slip_wing,
    roll_moment_side_slip_ht,
    roll_moment_side_slip_vt,
    roll_moment_side_slip_aircraft,
    roll_moment_roll_rate_wing,
    roll_moment_roll_rate_ht,
    roll_moment_roll_rate_vt,
    roll_moment_roll_rate_aircraft,
    roll_moment_yaw_rate_wing,
    roll_moment_yaw_rate_vt,
    roll_moment_yaw_rate_aircraft,
    roll_authority_aileron,
    roll_moment_rudder,
    pitch_moment_pitch_rate_wing,
    pitch_moment_pitch_rate_ht,
    pitch_moment_pitch_rate_aircraft,
    pitch_moment_aoa_rate_derivative,
    yaw_moment_sideslip_derivative_vt,
    yaw_moment_sideslip_aircraft,
    yaw_moment_aileron,
    yaw_moment_rudder,
    yaw_moment_roll_rate_wing,
    yaw_moment_roll_rate_vt,
    yaw_moment_roll_rate_aircraft,
    yaw_moment_yaw_rate_wing,
    yaw_moment_yaw_rate_vt,
    yaw_moment_yaw_rate_aircraft,
    polar_ext_folder,
)

XML_FILE = "beechcraft_76.xml"
SKIP_STEPS = True  # avoid some tests to accelerate validation process (intermediary VLM/OpenVSP)


def test_compute_reynolds():
    """Tests high and low speed reynolds calculation."""
    compute_reynolds(
        XML_FILE,
        mach_high_speed=0.2488,
        reynolds_high_speed=4629639,
        mach_low_speed=0.1179,
        reynolds_low_speed=2746999,
    )


def test_cd0_high_speed():
    """Tests drag coefficient @ high speed."""
    cd0_high_speed(
        XML_FILE,
        ENGINE_WRAPPER,
        cd0_wing=0.00657,
        cd0_fus=0.00490,
        cd0_ht=0.00119,
        cd0_vt=0.00066,
        cd0_nac=0.00209,
        cd0_lg=0.0,
        cd0_other=0.00205,
        cd0_total=0.02185,
    )


def test_cd0_low_speed():
    """Tests drag coefficient @ low speed."""
    cd0_low_speed(
        XML_FILE,
        ENGINE_WRAPPER,
        cd0_wing=0.00707,
        cd0_fus=0.00543,
        cd0_ht=0.00129,
        cd0_vt=0.00074,
        cd0_nac=0.00229,
        cd0_lg=0.01459,
        cd0_other=0.00205,
        cd0_total=0.04185,
    )


@pytest.mark.skipif(
    system() != "Windows" and xfoil_path is None or SKIP_STEPS,
    reason="No XFOIL executable available",
)
def test_polar():
    """Tests polar execution (XFOIL) @ high and low speed."""
    polar(
        XML_FILE,
        mach_high_speed=0.245,
        reynolds_high_speed=4571770 * 1.549,
        mach_low_speed=0.1179,
        reynolds_low_speed=2746999 * 1.549,
        cdp_1_high_speed=0.0046,
        cl_max_2d=1.6965,
        cdp_1_low_speed=0.0049,
    )


@pytest.mark.skipif(
    system() != "Windows" and xfoil_path is None or SKIP_STEPS,
    reason="No XFOIL executable available",
)
def test_polar_with_ext_folder():
    """Tests polar execution (XFOIL) @ high and low speed."""
    polar_ext_folder(
        XML_FILE,
        mach_high_speed=0.53835122,
        reynolds_high_speed=5381384,
        mach_low_speed=0.1284,
        reynolds_low_speed=2993524,
        cdp_1_high_speed=0.30059597156398105,
        cl_max_2d=1.6241,
        cdp_1_low_speed=0.005250849056603773,
    )


@pytest.mark.skipif(
    system() != "Windows" and xfoil_path is None or SKIP_STEPS,
    reason="No XFOIL executable available (or skipped)",
)
def test_airfoil_slope():
    """Tests polar execution (XFOIL) @ low speed."""
    airfoil_slope_xfoil(
        XML_FILE,
        wing_airfoil_file="naca63_415.af",
        htp_airfoil_file="naca0012.af",
        vtp_airfoil_file="naca0012.af",
        cl_alpha_wing=6.4975,
        cl_alpha_htp=6.3321,
        cl_alpha_vtp=6.3321,
    )


def test_airfoil_slope_wt_xfoil():
    """Tests polar reading @ low speed."""
    airfoil_slope_wt_xfoil(
        XML_FILE,
        wing_airfoil_file="naca63_415.af",
        htp_airfoil_file="naca0012.af",
        vtp_airfoil_file="naca0012.af",
    )


@pytest.mark.skipif(
    system() != "Windows" or SKIP_STEPS,
    reason="No XFOIL executable available: VLM basic function not computed with "
    "empty result folder (or skipped)",
)
def test_vlm_comp_high_speed():
    """Tests vlm components @ high speed."""
    comp_high_speed(
        XML_FILE,
        use_openvsp=False,
        cl0_wing=0.0894,
        cl_alpha_wing=4.820,
        cm0=-0.0247,
        coeff_k_wing=0.0522,
        cl0_htp=-0.0058,
        cl_alpha_htp=0.5068,
        cl_alpha_htp_isolated=0.8223,
        coeff_k_htp=0.4252,
        cl_alpha_vector=np.array([5.22, 5.22, 5.28, 5.36, 5.46, 5.57]),
        mach_vector=np.array([0.0, 0.15, 0.21, 0.27, 0.33, 0.39]),
    )


@pytest.mark.skipif(
    system() != "Windows" or SKIP_STEPS,
    reason="No XFOIL executable available: VLM basic function not computed with "
    "empty result folder (or skipped)",
)
def test_vlm_comp_low_speed():
    """Tests vlm components @ low speed."""
    y_vector_wing = np.array(
        [
            0.09981667,
            0.29945,
            0.49908333,
            0.84051074,
            1.32373222,
            1.8069537,
            2.29017518,
            2.77339666,
            3.25661814,
            3.73983962,
            4.11154845,
            4.37174463,
            4.63194081,
            4.89213699,
            5.15233317,
            5.41252935,
            5.67272554,
        ]
    )
    cl_vector_wing = np.array(
        [
            0.0992114,
            0.09916514,
            0.09906688,
            0.09886668,
            0.09832802,
            0.09747888,
            0.09626168,
            0.09457869,
            0.09227964,
            0.08915983,
            0.08571362,
            0.08229423,
            0.07814104,
            0.07282649,
            0.06570522,
            0.05554471,
            0.03926555,
        ]
    )
    chord_vector_wing = np.array(
        [
            1.45415954,
            1.45415954,
            1.45415954,
            1.45415954,
            1.45415954,
            1.45415954,
            1.45415954,
            1.45415954,
            1.45415954,
            1.45415954,
            1.45415954,
            1.45415954,
            1.45415954,
            1.45415954,
            1.45415954,
            1.45415954,
            1.45415954,
        ]
    )
    y_vector_htp = np.array(
        [
            0.05551452,
            0.16654356,
            0.2775726,
            0.38860163,
            0.49963067,
            0.61065971,
            0.72168875,
            0.83271779,
            0.94374682,
            1.05477586,
            1.1658049,
            1.27683394,
            1.38786298,
            1.49889201,
            1.60992105,
            1.72095009,
            1.83197913,
        ]
    )
    cl_vector_htp = np.array(
        [
            0.09855168,
            0.09833078,
            0.09788402,
            0.09720113,
            0.09626606,
            0.09505592,
            0.09353963,
            0.09167573,
            0.08940928,
            0.08666708,
            0.08335027,
            0.07932218,
            0.07438688,
            0.06824747,
            0.06041279,
            0.04994064,
            0.03442474,
        ]
    )
    comp_low_speed(
        XML_FILE,
        use_openvsp=False,
        cl0_wing=0.0872,
        cl_alpha_wing=4.701,
        cm0=-0.0241,
        coeff_k_wing=0.0500,
        cl0_htp=-0.0055,
        cl_alpha_htp=0.5019,
        cl_alpha_htp_isolated=0.8020,
        coeff_k_htp=0.4287,
        y_vector_wing=y_vector_wing,
        cl_vector_wing=cl_vector_wing,
        chord_vector_wing=chord_vector_wing,
        cl_ref_htp=0.0820,
        y_vector_htp=y_vector_htp,
        cl_vector_htp=cl_vector_htp,
    )


@pytest.mark.skipif(
    system() != "Windows" or SKIP_STEPS, reason="OPENVSP is windows dependent platform (or skipped)"
)
def test_openvsp_comp_high_speed():
    """Tests openvsp components @ high speed."""
    comp_high_speed(
        XML_FILE,
        use_openvsp=True,
        cl0_wing=0.1170,
        cl_alpha_wing=4.591,
        cm0=-0.0264,
        coeff_k_wing=0.0483,
        cl0_htp=-0.0046,
        cl_alpha_htp=0.5433,
        cl_alpha_htp_isolated=0.8438,
        coeff_k_htp=0.6684,
        cl_alpha_vector=np.array([5.06, 5.06, 5.10, 5.15, 5.22, 5.29]),
        mach_vector=np.array([0.0, 0.15, 0.21, 0.27, 0.33, 0.38]),
    )


@pytest.mark.skipif(
    system() != "Windows" or SKIP_STEPS, reason="OPENVSP is windows dependent platform (or skipped)"
)
def test_openvsp_comp_low_speed():
    """Tests openvsp components @ low speed."""
    y_vector_wing = np.array(
        [
            0.04278,
            0.12834,
            0.21389,
            0.29945,
            0.38501,
            0.47056,
            0.55612,
            0.68047,
            0.84409,
            1.00863,
            1.174,
            1.34009,
            1.50681,
            1.67405,
            1.84172,
            2.00971,
            2.17792,
            2.34624,
            2.51456,
            2.68279,
            2.85082,
            3.01853,
            3.18584,
            3.35264,
            3.51882,
            3.68429,
            3.84895,
            4.0127,
            4.17545,
            4.3371,
            4.49758,
            4.65679,
            4.81464,
            4.97107,
            5.12599,
            5.27933,
            5.43102,
            5.58099,
            5.72918,
        ]
    )
    cl_vector_wing = np.array(
        [
            0.12775493,
            0.127785,
            0.12768477,
            0.12756449,
            0.12749433,
            0.12745424,
            0.12734398,
            0.12666241,
            0.12616125,
            0.12662232,
            0.12676264,
            0.12664236,
            0.12611114,
            0.12582047,
            0.12587058,
            0.12554984,
            0.12503866,
            0.12455755,
            0.12389602,
            0.12322447,
            0.1222422,
            0.12147042,
            0.12058839,
            0.11969633,
            0.11841337,
            0.11723063,
            0.11562693,
            0.11409339,
            0.11226918,
            0.11052515,
            0.10843032,
            0.10615507,
            0.10331852,
            0.10026146,
            0.09581119,
            0.0902283,
            0.08231002,
            0.07021209,
            0.05284199,
        ]
    )
    chord_vector_wing = np.array(
        [
            1.44645,
            1.44645,
            1.44645,
            1.44645,
            1.44645,
            1.44645,
            1.44645,
            1.44645,
            1.44645,
            1.44645,
            1.44645,
            1.44645,
            1.44645,
            1.44645,
            1.44645,
            1.44645,
            1.44645,
            1.44645,
            1.44645,
            1.44645,
            1.44645,
            1.44645,
            1.44645,
            1.44645,
            1.44645,
            1.44645,
            1.44645,
            1.44645,
            1.44645,
            1.44645,
            1.44645,
            1.44645,
            1.44645,
            1.44645,
            1.44645,
            1.44645,
            1.44645,
            1.44645,
            1.44645,
        ]
    )
    y_vector_htp = np.array(
        [
            0.03932,
            0.11797,
            0.19661,
            0.27526,
            0.35391,
            0.43255,
            0.5112,
            0.58984,
            0.66849,
            0.74713,
            0.82578,
            0.90442,
            0.98307,
            1.06172,
            1.14036,
            1.21901,
            1.29765,
            1.3763,
            1.45494,
            1.53359,
            1.61223,
            1.69088,
            1.76953,
            1.84817,
        ]
    )
    cl_vector_htp = np.array(
        [
            0.10558592,
            0.10618191,
            0.10586742,
            0.10557932,
            0.10526483,
            0.10469962,
            0.10383751,
            0.10288744,
            0.10180321,
            0.10075857,
            0.09939723,
            0.09789734,
            0.09558374,
            0.09371877,
            0.09145355,
            0.08914873,
            0.08635569,
            0.08325035,
            0.07887824,
            0.07424443,
            0.06829986,
            0.06080702,
            0.05132166,
            0.04379143,
        ]
    )
    comp_low_speed(
        XML_FILE,
        use_openvsp=True,
        cl0_wing=0.1147,
        cl_alpha_wing=4.510,
        cm0=-0.0258,
        coeff_k_wing=0.0483,
        cl0_htp=-0.0044,
        cl_alpha_htp=0.5401,
        cl_alpha_htp_isolated=0.8318,
        coeff_k_htp=0.6648,
        y_vector_wing=y_vector_wing,
        cl_vector_wing=cl_vector_wing,
        chord_vector_wing=chord_vector_wing,
        cl_ref_htp=0.0897,
        y_vector_htp=y_vector_htp,
        cl_vector_htp=cl_vector_htp,
    )


def test_2d_hinge_moment():
    """Tests tail hinge-moments."""
    hinge_moment_2d(XML_FILE, ch_alpha_2d=-0.3998, ch_delta_2d=-0.6146)


def test_3d_hinge_moment():
    """Tests tail hinge-moments."""
    hinge_moment_3d(XML_FILE, ch_alpha=-0.2625, ch_delta=-0.6822)


def test_all_hinge_moment():
    """Tests tail hinge-moments full computation."""
    hinge_moments(XML_FILE, ch_alpha=-0.2625, ch_delta=-0.6822)


def test_elevator():
    """Tests elevator contribution."""
    elevator(
        XML_FILE,
        cl_delta_elev=0.5115,
        cd_delta_elev=0.0680,
    )


def test_high_lift():
    """Tests high-lift contribution."""
    high_lift(
        XML_FILE,
        delta_cl0_landing=0.5037,
        delta_cl0_landing_2d=1.0673,
        delta_clmax_landing=0.3613,
        delta_cm_landing=-0.1552,
        delta_cm_landing_2d=-0.2401,
        delta_cd_landing=0.005,
        delta_cd_landing_2d=0.0086,
        delta_cl0_takeoff=0.1930,
        delta_cl0_takeoff_2d=0.4090,
        delta_clmax_takeoff=0.0740,
        delta_cm_takeoff=-0.05949,
        delta_cm_takeoff_2d=-0.0920,
        delta_cd_takeoff=0.0004,
        delta_cd_takeoff_2d=0.0006,
    )


@pytest.mark.skipif(
    system() != "Windows",
    reason="No XFOIL executable available: not computed with empty result folder",
)
def test_extreme_cl_wing_clean():
    """Tests maximum/minimum cl component with default result cl=f(y) curve."""
    wing_extreme_cl_clean(
        XML_FILE,
        cl_max_clean_wing=1.50,
        cl_min_clean_wing=-1.20,
    )


@pytest.mark.skipif(
    system() != "Windows",
    reason="No XFOIL executable available: not computed with empty result folder",
)
def test_extreme_cl_htp_clean():
    """Tests maximum/minimum cl component with default result cl=f(y) curve."""
    htp_extreme_cl_clean(
        XML_FILE,
        cl_max_clean_htp=0.30,
        cl_min_clean_htp=-0.30,
        alpha_max_clean_htp=30.39,
        alpha_min_clean_htp=-30.36,
    )


@pytest.mark.skipif(
    system() != "Windows",
    reason="No XFOIL executable available: not computed with empty result folder",
)
def test_extreme_cl():
    """Tests maximum/minimum cl component with default result cl=f(y) curve."""
    extreme_cl(
        XML_FILE,
        cl_max_takeoff_wing=1.45,
        cl_max_landing_wing=1.73,
    )


def test_l_d_max():
    """Tests best lift/drag component."""
    l_d_max(XML_FILE, l_d_max_=15.422, optimal_cl=0.6475, optimal_cd=0.0419, optimal_alpha=4.92)


def test_cnbeta():
    """Tests cn beta fuselage."""
    cnbeta(XML_FILE, cn_beta_fus=-0.0558)


@pytest.mark.skipif(
    system() != "Windows" or SKIP_STEPS, reason="OPENVSP is windows dependent platform (or skipped)"
)
def test_slipstream_openvsp_cruise():
    """Compute slipstream @ high speed."""
    y_vector_prop_on = np.array(
        [
            0.04,
            0.13,
            0.21,
            0.3,
            0.39,
            0.47,
            0.56,
            0.68,
            0.84,
            1.01,
            1.17,
            1.34,
            1.51,
            1.67,
            1.84,
            2.01,
            2.18,
            2.35,
            2.51,
            2.68,
            2.85,
            3.02,
            3.19,
            3.35,
            3.52,
            3.68,
            3.85,
            4.01,
            4.18,
            4.34,
            4.5,
            4.66,
            4.81,
            4.97,
            5.13,
            5.28,
            5.43,
            5.58,
            5.73,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
    )
    cl_vector_prop_on = np.array(
        [
            1.44114,
            1.42469,
            1.424,
            1.4236,
            1.42298,
            1.4217,
            1.42592,
            1.41841,
            1.41474,
            1.42406,
            1.46203,
            1.49892,
            1.51803,
            1.51493,
            1.49662,
            1.41779,
            1.38589,
            1.36438,
            1.34718,
            1.34213,
            1.30898,
            1.3053,
            1.29663,
            1.28695,
            1.26843,
            1.25268,
            1.23072,
            1.21016,
            1.18169,
            1.15659,
            1.12409,
            1.09145,
            1.04817,
            1.00535,
            0.94375,
            0.87738,
            0.78817,
            0.67273,
            0.6258,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
    )
    slipstream_openvsp_cruise(
        XML_FILE,
        ENGINE_WRAPPER,
        y_vector_prop_on=y_vector_prop_on,
        cl_vector_prop_on=cl_vector_prop_on,
        ct=0.04436,
        delta_cl=0.00635,
    )


@pytest.mark.skipif(
    system() != "Windows" or SKIP_STEPS, reason="OPENVSP is windows dependent platform (or skipped)"
)
def test_slipstream_openvsp_low_speed():
    """Compute slipstream @ low speed."""
    y_vector_prop_on = np.array(
        [
            0.04,
            0.13,
            0.21,
            0.3,
            0.39,
            0.47,
            0.56,
            0.68,
            0.84,
            1.01,
            1.17,
            1.34,
            1.51,
            1.67,
            1.84,
            2.01,
            2.18,
            2.35,
            2.51,
            2.68,
            2.85,
            3.02,
            3.19,
            3.35,
            3.52,
            3.68,
            3.85,
            4.01,
            4.18,
            4.34,
            4.5,
            4.66,
            4.81,
            4.97,
            5.13,
            5.28,
            5.43,
            5.58,
            5.73,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
    )
    cl_vector_prop_on = np.array(
        [
            1.46684,
            1.4263,
            1.42597,
            1.42597,
            1.42592,
            1.42528,
            1.4305,
            1.42393,
            1.42232,
            1.43533,
            1.53227,
            1.64675,
            1.68906,
            1.69715,
            1.66992,
            1.5344,
            1.48763,
            1.45131,
            1.4174,
            1.39078,
            1.2942,
            1.29499,
            1.28963,
            1.28212,
            1.26494,
            1.25025,
            1.22919,
            1.20921,
            1.18114,
            1.15665,
            1.125,
            1.09314,
            1.05066,
            1.00871,
            0.94801,
            0.88231,
            0.79402,
            0.679,
            0.63579,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
    )
    slipstream_openvsp_low_speed(
        XML_FILE,
        ENGINE_WRAPPER,
        y_vector_prop_on=y_vector_prop_on,
        cl_vector_prop_on=cl_vector_prop_on,
        ct=0.03487,
        delta_cl=0.02241,
    )


def test_compute_mach_interpolation_roskam():
    """Tests computation of the mach interpolation vector using Roskam's approach."""
    compute_mach_interpolation_roskam(
        XML_FILE,
        cl_alpha_vector=np.array([5.33, 5.35, 5.42, 5.54, 5.72, 5.96]),
        mach_vector=np.array([0.0, 0.08, 0.15, 0.23, 0.31, 0.39]),
    )


def test_non_equilibrated_cl_cd_polar():
    """Tests computation of the non equilibrated cl/cd polar computation."""
    non_equilibrated_cl_cd_polar(
        XML_FILE,
        cl_polar_ls_=np.array(
            [0.25, 0.33, 0.4, 0.48, 0.56, 0.64, 0.72, 0.8, 0.88, 0.96, 1.04, 1.12, 1.2, 1.28, 1.36]
        ),
        cd_polar_ls_=np.array(
            [
                0.044,
                0.0463,
                0.0492,
                0.0528,
                0.057,
                0.0618,
                0.0673,
                0.0734,
                0.0801,
                0.0875,
                0.0955,
                0.1041,
                0.1134,
                0.1233,
                0.1339,
            ]
        ),
        cl_polar_cruise_=np.array(
            [
                0.25,
                0.33,
                0.41,
                0.49,
                0.57,
                0.66,
                0.74,
                0.82,
                0.9,
                0.98,
                1.06,
                1.14,
                1.22,
                1.31,
                1.39,
            ]
        ),
        cd_polar_cruise_=np.array(
            [
                0.0241,
                0.0265,
                0.0295,
                0.0332,
                0.0375,
                0.0425,
                0.0482,
                0.0545,
                0.0615,
                0.0691,
                0.0774,
                0.0864,
                0.096,
                0.1063,
                0.1172,
            ]
        ),
    )


def test_equilibrated_cl_cd_polar():
    """Tests computation of the non equilibrated cl/cd polar computation."""
    equilibrated_cl_cd_polar(
        XML_FILE,
        cl_polar_ls_=np.array(
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.59, 0.69, 0.79, 0.88, 0.98, 1.08, 1.17, 1.26]
        ),
        cd_polar_ls_=np.array(
            [
                0.042,
                0.0439,
                0.0469,
                0.0509,
                0.056,
                0.0622,
                0.0695,
                0.0778,
                0.0872,
                0.0976,
                0.1091,
                0.1216,
                0.1352,
            ]
        ),
        cl_polar_cruise_=np.array(
            [0.03, 0.14, 0.24, 0.34, 0.45, 0.55, 0.65, 0.75, 0.86, 0.96, 1.06, 1.16, 1.26]
        ),
        cd_polar_cruise_=np.array(
            [
                0.0214,
                0.0225,
                0.0249,
                0.0285,
                0.0333,
                0.0392,
                0.0464,
                0.0547,
                0.0641,
                0.0747,
                0.0865,
                0.0994,
                0.1134,
            ]
        ),
    )


def test_cl_alpha_vt():
    """Tests Cl alpha vt."""
    cl_alpha_vt(XML_FILE, cl_alpha_vt_ls=2.6814, k_ar_effective=1.8632, cl_alpha_vt_cruise=2.7322)


def test_cy_delta_r():
    """Tests cy delta of the rudder."""
    cy_delta_r(XML_FILE, cy_delta_r_=1.8882, cy_delta_r_cruise=1.9241)


def test_effective_efficiency():
    """Tests cy delta of the rudder."""
    effective_efficiency(
        XML_FILE, effective_efficiency_low_speed=0.9792, effective_efficiency_cruise=0.9850
    )


def test_cm_alpha_fus():
    """Tests cy delta of the rudder."""
    cm_alpha_fus(XML_FILE, cm_alpha_fus_=-0.2018)


@pytest.mark.skipif(
    system() != "Windows" or SKIP_STEPS, reason="OPENVSP is windows dependent platform (or skipped)"
)
def test_high_speed_connection_openvsp():
    """Tests high speed components connection."""
    high_speed_connection(XML_FILE, ENGINE_WRAPPER, use_openvsp=True)


@pytest.mark.skipif(SKIP_STEPS, reason="Skip test because already performed on Cirrus")
def test_high_speed_connection_vlm():
    """Tests high speed components connection."""
    high_speed_connection(XML_FILE, ENGINE_WRAPPER, use_openvsp=False)


@pytest.mark.skipif(
    system() != "Windows" or SKIP_STEPS, reason="OPENVSP is windows dependent platform (or skipped)"
)
def test_low_speed_connection_openvsp():
    """Tests low speed components connection."""
    low_speed_connection(XML_FILE, ENGINE_WRAPPER, use_openvsp=True)


@pytest.mark.skipif(SKIP_STEPS, reason="Skip test because already performed on Cirrus")
def test_low_speed_connection_vlm():
    """Tests low speed components connection."""
    low_speed_connection(XML_FILE, ENGINE_WRAPPER, use_openvsp=False)


@pytest.mark.skipif(SKIP_STEPS, reason="Skip test because already performed on Cirrus")
def test_v_n_diagram():
    # load all inputs
    velocity_vect = np.array(
        [
            34.688,
            45.763,
            67.62,
            56.42,
            0.0,
            0.0,
            77.998,
            77.998,
            77.998,
            109.139,
            109.139,
            109.139,
            109.139,
            98.225,
            77.998,
            0.0,
            30.871,
            43.659,
            55.568,
        ]
    )
    load_factor_vect = np.array(
        [
            1.0,
            -1.0,
            3.8,
            -1.52,
            0.0,
            0.0,
            -1.52,
            3.8,
            -1.52,
            3.8,
            0.0,
            2.805,
            -0.805,
            0.0,
            0.0,
            0.0,
            1.0,
            2.0,
            2.0,
        ]
    )
    v_n_diagram(
        XML_FILE, ENGINE_WRAPPER, velocity_vect=velocity_vect, load_factor_vect=load_factor_vect
    )


def test_load_factor():
    # load all inputs
    load_factor(
        XML_FILE,
        ENGINE_WRAPPER,
        load_factor_ultimate=5.7,
        load_factor_ultimate_mtow=5.7,
        load_factor_ultimate_mzfw=5.7,
        vh=102.09,
        va=67.62,
        vc=77.998,
        vd=109.139,
    )


@pytest.mark.skipif(
    system() != "Windows" and xfoil_path is None,
    reason="No XFOIL executable available",
)
def test_propeller():
    thrust_SL = np.array(
        [
            118.15673631,
            311.26564173,
            504.37454715,
            697.48345257,
            890.59235799,
            1083.70126341,
            1276.81016883,
            1469.91907425,
            1663.02797967,
            1856.13688509,
            2049.24579051,
            2242.35469593,
            2435.46360135,
            2628.57250677,
            2821.68141218,
            3014.7903176,
            3207.89922302,
            3401.00812844,
            3594.11703386,
            3787.22593928,
            3980.3348447,
            4173.44375012,
            4366.55265554,
            4559.66156096,
            4752.77046638,
            4945.8793718,
            5138.98827722,
            5332.09718264,
            5525.20608806,
            5718.31499348,
        ]
    )
    thrust_SL_limit = np.array(
        [
            3907.31825729,
            4181.05876975,
            4420.52060274,
            4631.7240291,
            4824.74356194,
            5003.75939428,
            5173.8483978,
            5347.00203048,
            5528.42049134,
            5718.31499348,
        ]
    )
    efficiency_SL = np.array(
        [
            [
                0.05351056,
                0.12369987,
                0.16576991,
                0.18753347,
                0.19610143,
                0.19774578,
                0.19497625,
                0.19050657,
                0.18467932,
                0.17848776,
                0.17212624,
                0.16592208,
                0.15973666,
                0.15368017,
                0.14778882,
                0.14190746,
                0.13611303,
                0.13019477,
                0.12349539,
                0.11370028,
                0.09811579,
                0.09811579,
                0.09811579,
                0.09811579,
                0.09811579,
                0.09811579,
                0.09811579,
                0.09811579,
                0.09811579,
                0.09811579,
            ],
            [
                0.15316419,
                0.31326342,
                0.39883791,
                0.44166287,
                0.45971816,
                0.46482553,
                0.46214488,
                0.45529492,
                0.44618283,
                0.43534621,
                0.42442907,
                0.41289184,
                0.4015802,
                0.39015373,
                0.37883427,
                0.36760548,
                0.35634224,
                0.34521311,
                0.33381119,
                0.32116257,
                0.3051059,
                0.26872978,
                0.2669735,
                0.2669735,
                0.2669735,
                0.2669735,
                0.2669735,
                0.2669735,
                0.2669735,
                0.2669735,
            ],
            [
                0.23249499,
                0.43298564,
                0.53077635,
                0.57723775,
                0.59931425,
                0.60719643,
                0.60682063,
                0.60214255,
                0.59429834,
                0.58465784,
                0.57413486,
                0.5629566,
                0.55156923,
                0.53994784,
                0.52822539,
                0.51638604,
                0.50428367,
                0.49240366,
                0.48039802,
                0.46761527,
                0.4539687,
                0.43678709,
                0.40504255,
                0.38560261,
                0.38560261,
                0.38560261,
                0.38560261,
                0.38560261,
                0.38560261,
                0.38560261,
            ],
            [
                0.28840121,
                0.50414683,
                0.60210151,
                0.65013385,
                0.67325264,
                0.68390388,
                0.68591645,
                0.68442004,
                0.67895396,
                0.67172214,
                0.66348699,
                0.65409219,
                0.64453445,
                0.63431149,
                0.62401009,
                0.61326678,
                0.60245716,
                0.59111481,
                0.58005166,
                0.56876875,
                0.55633772,
                0.54302513,
                0.52589979,
                0.49551323,
                0.47001107,
                0.47001107,
                0.47001107,
                0.47001107,
                0.47001107,
                0.47001107,
            ],
            [
                0.32153772,
                0.54285785,
                0.63999146,
                0.68825158,
                0.71392129,
                0.72620207,
                0.73108788,
                0.73151274,
                0.72880071,
                0.72415033,
                0.71808743,
                0.71110838,
                0.70341823,
                0.69518753,
                0.6865335,
                0.67760406,
                0.66827851,
                0.65857903,
                0.64869194,
                0.63887508,
                0.62839374,
                0.61674969,
                0.60379898,
                0.58653785,
                0.555843,
                0.53159122,
                0.53159122,
                0.53159122,
                0.53159122,
                0.53159122,
            ],
            [
                0.328329,
                0.55955737,
                0.6570821,
                0.70748245,
                0.73497363,
                0.7496915,
                0.7576936,
                0.76013781,
                0.75974212,
                0.7570607,
                0.75288364,
                0.74787114,
                0.74211586,
                0.73562068,
                0.72877219,
                0.72140578,
                0.71372128,
                0.7055231,
                0.69734727,
                0.68860765,
                0.67975807,
                0.67018117,
                0.65943572,
                0.64691571,
                0.62970018,
                0.59786839,
                0.5773866,
                0.5773866,
                0.5773866,
                0.5773866,
            ],
            [
                0.32548355,
                0.5616144,
                0.66242482,
                0.71519265,
                0.74460182,
                0.76219025,
                0.77218612,
                0.77729124,
                0.77873748,
                0.77856164,
                0.77636717,
                0.7729243,
                0.76855098,
                0.7635329,
                0.75809558,
                0.75227751,
                0.74598192,
                0.73928651,
                0.73242943,
                0.72510301,
                0.71747231,
                0.7094358,
                0.70065937,
                0.69026841,
                0.67841277,
                0.66021795,
                0.62770511,
                0.61180943,
                0.61180943,
                0.61180943,
            ],
            [
                0.3282459,
                0.56099649,
                0.66171706,
                0.71632964,
                0.74860855,
                0.76828564,
                0.78006309,
                0.78709031,
                0.7908213,
                0.79202528,
                0.79182489,
                0.79014396,
                0.78732694,
                0.78384068,
                0.7793503,
                0.77471876,
                0.76961938,
                0.76414774,
                0.75845943,
                0.75236972,
                0.74611564,
                0.73913714,
                0.73174018,
                0.72363914,
                0.71371799,
                0.70168895,
                0.68466274,
                0.64945049,
                0.63819146,
                0.63819146,
            ],
            [
                0.31094986,
                0.54852371,
                0.65684243,
                0.71388791,
                0.74828881,
                0.7703972,
                0.78437249,
                0.79278355,
                0.79820335,
                0.80097085,
                0.80208926,
                0.80203018,
                0.80059075,
                0.79826533,
                0.79541133,
                0.79169238,
                0.78753339,
                0.78305976,
                0.77824208,
                0.7733003,
                0.76800027,
                0.76234319,
                0.7559611,
                0.74913753,
                0.74140214,
                0.73208963,
                0.72024775,
                0.70290375,
                0.66289193,
                0.65867508,
            ],
            [
                0.31277394,
                0.53942159,
                0.64736691,
                0.70877075,
                0.74531503,
                0.76957757,
                0.78550341,
                0.79584958,
                0.80283755,
                0.80696136,
                0.80907387,
                0.81015072,
                0.81006266,
                0.80891064,
                0.80702879,
                0.80455954,
                0.80146148,
                0.79781889,
                0.79379633,
                0.78960261,
                0.78522748,
                0.78047135,
                0.77534987,
                0.7693819,
                0.76305555,
                0.75564827,
                0.7465654,
                0.73493111,
                0.71859512,
                0.67452182,
            ],
        ]
    )
    thrust_CL = np.array(
        [
            93.04070182,
            245.24623342,
            397.45176502,
            549.65729663,
            701.86282823,
            854.06835984,
            1006.27389144,
            1158.47942305,
            1310.68495465,
            1462.89048625,
            1615.09601786,
            1767.30154946,
            1919.50708107,
            2071.71261267,
            2223.91814427,
            2376.12367588,
            2528.32920748,
            2680.53473909,
            2832.74027069,
            2984.9458023,
            3137.1513339,
            3289.3568655,
            3441.56239711,
            3593.76792871,
            3745.97346032,
            3898.17899192,
            4050.38452352,
            4202.59005513,
            4354.79558673,
            4507.00111834,
        ]
    )
    thrust_CL_limit = np.array(
        [
            3076.9094815,
            3292.5376635,
            3481.52731959,
            3648.25461414,
            3800.28621691,
            3941.71129593,
            4076.27881492,
            4213.10525328,
            4356.4581707,
            4507.00111834,
        ]
    )
    efficiency_CL = np.array(
        [
            [
                0.05135009,
                0.11983937,
                0.16167256,
                0.18384129,
                0.1929643,
                0.19511278,
                0.19281487,
                0.18864496,
                0.18311458,
                0.17712183,
                0.17094633,
                0.16486827,
                0.15878814,
                0.15281727,
                0.14698438,
                0.14114596,
                0.13537194,
                0.12945959,
                0.12272385,
                0.11272061,
                0.09708633,
                0.09708633,
                0.09708633,
                0.09708633,
                0.09708633,
                0.09708633,
                0.09708633,
                0.09708633,
                0.09708633,
                0.09708633,
            ],
            [
                0.14755245,
                0.3048098,
                0.3906136,
                0.43446758,
                0.45365535,
                0.45967628,
                0.45787029,
                0.4515536,
                0.44296718,
                0.43250911,
                0.42192118,
                0.41061643,
                0.39952801,
                0.38826037,
                0.37708164,
                0.36594092,
                0.35473375,
                0.34364815,
                0.33225337,
                0.3195339,
                0.30328742,
                0.26563428,
                0.26467343,
                0.26467343,
                0.26467343,
                0.26467343,
                0.26467343,
                0.26467343,
                0.26467343,
                0.26467343,
            ],
            [
                0.2245676,
                0.4228998,
                0.52167645,
                0.56943738,
                0.59277614,
                0.60158177,
                0.6020627,
                0.5979322,
                0.59064276,
                0.58137893,
                0.57118341,
                0.56026204,
                0.5491086,
                0.53766664,
                0.52609059,
                0.514361,
                0.50235048,
                0.49049442,
                0.47852491,
                0.46573479,
                0.45200952,
                0.43462079,
                0.40181577,
                0.38261237,
                0.38261237,
                0.38261237,
                0.38261237,
                0.38261237,
                0.38261237,
                0.38261237,
            ],
            [
                0.27967917,
                0.49392669,
                0.59311081,
                0.64258772,
                0.66687849,
                0.67850119,
                0.68113913,
                0.68028773,
                0.67527854,
                0.66837437,
                0.66049801,
                0.65129405,
                0.64196886,
                0.6319196,
                0.62175338,
                0.61112683,
                0.60040971,
                0.58909036,
                0.57807955,
                0.56681535,
                0.55435068,
                0.54094259,
                0.52356733,
                0.4921342,
                0.46660809,
                0.46660809,
                0.46660809,
                0.46660809,
                0.46660809,
                0.46660809,
            ],
            [
                0.31293061,
                0.5332046,
                0.63161836,
                0.6811634,
                0.70786346,
                0.72103065,
                0.72651777,
                0.72751346,
                0.72522138,
                0.7208658,
                0.71510153,
                0.70834182,
                0.70083442,
                0.69278071,
                0.68425382,
                0.67543234,
                0.66619774,
                0.65655044,
                0.64671071,
                0.63690086,
                0.62641666,
                0.61471349,
                0.60167923,
                0.58411195,
                0.55221499,
                0.52790009,
                0.52790009,
                0.52790009,
                0.52790009,
                0.52790009,
            ],
            [
                0.31866055,
                0.54975669,
                0.64888694,
                0.70060032,
                0.72918717,
                0.7446237,
                0.75326111,
                0.75623227,
                0.75622306,
                0.7538538,
                0.74993928,
                0.7451137,
                0.73955694,
                0.73320501,
                0.7264931,
                0.71922883,
                0.7116379,
                0.70350186,
                0.69536154,
                0.68662705,
                0.67780267,
                0.66819969,
                0.6574029,
                0.64469433,
                0.62715997,
                0.59414203,
                0.57348938,
                0.57348938,
                0.57348938,
                0.57348938,
            ],
            [
                0.31732236,
                0.55278806,
                0.65480547,
                0.70875745,
                0.73908033,
                0.75734002,
                0.76781983,
                0.77339932,
                0.77521285,
                0.77533137,
                0.77339588,
                0.77018623,
                0.76598708,
                0.7611292,
                0.75579066,
                0.75008531,
                0.74387844,
                0.73725439,
                0.73043446,
                0.72311792,
                0.71552307,
                0.70748654,
                0.69866654,
                0.68822751,
                0.67618409,
                0.6576452,
                0.62356461,
                0.60772092,
                0.60772092,
                0.60772092,
            ],
            [
                0.32163726,
                0.55137662,
                0.65396213,
                0.70987047,
                0.74302563,
                0.76346994,
                0.77572446,
                0.78314575,
                0.78726551,
                0.78876916,
                0.7887997,
                0.78736136,
                0.78471906,
                0.78138074,
                0.77703667,
                0.77248105,
                0.76748315,
                0.76209264,
                0.75643085,
                0.75037872,
                0.74417532,
                0.73717379,
                0.7297921,
                0.72162707,
                0.71157604,
                0.69938369,
                0.68191192,
                0.64515458,
                0.63389919,
                0.63389919,
            ],
            [
                0.3030603,
                0.54017683,
                0.64973323,
                0.70717845,
                0.74258302,
                0.76549176,
                0.77988321,
                0.78874977,
                0.79454055,
                0.79765915,
                0.79897036,
                0.79911829,
                0.79788846,
                0.79571316,
                0.79299536,
                0.78941845,
                0.78532497,
                0.78095253,
                0.77615298,
                0.77128604,
                0.76600316,
                0.76036393,
                0.75401661,
                0.74714185,
                0.73942812,
                0.72995236,
                0.71793199,
                0.70010526,
                0.65806114,
                0.65417021,
            ],
            [
                0.30746,
                0.5303226,
                0.63983347,
                0.70226922,
                0.73939038,
                0.76447496,
                0.78079496,
                0.79163968,
                0.79904586,
                0.8034157,
                0.80582171,
                0.80710005,
                0.80725125,
                0.8062281,
                0.80449208,
                0.80215623,
                0.7991652,
                0.79564184,
                0.79163818,
                0.78752989,
                0.78314849,
                0.77845755,
                0.7733498,
                0.76740544,
                0.76103672,
                0.7535545,
                0.74448892,
                0.73261788,
                0.71564063,
                0.66980238,
            ],
        ]
    )
    speed = np.array(
        [
            5.0,
            15.41925926,
            25.83851852,
            36.25777778,
            46.67703704,
            57.0962963,
            67.51555556,
            77.93481481,
            88.35407407,
            98.77333333,
        ]
    )
    propeller(
        XML_FILE,
        thrust_SL=thrust_SL,
        thrust_SL_limit=thrust_SL_limit,
        efficiency_SL=efficiency_SL,
        thrust_CL=thrust_CL,
        thrust_CL_limit=thrust_CL_limit,
        efficiency_CL=efficiency_CL,
        speed=speed,
    )


def test_cy_beta_fus():
    """Tests cy beta of the fuselage."""
    cy_beta_fus(XML_FILE, cy_beta_fus_=-0.2105)


def test_downwash_gradient():
    """Tests cy beta of the fuselage."""
    downwash_gradient(XML_FILE, downwash_gradient_ls_=0.3620, downwash_gradient_cruise_=0.3685)


def test_cl_alpha_dot():
    """Tests cl alpha dot of the aircraft."""
    lift_aoa_rate_derivative(XML_FILE, cl_aoa_dot_low_speed_=1.362, cl_aoa_dot_cruise_=1.397)


def test_cl_q_ht():
    """Tests cl q of the tail."""
    lift_pitch_velocity_derivative_ht(XML_FILE, cl_q_ht_low_speed_=3.763, cl_q_ht_cruise_=3.793)


def test_cl_q_wing():
    """Tests cl q of the wing."""
    lift_pitch_velocity_derivative_wing(
        XML_FILE, cl_q_wing_low_speed_=2.282, cl_q_wing_cruise_=2.370
    )


def test_cl_q_aircraft():
    """Tests cl q of the aircraft."""
    lift_pitch_velocity_derivative_aircraft(XML_FILE, cl_q_low_speed_=6.045, cl_q_cruise_=6.163)


def test_cy_beta_wing():
    """Tests cy beta of the wing."""
    side_force_sideslip_derivative_wing(XML_FILE, cy_beta_wing_=-0.03438)


def test_cy_beta_vt():
    """Tests cy beta of the vertical tail."""
    side_force_sideslip_derivative_vt(
        XML_FILE, cy_beta_vt_low_speed_=-0.2987, cy_beta_vt_cruise_=-0.3044
    )


def test_cy_beta_aircraft():
    """Tests cy beta of the aircraft."""
    side_force_sideslip_aircraft(XML_FILE, cy_beta_low_speed_=-0.5437)


def test_cy_r_aircraft():
    """Tests cy beta of the aircraft."""
    side_force_yaw_rate_aircraft(XML_FILE, cy_yaw_rate_low_speed_=0.2286, cy_yaw_rate_cruise_=0.227)


def test_cy_p_aircraft():
    """Tests cy roll rate of the aircraft."""
    side_force_roll_rate_aircraft(
        XML_FILE, cy_roll_rate_low_speed_=-0.07683, cy_roll_rate_cruise_=-0.0943
    )


def test_cl_beta_wing():
    """Test cl beta of the wing."""
    roll_moment_side_slip_wing(
        XML_FILE, cl_beta_wing_low_speed_=-0.06787, cl_beta_wing_cruise_=-0.0550
    )


def test_cl_beta_ht():
    """Test cl beta of the wing."""
    roll_moment_side_slip_ht(
        XML_FILE, cl_beta_ht_low_speed_=-0.00717844, cl_beta_ht_cruise_=-0.00503697
    )


def test_cl_beta_vt():
    """Test cl beta of the vt."""
    roll_moment_side_slip_vt(XML_FILE, cl_beta_vt_low_speed_=-0.0384166, cl_beta_vt_cruise_=-0.0472)


def test_cl_beta_aircraft():
    """Test cl beta of the aircraft."""
    roll_moment_side_slip_aircraft(
        XML_FILE, cl_beta_low_speed_=-0.11347027, cl_beta_cruise_=-0.1072
    )


def test_cl_p_wing():
    """Test cl p of the wing."""
    roll_moment_roll_rate_wing(XML_FILE, cl_p_wing_low_speed_=-0.5146, cl_p_wing_cruise_=-0.5175)


def test_cl_p_ht():
    """Test cl p of the ht."""
    roll_moment_roll_rate_ht(XML_FILE, cl_p_ht_low_speed_=-0.00749868, cl_p_ht_cruise_=-0.007528)


def test_cl_p_vt():
    """Test cl p of the vt."""
    roll_moment_roll_rate_vt(XML_FILE, cl_p_vt_low_speed_=-0.01557772, cl_p_vt_cruise_=-0.01587499)


def test_cl_p():
    """Test cl p of the aircraft."""
    roll_moment_roll_rate_aircraft(XML_FILE, cl_p_low_speed_=-0.53769672, cl_p_cruise_=-0.541)


def test_cl_r_wing():
    """Test cl r of the wing."""
    roll_moment_yaw_rate_wing(XML_FILE, cl_r_wing_low_speed_=0.18017549, cl_r_wing_cruise_=0.0938)


def test_cl_r_vt():
    """Test cl r of the vt."""
    roll_moment_yaw_rate_vt(XML_FILE, cl_r_vt_low_speed_=0.02940615, cl_r_vt_cruise_=0.0352)


def test_cl_r_aircraft():
    """Test cl r of the aircraft."""
    roll_moment_yaw_rate_aircraft(XML_FILE, cl_r_low_speed_=0.20958164, cl_r_cruise_=0.129)


def test_cl_delta_a_aircraft():
    """Test roll authority of the aileron."""
    roll_authority_aileron(XML_FILE, cl_delta_a_low_speed_=0.400, cl_delta_a_cruise_=0.410)


def test_cl_delta_r_aircraft():
    """Test roll authority of the rudder."""
    roll_moment_rudder(XML_FILE, cl_delta_r_low_speed_=0.02552507, cl_delta_r_cruise_=0.03134563)


def test_cm_q_wing():
    """Test cm q of the wing."""
    pitch_moment_pitch_rate_wing(
        XML_FILE, cm_q_wing_low_speed_=-1.16358205, cm_q_wing_cruise_=-1.22284125
    )


def test_cm_q_ht():
    """Test cm q of the ht."""
    pitch_moment_pitch_rate_ht(XML_FILE, cm_q_ht_low_speed_=-12.423, cm_q_ht_cruise_=-12.522)


def test_cm_q_aircraft():
    """Test cm q of the aircraft."""
    pitch_moment_pitch_rate_aircraft(XML_FILE, cm_q_low_speed_=-13.587, cm_q_cruise_=-13.744)


def test_cm_alpha_dot():
    """Tests cm alpha dot of the aircraft."""
    pitch_moment_aoa_rate_derivative(
        XML_FILE, cm_aoa_dot_low_speed_=-4.497, cm_aoa_dot_cruise_=-4.614
    )


def test_cn_beta_vt():
    """Tests cn beta of the vt."""
    yaw_moment_sideslip_derivative_vt(
        XML_FILE, cn_beta_vt_low_speed_=0.11432046, cn_beta_vt_cruise_=0.1135
    )


def test_cn_beta_aircraft():
    """Tests cn beta of the aircraft."""
    yaw_moment_sideslip_aircraft(XML_FILE, cn_beta_low_speed_=0.0585625)


def test_cn_delta_a_aircraft():
    """Test yaw moment of the aileron."""
    yaw_moment_aileron(XML_FILE, cn_delta_a_low_speed_=-0.01853731, cn_delta_a_cruise_=-0.00979933)


def test_cn_delta_r_aircraft():
    """Test yaw moment of the rudder."""
    yaw_moment_rudder(XML_FILE, cn_delta_r_low_speed_=-0.07595763, cn_delta_r_cruise_=-0.07539728)


def test_cn_p_wing():
    """Test cn p of the wing."""
    yaw_moment_roll_rate_wing(
        XML_FILE, cn_p_wing_low_speed_=0.07106144, cn_p_wing_cruise_=0.03633125
    )


def test_cn_p_vt():
    """Test cn p of the vt."""
    yaw_moment_roll_rate_vt(XML_FILE, cn_p_vt_low_speed_=-0.00751484, cn_p_vt_cruise_=-0.00147149)


def test_cn_p_aircraft():
    """Tests cn p of the aircraft."""
    yaw_moment_roll_rate_aircraft(XML_FILE, cn_p_low_speed_=0.0635466, cn_p_cruise_=0.03485976)


def test_cn_r_wing():
    """Test cn r of the wing."""
    yaw_moment_yaw_rate_wing(
        XML_FILE, cn_r_wing_low_speed_=-0.0093161, cn_r_wing_cruise_=-0.00367176
    )


def test_cn_r_vt():
    """Test cn r of the vt."""
    yaw_moment_yaw_rate_vt(XML_FILE, cn_r_vt_low_speed_=-0.08750698, cn_r_vt_cruise_=-0.08462127)


def test_cn_r_aircraft():
    """Tests cn r of the aircraft."""
    yaw_moment_yaw_rate_aircraft(XML_FILE, cn_r_low_speed_=-0.09682307, cn_r_cruise_=-0.08829304)
