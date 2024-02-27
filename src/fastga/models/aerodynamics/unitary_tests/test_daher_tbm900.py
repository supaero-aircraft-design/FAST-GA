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

from .dummy_engines import ENGINE_WRAPPER_TBM900 as ENGINE_WRAPPER
from .test_functions import (
    xfoil_path,
    compute_reynolds,
    cd0_high_speed,
    cd0_low_speed,
    polar,
    polar_single_aoa,
    polar_single_aoa_inv,
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

XML_FILE = "daher_tbm900.xml"
SKIP_STEPS = True  # avoid some tests to accelerate validation process (intermediary VLM/OpenVSP)


def test_compute_reynolds():
    """Tests high and low speed reynolds calculation."""
    compute_reynolds(
        XML_FILE,
        mach_high_speed=0.53835122,
        reynolds_high_speed=5381384,
        mach_low_speed=0.1284,
        reynolds_low_speed=2993524,
    )


def test_cd0_high_speed():
    """Tests drag coefficient @ high speed."""
    cd0_high_speed(
        XML_FILE,
        ENGINE_WRAPPER,
        cd0_wing=0.0058063,
        cd0_fus=0.0060731,
        cd0_ht=0.00206023,
        cd0_vt=0.00105858,
        cd0_nac=0.0,
        cd0_lg=0.0,
        cd0_other=0.00517708,
        cd0_total=0.02521912,
    )


def test_cd0_low_speed():
    """Tests drag coefficient @ low speed."""
    cd0_low_speed(
        XML_FILE,
        ENGINE_WRAPPER,
        cd0_wing=0.0055692,
        cd0_fus=0.006808,
        cd0_ht=0.00198062,
        cd0_vt=0.00109524,
        cd0_nac=0.0,
        cd0_lg=0.01599339,
        cd0_other=0.00517708,
        cd0_total=0.0457794,
    )


@pytest.mark.skipif(
    system() != "Windows" and xfoil_path is None or SKIP_STEPS,
    reason="No XFOIL executable available",
)
def test_polar():
    """Tests polar execution (XFOIL) @ high and low speed."""
    polar(
        XML_FILE,
        mach_high_speed=0.53835122,
        reynolds_high_speed=5381384,
        mach_low_speed=0.1284,
        reynolds_low_speed=2993524,
        cdp_1_high_speed=0.005546509572901325,
        cl_max_2d=1.637,
        cdp_1_low_speed=0.005067590361445783,
    )


@pytest.mark.skipif(
    system() != "Windows" and xfoil_path is None or SKIP_STEPS,
    reason="No XFOIL executable available (or skipped)",
)
def test_polar_single_aoa():
    """Tests polar execution (XFOIL) @ low speed."""
    polar_single_aoa(
        XML_FILE,
        mach_low_speed=0.1284,
        reynolds_low_speed=2993524,
    )


@pytest.mark.skipif(
    system() != "Windows" and xfoil_path is None or SKIP_STEPS,
    reason="No XFOIL executable available (or skipped)",
)
def test_polar_single_aoa_inv():
    """Tests polar execution (XFOIL) @ high and low speed."""
    polar_single_aoa_inv(
        XML_FILE,
        mach_low_speed=0.1284,
        reynolds_low_speed=2993524,
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
        cl_alpha_wing=6.50349945,
        cl_alpha_htp=6.35198165,
        cl_alpha_vtp=6.35198165,
    )


@pytest.mark.skipif(
    system() != "Windows" and xfoil_path is None or SKIP_STEPS,
    reason="No XFOIL executable available (or skipped)",
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
        cl0_wing=0.03868354,
        cl_ref_wing=1.03681066,
        cl_alpha_wing=5.64556031,
        cm0=-0.03122047,
        coeff_k_wing=0.06560369,
        cl0_htp=-0.00403328,
        cl_alpha_htp=0.74395071,
        cl_alpha_htp_isolated=1.33227995,
        coeff_k_htp=0.26432897,
        cl_alpha_vector=np.array(
            [5.52158091, 5.52158091, 5.84690417, 6.44350232, 7.43487177, 9.28709846]
        ),
        mach_vector=np.array([0.0, 0.15, 0.36967152, 0.54967167, 0.7021609, 0.83444439]),
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
            0.11333333,
            0.34,
            0.56666667,
            0.97062582,
            1.55187746,
            2.1331291,
            2.71438074,
            3.29563238,
            3.87688402,
            4.45813566,
            4.84563676,
            5.0393873,
            5.23313785,
            5.4268884,
            5.62063894,
            5.81438949,
            6.00814004,
        ]
    )
    cl_vector_wing = np.array(
        [
            0.94143722,
            0.93937731,
            0.93476706,
            0.94348403,
            0.95227165,
            0.95171913,
            0.94336188,
            0.92703571,
            0.90112313,
            0.86274848,
            0.82385374,
            0.78356254,
            0.7393898,
            0.68564668,
            0.61586078,
            0.51858127,
            0.3655492,
        ]
    )
    chord_vector_wing = np.array(
        [
            1.79674552,
            1.79674552,
            1.79674552,
            1.75920638,
            1.68412808,
            1.60904979,
            1.53397149,
            1.4588932,
            1.3838149,
            1.30873661,
            1.25868441,
            1.23365831,
            1.20863221,
            1.18360611,
            1.15858002,
            1.13355392,
            1.10852782,
        ]
    )
    y_vector_htp = np.array(
        [
            0.07363186,
            0.22089559,
            0.36815931,
            0.51542303,
            0.66268676,
            0.80995048,
            0.9572142,
            1.10447793,
            1.25174165,
            1.39900537,
            1.5462691,
            1.69353282,
            1.84079655,
            1.98806027,
            2.13532399,
            2.28258772,
            2.42985144,
        ]
    )
    cl_vector_htp = np.array(
        [
            0.12769726,
            0.12932555,
            0.13056921,
            0.13146178,
            0.13200424,
            0.13217929,
            0.13195305,
            0.13127257,
            0.13006013,
            0.12820381,
            0.12554199,
            0.1218372,
            0.11672994,
            0.10964977,
            0.09962262,
            0.08476022,
            0.0603171,
        ]
    )

    comp_low_speed(
        XML_FILE,
        use_openvsp=False,
        cl0_wing=0.03287854,
        cl_ref_wing=0.88122299,
        cl_alpha_wing=4.79836648,
        cm0=-0.02653541,
        coeff_k_wing=0.04760335,
        cl0_htp=-0.00291361,
        cl_alpha_htp=0.70734892,
        cl_alpha_htp_isolated=1.13235305,
        coeff_k_htp=0.26288118,
        y_vector_wing=y_vector_wing,
        cl_vector_wing=cl_vector_wing,
        chord_vector_wing=chord_vector_wing,
        cl_ref_htp=0.12054207,
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
        cl0_wing=0.0804987,
        cl_ref_wing=0.99980785,
        cl_alpha_wing=5.26725348,
        cm0=-0.03043512,
        coeff_k_wing=0.04276782,
        cl0_htp=-0.01313,
        cl_alpha_htp=0.65987549,
        cl_alpha_htp_isolated=1.35235907,
        coeff_k_htp=0.64924079,
        cl_alpha_vector=np.array(
            [5.34284211, 5.34284211, 5.56666463, 5.96296689, 6.58293027, 7.59851953]
        ),
        mach_vector=np.array([0.0, 0.15, 0.36967152, 0.54967167, 0.7021609, 0.83444439]),
    )


@pytest.mark.skipif(
    system() != "Windows" or SKIP_STEPS, reason="OPENVSP is windows dependent platform (or skipped)"
)
def test_openvsp_comp_low_speed():
    """Tests openvsp components @ low speed."""
    y_vector_wing = np.array(
        [
            0.04857,
            0.14571,
            0.24286,
            0.34,
            0.43714,
            0.53429,
            0.63143,
            0.76486,
            0.93543,
            1.10695,
            1.27934,
            1.45248,
            1.62628,
            1.80063,
            1.97542,
            2.15054,
            2.32589,
            2.50136,
            2.67683,
            2.8522,
            3.02736,
            3.20221,
            3.37662,
            3.5505,
            3.72374,
            3.89624,
            4.06789,
            4.23859,
            4.40826,
            4.57678,
            4.74407,
            4.91004,
            5.07461,
            5.23768,
            5.39918,
            5.55904,
            5.71717,
            5.87352,
            6.028,
        ]
    )
    cl_vector_wing = np.array(
        [
            0.94866161,
            0.93039652,
            0.92896299,
            0.92796051,
            0.92574504,
            0.92279777,
            0.91974022,
            0.92063242,
            0.9217552,
            0.93272227,
            0.9412834,
            0.94644614,
            0.94418055,
            0.94455146,
            0.94543364,
            0.94811025,
            0.94685716,
            0.9472381,
            0.94457151,
            0.94313797,
            0.93687251,
            0.93418588,
            0.92913341,
            0.92461225,
            0.91405619,
            0.90660781,
            0.89514952,
            0.88470374,
            0.86913531,
            0.85785747,
            0.84187803,
            0.82618931,
            0.80123772,
            0.77768959,
            0.74080856,
            0.69612827,
            0.632411,
            0.53773731,
            0.43560522,
        ]
    )
    chord_vector_wing = np.array(
        [
            1.78723,
            1.78723,
            1.78723,
            1.78723,
            1.78723,
            1.78723,
            1.78723,
            1.77629,
            1.75435,
            1.7323,
            1.71013,
            1.68786,
            1.66552,
            1.6431,
            1.62063,
            1.59811,
            1.57557,
            1.55302,
            1.53046,
            1.50792,
            1.48541,
            1.46294,
            1.44052,
            1.41818,
            1.39592,
            1.37376,
            1.35171,
            1.32978,
            1.30798,
            1.28634,
            1.26486,
            1.24354,
            1.22242,
            1.20148,
            1.18075,
            1.16023,
            1.13994,
            1.11988,
            1.10006,
        ]
    )
    y_vector_htp = np.array(
        [
            0.05205,
            0.15636,
            0.26067,
            0.36498,
            0.46929,
            0.5736,
            0.67791,
            0.78222,
            0.88653,
            0.99084,
            1.09515,
            1.19946,
            1.30377,
            1.40808,
            1.51239,
            1.6167,
            1.72101,
            1.82532,
            1.92963,
            2.03394,
            2.13825,
            2.24256,
            2.34686,
            2.45117,
        ]
    )
    cl_vector_htp = np.array(
        [
            0.12393312,
            0.10761786,
            0.10756315,
            0.09902943,
            0.09670454,
            0.1008155,
            0.10554734,
            0.10881039,
            0.10709818,
            0.10754401,
            0.10915229,
            0.10992087,
            0.10660585,
            0.1053422,
            0.10343305,
            0.10229249,
            0.10281217,
            0.10171263,
            0.09872036,
            0.09600161,
            0.09152687,
            0.08450023,
            0.07229865,
            0.06189955,
        ]
    )
    comp_low_speed(
        XML_FILE,
        use_openvsp=True,
        cl0_wing=0.06999277,
        cl_ref_wing=0.88722998,
        cl_alpha_wing=4.6824243,
        cm0=-0.02618463,
        coeff_k_wing=0.04241406,
        cl0_htp=-0.01092,
        cl_alpha_htp=0.64939037,
        cl_alpha_htp_isolated=1.20916995,
        coeff_k_htp=0.59581369,
        y_vector_wing=y_vector_wing,
        cl_vector_wing=cl_vector_wing,
        chord_vector_wing=chord_vector_wing,
        cl_ref_htp=0.10242,
        y_vector_htp=y_vector_htp,
        cl_vector_htp=cl_vector_htp,
    )


def test_2d_hinge_moment():
    """Tests tail hinge-moments."""
    hinge_moment_2d(XML_FILE, ch_alpha_2d=-0.5340, ch_delta_2d=-0.7401)


def test_3d_hinge_moment():
    """Tests tail hinge-moments."""
    hinge_moment_3d(XML_FILE, ch_alpha=-0.3816, ch_delta=-0.8232)


def test_all_hinge_moment():
    """Tests tail hinge-moments full computation."""
    hinge_moments(XML_FILE, ch_alpha=-0.3816, ch_delta=-0.8232)


def test_elevator():
    """Tests elevator contribution."""
    elevator(
        XML_FILE,
        cl_delta_elev=0.70624384,
        cd_delta_elev=0.07627108,
    )


def test_high_lift():
    """Tests high-lift contribution."""
    high_lift(
        XML_FILE,
        delta_cl0_landing=0.8282,
        delta_cl0_landing_2d=1.5406,
        delta_clmax_landing=0.6517,
        delta_cm_landing=-0.2148,
        delta_cm_landing_2d=-0.2850,
        delta_cd_landing=0.0200,
        delta_cd_landing_2d=0.0291,
        delta_cl0_takeoff=0.3080,
        delta_cl0_takeoff_2d=0.5729,
        delta_clmax_takeoff=0.1334,
        delta_cm_takeoff=-0.0798,
        delta_cm_takeoff_2d=-0.1060,
        delta_cd_takeoff=0.0016,
        delta_cd_takeoff_2d=0.0023,
    )


@pytest.mark.skipif(
    system() != "Windows",
    reason="No XFOIL executable available: not computed with empty result folder",
)
def test_extreme_cl_wing_clean():
    """Tests maximum/minimum cl component with default result cl=f(y) curve."""
    wing_extreme_cl_clean(
        XML_FILE,
        cl_max_clean_wing=1.79808306,
        cl_min_clean_wing=-1.41861461,
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
        alpha_max_clean_htp=24.50,
        alpha_min_clean_htp=-24.48,
    )


@pytest.mark.skipif(
    system() != "Windows",
    reason="No XFOIL executable available: not computed with empty result folder",
)
def test_extreme_cl():
    """Tests maximum/minimum cl component with default result cl=f(y) curve."""
    extreme_cl(
        XML_FILE,
        cl_max_takeoff_wing=1.68,
        cl_max_landing_wing=2.51,
    )


def test_l_d_max():
    """Tests best lift/drag component."""
    l_d_max(XML_FILE, l_d_max_=14.30, optimal_cl=0.7147, optimal_cd=0.0499811, optimal_alpha=4.57)


def test_cnbeta():
    """Tests cn beta fuselage."""
    cnbeta(XML_FILE, cn_beta_fus=-0.08859236)


@pytest.mark.skipif(
    system() != "Windows" or SKIP_STEPS, reason="OPENVSP is windows dependent platform (or skipped)"
)
def test_slipstream_openvsp_cruise():
    """Compute slipstream @ high speed."""
    y_vector_prop_on = np.array(
        [
            0.04857,
            0.14571,
            0.24286,
            0.34,
            0.43714,
            0.53429,
            0.63143,
            0.76486,
            0.93543,
            1.10695,
            1.27934,
            1.45248,
            1.62628,
            1.80063,
            1.97542,
            2.15054,
            2.32589,
            2.50136,
            2.67683,
            2.8522,
            3.02736,
            3.20221,
            3.37662,
            3.5505,
            3.72374,
            3.89624,
            4.06789,
            4.23859,
            4.40826,
            4.57678,
            4.74407,
            4.91004,
            5.07461,
            5.23768,
            5.39918,
            5.55904,
            5.71717,
            5.87352,
            6.028,
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
            1.30585,
            1.28739,
            1.28509,
            1.28354,
            1.28088,
            1.27676,
            1.2724,
            1.27416,
            1.27672,
            1.29464,
            1.30805,
            1.31665,
            1.31406,
            1.31523,
            1.31804,
            1.32324,
            1.32231,
            1.32402,
            1.32134,
            1.3206,
            1.31276,
            1.31033,
            1.30438,
            1.29931,
            1.28532,
            1.27622,
            1.26118,
            1.24828,
            1.22726,
            1.21202,
            1.18897,
            1.16672,
            1.13046,
            1.09526,
            1.03908,
            0.97323,
            0.87948,
            0.74913,
            0.63005,
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
        ct=0.19231,
        delta_cl=-0.00023,
    )


@pytest.mark.skipif(
    system() != "Windows" or SKIP_STEPS, reason="OPENVSP is windows dependent platform (or skipped)"
)
def test_slipstream_openvsp_low_speed():
    """Compute slipstream @ low speed."""
    y_vector_prop_on = np.array(
        [
            0.04857,
            0.14571,
            0.24286,
            0.34,
            0.43714,
            0.53429,
            0.63143,
            0.76486,
            0.93543,
            1.10695,
            1.27934,
            1.45248,
            1.62628,
            1.80063,
            1.97542,
            2.15054,
            2.32589,
            2.50136,
            2.67683,
            2.8522,
            3.02736,
            3.20221,
            3.37662,
            3.5505,
            3.72374,
            3.89624,
            4.06789,
            4.23859,
            4.40826,
            4.57678,
            4.74407,
            4.91004,
            5.07461,
            5.23768,
            5.39918,
            5.55904,
            5.71717,
            5.87352,
            6.028,
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
            1.33181,
            1.31088,
            1.3092,
            1.30765,
            1.30507,
            1.30082,
            1.29613,
            1.29748,
            1.30044,
            1.31715,
            1.33083,
            1.33932,
            1.33698,
            1.33877,
            1.34169,
            1.34717,
            1.34625,
            1.34845,
            1.34622,
            1.34598,
            1.33811,
            1.33613,
            1.33066,
            1.32613,
            1.31219,
            1.30363,
            1.28906,
            1.27641,
            1.25493,
            1.2406,
            1.21902,
            1.19844,
            1.1628,
            1.13024,
            1.07746,
            1.01483,
            0.92536,
            0.7962,
            0.69244,
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
        ct=0.07741,
        delta_cl=-0.0006,
    )


def test_compute_mach_interpolation_roskam():
    """Tests computation of the mach interpolation vector using Roskam's approach."""
    compute_mach_interpolation_roskam(
        XML_FILE,
        cl_alpha_vector=np.array(
            [5.42636279, 5.53599343, 5.89199453, 6.59288196, 7.87861309, 10.28594769]
        ),
        mach_vector=np.array([0.0, 0.16688888, 0.33377775, 0.50066663, 0.66755551, 0.83444439]),
    )


def test_non_equilibrated_cl_cd_polar():
    """Tests computation of the non equilibrated cl/cd polar computation."""
    non_equilibrated_cl_cd_polar(
        XML_FILE,
        cl_polar_ls_=np.array(
            [
                0.25450869,
                0.33784429,
                0.42117989,
                0.50451549,
                0.58785108,
                0.67118668,
                0.75452228,
                0.83785788,
                0.92119348,
                1.00452908,
                1.08786468,
                1.17120028,
                1.25453588,
                1.33787148,
                1.42120708,
            ]
        ),
        cd_polar_ls_=np.array(
            [
                0.0486462,
                0.05104917,
                0.05412827,
                0.05788349,
                0.06231484,
                0.06742231,
                0.07320592,
                0.07966564,
                0.0868015,
                0.09461348,
                0.10310159,
                0.11226583,
                0.12210619,
                0.13262268,
                0.14381529,
            ]
        ),
        cl_polar_cruise_=np.array(
            [
                0.28860703,
                0.38247004,
                0.47633305,
                0.57019606,
                0.66405907,
                0.75792208,
                0.85178509,
                0.9456481,
                1.03951111,
                1.13337412,
                1.22723713,
                1.32110014,
                1.41496315,
                1.50882615,
                1.60268916,
            ]
        ),
        cd_polar_cruise_=np.array(
            [
                0.02906514,
                0.03214647,
                0.03608975,
                0.040895,
                0.04656222,
                0.05309139,
                0.06048254,
                0.06873564,
                0.07785071,
                0.08782775,
                0.09866675,
                0.11036771,
                0.12293064,
                0.13635553,
                0.15064238,
            ]
        ),
    )


def test_equilibrated_cl_cd_polar():
    """Tests computation of the non equilibrated cl/cd polar computation."""
    equilibrated_cl_cd_polar(
        XML_FILE,
        cl_polar_ls_=np.array(
            [
                0.15627708,
                0.25750628,
                0.35861429,
                0.45953643,
                0.5602089,
                0.66056889,
                0.76055486,
                0.86010669,
                0.95916586,
                1.05767562,
                1.15558103,
                1.2528296,
                1.34937039,
                1.44515559,
            ]
        ),
        cd_polar_ls_=np.array(
            [
                0.04729761,
                0.04954077,
                0.05283397,
                0.05717517,
                0.06256243,
                0.06899388,
                0.07646778,
                0.08498244,
                0.09453621,
                0.10512758,
                0.11675562,
                0.12941718,
                0.14311259,
                0.15783796,
            ]
        ),
        cl_polar_cruise_=np.array(
            [
                0.02836416,
                0.13871714,
                0.24902817,
                0.35922328,
                0.46922918,
                0.57897353,
                0.68838518,
                0.7973945,
                0.90593356,
                1.01393637,
                1.12133911,
                1.22808025,
                1.33410081,
                1.43934438,
            ]
        ),
        cd_polar_cruise_=np.array(
            [
                0.02558912,
                0.02668415,
                0.02900268,
                0.03254172,
                0.03729797,
                0.04326831,
                0.05044917,
                0.05883663,
                0.06842674,
                0.07921535,
                0.0911981,
                0.10437045,
                0.11872772,
                0.13426507,
            ]
        ),
    )


def test_cl_alpha_vt():
    """Tests Cl alpha vt."""
    cl_alpha_vt(XML_FILE, cl_alpha_vt_ls=2.3328, k_ar_effective=1.3001, cl_alpha_vt_cruise=2.5370)


def test_cy_delta_r():
    """Tests cy delta of the rudder."""
    cy_delta_r(XML_FILE, cy_delta_r_=1.6254, cy_delta_r_cruise=1.7677)


def test_effective_efficiency():
    """Tests cy delta of the rudder."""
    effective_efficiency(
        XML_FILE, effective_efficiency_low_speed=0.9641, effective_efficiency_cruise=0.9871
    )


def test_cm_alpha_fus():
    """Tests cy delta of the rudder."""
    cm_alpha_fus(XML_FILE, cm_alpha_fus_=-0.3195)


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
            45.12871046,
            42.68884934,
            84.17368458,
            50.35792871,
            0.0,
            0.0,
            94.30284589,
            94.30284589,
            94.30284589,
            117.87855736,
            117.87855736,
            117.87855736,
            117.87855736,
            106.09070163,
            101.56759831,
            0.0,
            34.42068063,
            48.67819337,
            63.18019465,
        ]
    )
    load_factor_vect = np.array(
        [
            1.0,
            -1.0,
            3.47893912,
            -1.39157565,
            0.0,
            0.0,
            -1.39157565,
            3.47893912,
            -1.39157565,
            3.47893912,
            0.0,
            2.2737062,
            -0.2737062,
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
        load_factor_ultimate=5.218,
        load_factor_ultimate_mtow=5.218,
        load_factor_ultimate_mzfw=5.218,
        vh=94.302,
        va=84.1736,
        vc=94.3028,
        vd=117.8785,
    )


@pytest.mark.skipif(
    system() != "Windows" and xfoil_path is None,
    reason="No XFOIL executable available",
)
def test_propeller():
    thrust_SL = np.array(
        [
            1619.09853324,
            2602.82704064,
            3586.55554805,
            4570.28405545,
            5554.01256285,
            6537.74107026,
            7521.46957766,
            8505.19808506,
            9488.92659247,
            10472.65509987,
            11456.38360727,
            12440.11211468,
            13423.84062208,
            14407.56912948,
            15391.29763689,
            16375.02614429,
            17358.75465169,
            18342.4831591,
            19326.2116665,
            20309.9401739,
            21293.66868131,
            22277.39718871,
            23261.12569611,
            24244.85420352,
            25228.58271092,
            26212.31121832,
            27196.03972573,
            28179.76823313,
            29163.49674053,
            30147.22524794,
        ]
    )
    thrust_SL_limit = np.array(
        [
            12690.75906508,
            14452.03746608,
            16361.14103439,
            18127.72811316,
            19829.48638845,
            21615.24341967,
            23527.81586163,
            25566.66895535,
            27775.34315136,
            30147.22524794,
        ]
    )
    efficiency_SL = np.array(
        [
            [
                0.13570735,
                0.15106952,
                0.1486958,
                0.14073689,
                0.13184291,
                0.123116,
                0.11499258,
                0.10754134,
                0.10066374,
                0.09420358,
                0.08791837,
                0.08032182,
                0.07714809,
                0.07714809,
                0.07714809,
                0.07714809,
                0.07714809,
                0.07714809,
                0.07714809,
                0.07714809,
                0.07714809,
                0.07714809,
                0.07714809,
                0.07714809,
                0.07714809,
                0.07714809,
                0.07714809,
                0.07714809,
                0.07714809,
                0.07714809,
            ],
            [
                0.49086071,
                0.53574852,
                0.53605402,
                0.52145497,
                0.50157065,
                0.47995812,
                0.45882415,
                0.43863121,
                0.41931143,
                0.40089268,
                0.38317133,
                0.36612626,
                0.34837845,
                0.32371967,
                0.32210935,
                0.32210935,
                0.32210935,
                0.32210935,
                0.32210935,
                0.32210935,
                0.32210935,
                0.32210935,
                0.32210935,
                0.32210935,
                0.32210935,
                0.32210935,
                0.32210935,
                0.32210935,
                0.32210935,
                0.32210935,
            ],
            [
                0.63628586,
                0.68407107,
                0.69383579,
                0.68540138,
                0.671207,
                0.65337285,
                0.63490542,
                0.61599581,
                0.59748463,
                0.57928448,
                0.56140313,
                0.54367468,
                0.52648251,
                0.5086539,
                0.48885733,
                0.453276,
                0.453276,
                0.453276,
                0.453276,
                0.453276,
                0.453276,
                0.453276,
                0.453276,
                0.453276,
                0.453276,
                0.453276,
                0.453276,
                0.453276,
                0.453276,
                0.453276,
            ],
            [
                0.68317267,
                0.73934389,
                0.75653735,
                0.7571197,
                0.74961593,
                0.73826701,
                0.72513254,
                0.71087643,
                0.69624596,
                0.68138942,
                0.66642509,
                0.65143485,
                0.636374,
                0.62134979,
                0.60591193,
                0.58906629,
                0.56758262,
                0.52783961,
                0.52783961,
                0.52783961,
                0.52783961,
                0.52783961,
                0.52783961,
                0.52783961,
                0.52783961,
                0.52783961,
                0.52783961,
                0.52783961,
                0.52783961,
                0.52783961,
            ],
            [
                0.68638823,
                0.75623255,
                0.78148109,
                0.78884328,
                0.78798731,
                0.78218123,
                0.77392035,
                0.76417658,
                0.75345291,
                0.742126,
                0.73046688,
                0.71845837,
                0.70620402,
                0.6937825,
                0.68121266,
                0.6681134,
                0.65418756,
                0.63783862,
                0.61382019,
                0.57936623,
                0.57936623,
                0.57936623,
                0.57936623,
                0.57936623,
                0.57936623,
                0.57936623,
                0.57936623,
                0.57936623,
                0.57936623,
                0.57936623,
            ],
            [
                0.68877265,
                0.76114212,
                0.79139152,
                0.80340228,
                0.80708426,
                0.80591733,
                0.80168953,
                0.79567902,
                0.78838095,
                0.7801391,
                0.77127693,
                0.76191848,
                0.75227699,
                0.74222554,
                0.73188952,
                0.72140288,
                0.71042605,
                0.69876636,
                0.68597997,
                0.66988957,
                0.6433912,
                0.6216385,
                0.6216385,
                0.6216385,
                0.6216385,
                0.6216385,
                0.6216385,
                0.6216385,
                0.6216385,
                0.6216385,
            ],
            [
                0.65818804,
                0.74866759,
                0.78883082,
                0.80765768,
                0.81638988,
                0.8187754,
                0.81781098,
                0.81468138,
                0.81016684,
                0.80462498,
                0.79814705,
                0.79102045,
                0.7835012,
                0.77552978,
                0.76725202,
                0.75855161,
                0.74968923,
                0.7405345,
                0.73066572,
                0.72013319,
                0.7081813,
                0.6926135,
                0.66627085,
                0.64455868,
                0.64455868,
                0.64455868,
                0.64455868,
                0.64455868,
                0.64455868,
                0.64455868,
            ],
            [
                0.63895133,
                0.74147953,
                0.78407232,
                0.80558896,
                0.81814227,
                0.82465119,
                0.82639728,
                0.82588777,
                0.82375198,
                0.82014037,
                0.81585385,
                0.81061492,
                0.80482023,
                0.79859641,
                0.79198909,
                0.78508085,
                0.77771636,
                0.77017312,
                0.7623335,
                0.75412308,
                0.74525184,
                0.73571908,
                0.72466562,
                0.710447,
                0.68712086,
                0.65989749,
                0.65989749,
                0.65989749,
                0.65989749,
                0.65989749,
            ],
            [
                0.62699198,
                0.72104764,
                0.77024112,
                0.80077678,
                0.81502794,
                0.82392585,
                0.82951724,
                0.83077812,
                0.83074618,
                0.82960826,
                0.82647097,
                0.82294382,
                0.81905397,
                0.81411439,
                0.80897178,
                0.80354558,
                0.79756561,
                0.79144877,
                0.78493336,
                0.77814803,
                0.77120581,
                0.76387318,
                0.75587905,
                0.74706524,
                0.73722947,
                0.72519168,
                0.70627209,
                0.67974103,
                0.67974103,
                0.67974103,
            ],
            [
                0.6144616,
                0.69134058,
                0.75716697,
                0.78525393,
                0.80610881,
                0.81817152,
                0.8254837,
                0.83021363,
                0.83166414,
                0.83234654,
                0.83110742,
                0.82934953,
                0.8265146,
                0.82316047,
                0.81954338,
                0.81515891,
                0.81060329,
                0.80580983,
                0.80053893,
                0.7950676,
                0.78940613,
                0.78327861,
                0.77694646,
                0.77036819,
                0.76334488,
                0.75549931,
                0.74685472,
                0.73653179,
                0.72229007,
                0.68542609,
            ],
        ]
    )
    thrust_CL = np.array(
        [
            634.37045222,
            1043.01485082,
            1451.65924941,
            1860.30364801,
            2268.94804661,
            2677.5924452,
            3086.2368438,
            3494.88124239,
            3903.52564099,
            4312.17003959,
            4720.81443818,
            5129.45883678,
            5538.10323538,
            5946.74763397,
            6355.39203257,
            6764.03643116,
            7172.68082976,
            7581.32522836,
            7989.96962695,
            8398.61402555,
            8807.25842415,
            9215.90282274,
            9624.54722134,
            10033.19161993,
            10441.83601853,
            10850.48041713,
            11259.12481572,
            11667.76921432,
            12076.41361292,
            12485.05801151,
        ]
    )
    thrust_CL_limit = np.array(
        [
            5141.33873835,
            5859.18963494,
            6636.81246242,
            7350.87824006,
            8051.12299935,
            8792.6564392,
            9588.99001105,
            10466.51566556,
            11423.44161641,
            12485.05801151,
        ]
    )
    efficiency_CL = np.array(
        [
            [
                0.12333056,
                0.14285193,
                0.1430762,
                0.13655046,
                0.12847886,
                0.12021237,
                0.11239194,
                0.10511165,
                0.09829881,
                0.09183378,
                0.08529839,
                0.07593118,
                0.07549633,
                0.07549633,
                0.07549633,
                0.07549633,
                0.07549633,
                0.07549633,
                0.07549633,
                0.07549633,
                0.07549633,
                0.07549633,
                0.07549633,
                0.07549633,
                0.07549633,
                0.07549633,
                0.07549633,
                0.07549633,
                0.07549633,
                0.07549633,
            ],
            [
                0.45440186,
                0.5118183,
                0.51937963,
                0.50870812,
                0.49088433,
                0.47068788,
                0.45035362,
                0.43066549,
                0.41169668,
                0.39337744,
                0.37565149,
                0.35818762,
                0.33929959,
                0.31655753,
                0.31655753,
                0.31655753,
                0.31655753,
                0.31655753,
                0.31655753,
                0.31655753,
                0.31655753,
                0.31655753,
                0.31655753,
                0.31655753,
                0.31655753,
                0.31655753,
                0.31655753,
                0.31655753,
                0.31655753,
                0.31655753,
            ],
            [
                0.59370273,
                0.65886668,
                0.67518688,
                0.6709731,
                0.65894439,
                0.64251976,
                0.62498621,
                0.60659673,
                0.58847376,
                0.57045626,
                0.55261394,
                0.53472844,
                0.51721436,
                0.49857159,
                0.47614623,
                0.44484842,
                0.44484842,
                0.44484842,
                0.44484842,
                0.44484842,
                0.44484842,
                0.44484842,
                0.44484842,
                0.44484842,
                0.44484842,
                0.44484842,
                0.44484842,
                0.44484842,
                0.44484842,
                0.44484842,
            ],
            [
                0.64705371,
                0.71421953,
                0.73756136,
                0.74197754,
                0.73685059,
                0.72691312,
                0.71472819,
                0.70106575,
                0.68687958,
                0.67226796,
                0.65742714,
                0.64246449,
                0.62719711,
                0.61184951,
                0.59577064,
                0.57742956,
                0.54992961,
                0.51657519,
                0.51657519,
                0.51657519,
                0.51657519,
                0.51657519,
                0.51657519,
                0.51657519,
                0.51657519,
                0.51657519,
                0.51657519,
                0.51657519,
                0.51657519,
                0.51657519,
            ],
            [
                0.64690793,
                0.72929195,
                0.7613184,
                0.77255007,
                0.77416574,
                0.77019423,
                0.76283932,
                0.75390046,
                0.74362784,
                0.73268477,
                0.72126512,
                0.70936041,
                0.69709996,
                0.68462952,
                0.67169114,
                0.6582501,
                0.64344859,
                0.62502786,
                0.58921368,
                0.56613697,
                0.56613697,
                0.56613697,
                0.56613697,
                0.56613697,
                0.56613697,
                0.56613697,
                0.56613697,
                0.56613697,
                0.56613697,
                0.56613697,
            ],
            [
                0.64547795,
                0.72994359,
                0.76779654,
                0.78469262,
                0.79132737,
                0.79215213,
                0.78925062,
                0.78418575,
                0.77757981,
                0.76987526,
                0.76138931,
                0.7523206,
                0.74283989,
                0.73283977,
                0.72243211,
                0.71184564,
                0.70058683,
                0.68830915,
                0.67444215,
                0.65560716,
                0.60726434,
                0.60726434,
                0.60726434,
                0.60726434,
                0.60726434,
                0.60726434,
                0.60726434,
                0.60726434,
                0.60726434,
                0.60726434,
            ],
            [
                0.60700744,
                0.71456823,
                0.76288838,
                0.78668932,
                0.7974669,
                0.80207181,
                0.80293046,
                0.80115585,
                0.79759664,
                0.79273702,
                0.78692097,
                0.78020373,
                0.77304794,
                0.76528252,
                0.75718476,
                0.74854281,
                0.73972292,
                0.73034347,
                0.72023666,
                0.70918342,
                0.69611618,
                0.6779283,
                0.62754228,
                0.62754228,
                0.62754228,
                0.62754228,
                0.62754228,
                0.62754228,
                0.62754228,
                0.62754228,
            ],
            [
                0.58736137,
                0.70228109,
                0.74763776,
                0.7775739,
                0.79462253,
                0.80330687,
                0.80724905,
                0.80861033,
                0.80800815,
                0.80525141,
                0.80168133,
                0.79733574,
                0.79220498,
                0.78641121,
                0.78021791,
                0.77356593,
                0.76655101,
                0.75906316,
                0.75118169,
                0.74299536,
                0.7340609,
                0.72404511,
                0.71212,
                0.69607049,
                0.65876601,
                0.65006983,
                0.65006983,
                0.65006983,
                0.65006983,
                0.65006983,
            ],
            [
                0.57003245,
                0.65799217,
                0.72566447,
                0.76032698,
                0.78090251,
                0.79483242,
                0.80194359,
                0.80656216,
                0.80867876,
                0.80839903,
                0.80718959,
                0.80469465,
                0.80151874,
                0.79781389,
                0.7933785,
                0.78846081,
                0.78315016,
                0.77748074,
                0.77158866,
                0.76498277,
                0.75818211,
                0.75089231,
                0.74307419,
                0.73416382,
                0.72411293,
                0.71094159,
                0.6886737,
                0.6544762,
                0.6544762,
                0.6544762,
            ],
            [
                0.43755858,
                0.61043062,
                0.67747725,
                0.7226726,
                0.74973033,
                0.76823355,
                0.78079882,
                0.78878455,
                0.79430656,
                0.79707219,
                0.79883664,
                0.79865031,
                0.79805297,
                0.79596231,
                0.79361861,
                0.79058591,
                0.78715869,
                0.78346059,
                0.7791489,
                0.77457979,
                0.76969748,
                0.76437368,
                0.75865129,
                0.75249474,
                0.74585436,
                0.73849754,
                0.73011223,
                0.71958991,
                0.70573877,
                0.65930626,
            ],
        ]
    )
    speed = np.array(
        [
            5.0,
            26.39407407,
            47.78814815,
            69.18222222,
            90.5762963,
            111.97037037,
            133.36444444,
            154.75851852,
            176.15259259,
            197.54666667,
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
    cy_beta_fus(XML_FILE, cy_beta_fus_=-0.2755)


def test_downwash_gradient():
    """Tests cy beta of the fuselage."""
    downwash_gradient(XML_FILE, downwash_gradient_ls_=0.3673, downwash_gradient_cruise_=0.4137)


def test_cl_alpha_dot():
    """Tests cl alpha dot of the aircraft."""
    lift_aoa_rate_derivative(XML_FILE, cl_aoa_dot_low_speed_=1.700, cl_aoa_dot_cruise_=1.981)


def test_cl_q_ht():
    """Tests cl q of the tail."""
    lift_pitch_velocity_derivative_ht(XML_FILE, cl_q_ht_low_speed_=4.631, cl_q_ht_cruise_=4.790)


def test_cl_q_wing():
    """Tests cl q of the wing."""
    lift_pitch_velocity_derivative_wing(
        XML_FILE, cl_q_wing_low_speed_=2.387, cl_q_wing_cruise_=3.057
    )


def test_cl_q_aircraft():
    """Tests cl q of the aircraft."""
    lift_pitch_velocity_derivative_aircraft(XML_FILE, cl_q_low_speed_=7.018, cl_q_cruise_=7.848)


def test_cy_beta_wing():
    """Tests cy beta of the wing."""
    side_force_sideslip_derivative_wing(XML_FILE, cy_beta_wing_=-0.037245)


def test_cy_beta_vt():
    """Tests cy beta of the vertical tail."""
    side_force_sideslip_derivative_vt(
        XML_FILE, cy_beta_vt_low_speed_=-0.4218, cy_beta_vt_cruise_=-0.4587
    )


def test_cy_beta_aircraft():
    """Tests cy beta of the aircraft."""
    side_force_sideslip_aircraft(XML_FILE, cy_beta_low_speed_=-0.7346)


def test_cy_r_aircraft():
    """Tests cy beta of the aircraft."""
    side_force_yaw_rate_aircraft(
        XML_FILE, cy_yaw_rate_low_speed_=0.3376, cy_yaw_rate_cruise_=0.35616308
    )


def test_cy_p_aircraft():
    """Tests cy beta of the aircraft."""
    side_force_roll_rate_aircraft(
        XML_FILE, cy_roll_rate_low_speed_=-0.1335, cy_roll_rate_cruise_=-0.1704401
    )


def test_cl_beta_wing():
    """Test cl beta of the wing."""
    roll_moment_side_slip_wing(
        XML_FILE, cl_beta_wing_low_speed_=-0.02397188, cl_beta_wing_cruise_=-0.0325224
    )


def test_cl_beta_ht():
    """Test cl beta of the ht."""
    roll_moment_side_slip_ht(
        XML_FILE, cl_beta_ht_low_speed_=-0.00115732, cl_beta_ht_cruise_=0.00032707
    )


def test_cl_beta_vt():
    """Test cl beta of the vt."""
    roll_moment_side_slip_vt(
        XML_FILE, cl_beta_vt_low_speed_=-0.06675062, cl_beta_vt_cruise_=-0.08522005
    )


def test_cl_beta_aircraft():
    """Test cl beta of the vt."""
    roll_moment_side_slip_aircraft(
        XML_FILE, cl_beta_low_speed_=-0.09187982, cl_beta_cruise_=-0.11741537
    )


def test_cl_p_wing():
    """Test cl p of the wing."""
    roll_moment_roll_rate_wing(
        XML_FILE, cl_p_wing_low_speed_=-0.4923, cl_p_wing_cruise_=-0.53197835
    )


def test_cl_p_ht():
    """Test cl p of the ht."""
    roll_moment_roll_rate_ht(XML_FILE, cl_p_ht_low_speed_=-0.01743478, cl_p_ht_cruise_=-0.01836568)


def test_cl_p_vt():
    """Test cl p of the vt."""
    roll_moment_roll_rate_vt(XML_FILE, cl_p_vt_low_speed_=-0.03127147, cl_p_vt_cruise_=-0.03400)


def test_cl_p():
    """Test cl p of the aircraft."""
    roll_moment_roll_rate_aircraft(XML_FILE, cl_p_low_speed_=-0.54109797, cl_p_cruise_=-0.58434182)


def test_cl_r_wing():
    """Test cl r of the wing."""
    roll_moment_yaw_rate_wing(
        XML_FILE, cl_r_wing_low_speed_=0.13941865, cl_r_wing_cruise_=0.07455721
    )


def test_cl_r_vt():
    """Test cl r of the vt."""
    roll_moment_yaw_rate_vt(XML_FILE, cl_r_vt_low_speed_=0.05343321, cl_r_vt_cruise_=0.06617012)


def test_cl_r_aircraft():
    """Test cl r of the aircraft."""
    roll_moment_yaw_rate_aircraft(XML_FILE, cl_r_low_speed_=0.19285186, cl_r_cruise_=0.14072734)


def test_cl_delta_a_aircraft():
    """Test roll authority of the aileron."""
    roll_authority_aileron(XML_FILE, cl_delta_a_low_speed_=0.172, cl_delta_a_cruise_=0.2024)


def test_cl_delta_r_aircraft():
    """Test roll authority of the rudder."""
    roll_moment_rudder(XML_FILE, cl_delta_r_low_speed_=0.04065428, cl_delta_r_cruise_=0.05190618)


def test_cm_q_wing():
    """Test cm q of the wing."""
    pitch_moment_pitch_rate_wing(
        XML_FILE, cm_q_wing_low_speed_=-1.8946773, cm_q_wing_cruise_=-2.5323024
    )


def test_cm_q_ht():
    """Test cm q of the ht."""
    pitch_moment_pitch_rate_ht(XML_FILE, cm_q_ht_low_speed_=-16.900, cm_q_ht_cruise_=-17.482)


def test_cm_q_aircraft():
    """Test cm q of the aircraft."""
    pitch_moment_pitch_rate_aircraft(XML_FILE, cm_q_low_speed_=-18.795, cm_q_cruise_=-20.015)


def test_cm_alpha_dot():
    """Tests cm alpha dot of the aircraft."""
    pitch_moment_aoa_rate_derivative(
        XML_FILE, cm_aoa_dot_low_speed_=-6.207, cm_aoa_dot_cruise_=-7.232
    )


def test_cn_beta_vt():
    """Tests cn beta of the vt."""
    yaw_moment_sideslip_derivative_vt(
        XML_FILE, cn_beta_vt_low_speed_=0.1688, cn_beta_vt_cruise_=0.17808154
    )


def test_cn_beta_aircraft():
    """Tests cn beta of the aircraft."""
    yaw_moment_sideslip_aircraft(XML_FILE, cn_beta_low_speed_=0.0802328)


def test_cn_delta_a_aircraft():
    """Test yaw moment of the aileron."""
    yaw_moment_aileron(XML_FILE, cn_delta_a_low_speed_=-0.01743657, cn_delta_a_cruise_=-0.01172212)


def test_cn_delta_r_aircraft():
    """Test yaw moment of the rudder."""
    yaw_moment_rudder(XML_FILE, cn_delta_r_low_speed_=-0.10282138, cn_delta_r_cruise_=-0.10846663)


def test_cn_p_wing():
    """Test cn p of the wing."""
    yaw_moment_roll_rate_wing(
        XML_FILE, cn_p_wing_low_speed_=0.07430237, cn_p_wing_cruise_=0.03991506
    )


def test_cn_p_vt():
    """Test cn p of the vt."""
    yaw_moment_roll_rate_vt(XML_FILE, cn_p_vt_low_speed_=-0.01157502, cn_p_vt_cruise_=-0.00240313)


def test_cn_p_aircraft():
    """Tests cn p of the aircraft."""
    yaw_moment_roll_rate_aircraft(XML_FILE, cn_p_low_speed_=0.06272734, cn_p_cruise_=0.03751194)


def test_cn_r_wing():
    """Test cn r of the wing."""
    yaw_moment_yaw_rate_wing(
        XML_FILE, cn_r_wing_low_speed_=-0.00724264, cn_r_wing_cruise_=-0.00359692
    )


def test_cn_r_vt():
    """Test cn r of the vt."""
    yaw_moment_yaw_rate_vt(XML_FILE, cn_r_vt_low_speed_=-0.13514142, cn_r_vt_cruise_=-0.13827354)


def test_cn_r_aircraft():
    """Tests cn r of the aircraft."""
    yaw_moment_yaw_rate_aircraft(XML_FILE, cn_r_low_speed_=-0.14238406, cn_r_cruise_=-0.14187046)
