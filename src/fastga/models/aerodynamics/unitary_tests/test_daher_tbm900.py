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
        coeff_k_wing=0.04715152,
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
        coeff_k_wing=0.0467615,
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
            514.2204107,
            1535.78395058,
            2557.34749045,
            3578.91103032,
            4600.4745702,
            5622.03811007,
            6643.60164994,
            7665.16518982,
            8686.72872969,
            9708.29226956,
            10729.85580943,
            11751.41934931,
            12772.98288918,
            13794.54642905,
            14816.10996893,
            15837.6735088,
            16859.23704867,
            17880.80058855,
            18902.36412842,
            19923.92766829,
            20945.49120816,
            21967.05474804,
            22988.61828791,
            24010.18182778,
            25031.74536766,
            26053.30890753,
            27074.8724474,
            28096.43598728,
            29117.99952715,
            30139.56306702,
        ]
    )
    thrust_SL_limit = np.array(
        [
            12690.75906508,
            14452.03746608,
            16361.14103439,
            18127.72811316,
            19830.77998727,
            21624.1966522,
            23535.53147827,
            25569.30975885,
            27796.04974286,
            30139.56306702,
        ]
    )
    efficiency_SL = np.array(
        [
            [
                0.06268934,
                0.13350106,
                0.15124139,
                0.14880385,
                0.14046632,
                0.13121184,
                0.12219378,
                0.11385834,
                0.10624182,
                0.09918256,
                0.09256645,
                0.0859352,
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
                0.26960116,
                0.4833758,
                0.53548431,
                0.53645756,
                0.52098868,
                0.50010553,
                0.47764496,
                0.4557525,
                0.43495506,
                0.41513241,
                0.39617339,
                0.37803318,
                0.36019558,
                0.34075111,
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
                0.39692462,
                0.62802538,
                0.68526128,
                0.69387247,
                0.68506288,
                0.67012715,
                0.65137409,
                0.63214575,
                0.61257988,
                0.59336546,
                0.57456598,
                0.55609898,
                0.53780758,
                0.51992797,
                0.5009556,
                0.47715458,
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
                0.43839235,
                0.67813935,
                0.74059736,
                0.75746823,
                0.75697371,
                0.74898236,
                0.73711288,
                0.72311714,
                0.70828415,
                0.69290688,
                0.67752152,
                0.66192717,
                0.64628267,
                0.63071754,
                0.61503052,
                0.59851886,
                0.57975543,
                0.54839456,
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
                0.44093012,
                0.68810282,
                0.75793588,
                0.78273516,
                0.78921313,
                0.78777122,
                0.78134433,
                0.77275688,
                0.76228855,
                0.75100335,
                0.73911183,
                0.72690827,
                0.71436443,
                0.70150011,
                0.6885878,
                0.67529536,
                0.66139407,
                0.64590332,
                0.62636703,
                0.57987099,
                0.57987099,
                0.57987099,
                0.57987099,
                0.57987099,
                0.57987099,
                0.57987099,
                0.57987099,
                0.57987099,
                0.57987099,
                0.57987099,
            ],
            [
                0.42572439,
                0.6808189,
                0.75936425,
                0.79128769,
                0.80371642,
                0.80731003,
                0.80572445,
                0.80099433,
                0.79449617,
                0.78667394,
                0.77789638,
                0.76852196,
                0.75869622,
                0.74853284,
                0.73791275,
                0.72715887,
                0.71614356,
                0.70434062,
                0.69178903,
                0.67685161,
                0.65523735,
                0.62020807,
                0.62020807,
                0.62020807,
                0.62020807,
                0.62020807,
                0.62020807,
                0.62020807,
                0.62020807,
                0.62020807,
            ],
            [
                0.40280863,
                0.66638953,
                0.75293675,
                0.79096884,
                0.80901759,
                0.81678664,
                0.81895379,
                0.81772708,
                0.81418424,
                0.80913663,
                0.80309676,
                0.79610352,
                0.78855824,
                0.78050145,
                0.77215527,
                0.76329914,
                0.75428123,
                0.744909,
                0.73505882,
                0.72438969,
                0.7127915,
                0.69825585,
                0.67706437,
                0.64359819,
                0.64359819,
                0.64359819,
                0.64359819,
                0.64359819,
                0.64359819,
                0.64359819,
            ],
            [
                0.3792615,
                0.64814155,
                0.74164501,
                0.78578839,
                0.80819361,
                0.81980821,
                0.82518457,
                0.82671003,
                0.82592405,
                0.82322709,
                0.81917973,
                0.81446503,
                0.80880368,
                0.80251485,
                0.79599755,
                0.78887916,
                0.78153444,
                0.77371036,
                0.76576694,
                0.75747299,
                0.74854376,
                0.73878524,
                0.72803671,
                0.71449283,
                0.69400003,
                0.66273073,
                0.66273073,
                0.66273073,
                0.66273073,
                0.66273073,
            ],
            [
                0.37144162,
                0.62755811,
                0.72622209,
                0.7753951,
                0.80198993,
                0.81694045,
                0.82577559,
                0.82995477,
                0.83131511,
                0.83090759,
                0.828885,
                0.82553564,
                0.82189984,
                0.81728407,
                0.81205333,
                0.80663593,
                0.80064911,
                0.79438711,
                0.7878645,
                0.78084178,
                0.77375844,
                0.76627133,
                0.75822326,
                0.74926348,
                0.73951055,
                0.7275205,
                0.70969473,
                0.67700305,
                0.67700305,
                0.67700305,
            ],
            [
                0.3118491,
                0.59748583,
                0.70560858,
                0.75873146,
                0.79020556,
                0.80869589,
                0.82016028,
                0.82729559,
                0.83095838,
                0.83238514,
                0.83224541,
                0.83103328,
                0.82843586,
                0.82539958,
                0.8218673,
                0.81768462,
                0.81302208,
                0.80815502,
                0.8028398,
                0.79728014,
                0.79146556,
                0.78516583,
                0.77885831,
                0.77203601,
                0.76490485,
                0.75685161,
                0.74809396,
                0.7376948,
                0.72311432,
                0.69058815,
            ],
        ]
    )
    thrust_CL = np.array(
        [
            202.47447884,
            626.21849655,
            1049.96251426,
            1473.70653197,
            1897.45054968,
            2321.19456739,
            2744.9385851,
            3168.68260281,
            3592.42662052,
            4016.17063823,
            4439.91465594,
            4863.65867366,
            5287.40269137,
            5711.14670908,
            6134.89072679,
            6558.6347445,
            6982.37876221,
            7406.12277992,
            7829.86679763,
            8253.61081534,
            8677.35483305,
            9101.09885076,
            9524.84286847,
            9948.58688618,
            10372.33090389,
            10796.0749216,
            11219.81893931,
            11643.56295703,
            12067.30697474,
            12491.05099245,
        ]
    )
    thrust_CL_limit = np.array(
        [
            5141.33873835,
            5859.18963494,
            6636.81246242,
            7350.87824006,
            8052.09311552,
            8777.76322785,
            9586.37415027,
            10466.34375643,
            11420.73698075,
            12491.05099245,
        ]
    )
    efficiency_CL = np.array(
        [
            [
                0.05315209,
                0.12295759,
                0.1433242,
                0.14293574,
                0.1358365,
                0.12740032,
                0.11888282,
                0.1108729,
                0.10344103,
                0.09647314,
                0.0898715,
                0.08282266,
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
                0.23452478,
                0.45265347,
                0.51318283,
                0.51954672,
                0.50739007,
                0.48846474,
                0.4672588,
                0.44626738,
                0.42605523,
                0.4065763,
                0.38774885,
                0.36957739,
                0.35116945,
                0.3294163,
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
                0.35036576,
                0.59474664,
                0.66167829,
                0.67560837,
                0.67017371,
                0.65714347,
                0.63965797,
                0.62130002,
                0.60230671,
                0.5834439,
                0.5648472,
                0.54636218,
                0.52794868,
                0.50957248,
                0.48935527,
                0.45696352,
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
                0.39077689,
                0.64582059,
                0.71756227,
                0.73932579,
                0.74190149,
                0.73591063,
                0.72523708,
                0.71211523,
                0.6978055,
                0.6828478,
                0.66768651,
                0.65220974,
                0.63649297,
                0.62069092,
                0.60460499,
                0.58717167,
                0.56571095,
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
                0.39576493,
                0.65525283,
                0.73398151,
                0.76404041,
                0.77348603,
                0.77413337,
                0.76907808,
                0.76131939,
                0.75154811,
                0.74072651,
                0.72917081,
                0.71717986,
                0.70471838,
                0.69176649,
                0.6787151,
                0.66507989,
                0.65052832,
                0.63329784,
                0.60751244,
                0.56670973,
                0.56670973,
                0.56670973,
                0.56670973,
                0.56670973,
                0.56670973,
                0.56670973,
                0.56670973,
                0.56670973,
                0.56670973,
                0.56670973,
            ],
            [
                0.38024791,
                0.64668063,
                0.73325136,
                0.77042095,
                0.78623276,
                0.79218562,
                0.79218849,
                0.78854902,
                0.78288492,
                0.77563758,
                0.76735239,
                0.75831734,
                0.74870596,
                0.7386761,
                0.72808943,
                0.71723943,
                0.70600331,
                0.69375456,
                0.68039679,
                0.66332362,
                0.63331588,
                0.6155829,
                0.6155829,
                0.6155829,
                0.6155829,
                0.6155829,
                0.6155829,
                0.6155829,
                0.6155829,
                0.6155829,
            ],
            [
                0.3492019,
                0.62786604,
                0.72201039,
                0.76654491,
                0.78822383,
                0.79874484,
                0.80296269,
                0.80308491,
                0.80074214,
                0.79642265,
                0.79109959,
                0.78468931,
                0.77747345,
                0.76982277,
                0.76163301,
                0.75298613,
                0.74385626,
                0.73438969,
                0.72440965,
                0.71328198,
                0.70086608,
                0.68391259,
                0.65172812,
                0.6370844,
                0.6370844,
                0.6370844,
                0.6370844,
                0.6370844,
                0.6370844,
                0.6370844,
            ],
            [
                0.3336473,
                0.60105677,
                0.70327433,
                0.75425167,
                0.78153631,
                0.79653488,
                0.80474546,
                0.80831387,
                0.80885529,
                0.80756685,
                0.80433345,
                0.80039957,
                0.79555513,
                0.78976407,
                0.78370943,
                0.77696718,
                0.7699338,
                0.76236512,
                0.75438071,
                0.74600066,
                0.73708924,
                0.72692068,
                0.71554386,
                0.70025224,
                0.67116142,
                0.6501757,
                0.6501757,
                0.6501757,
                0.6501757,
                0.6501757,
            ],
            [
                0.27845075,
                0.56240429,
                0.67402315,
                0.73236067,
                0.76530728,
                0.7846813,
                0.79669147,
                0.80403422,
                0.80765291,
                0.80891893,
                0.80846262,
                0.80661945,
                0.80361676,
                0.80019795,
                0.79587529,
                0.79099252,
                0.78579673,
                0.78002013,
                0.77391773,
                0.76744059,
                0.76030466,
                0.75304301,
                0.74523343,
                0.73622737,
                0.72599615,
                0.71298539,
                0.69187112,
                0.66341644,
                0.66341644,
                0.66341644,
            ],
            [
                0.234251,
                0.50247548,
                0.62379532,
                0.68918689,
                0.72966275,
                0.75517321,
                0.77238402,
                0.78378774,
                0.79105467,
                0.7957369,
                0.79832641,
                0.79913837,
                0.79872723,
                0.79727825,
                0.79503428,
                0.79228652,
                0.7889125,
                0.78521291,
                0.78091034,
                0.77633714,
                0.77133732,
                0.76594595,
                0.76022882,
                0.75380559,
                0.7470767,
                0.73956588,
                0.73109294,
                0.72046528,
                0.70630642,
                0.66200086,
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
