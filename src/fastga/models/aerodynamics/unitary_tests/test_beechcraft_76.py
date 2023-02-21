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
        mach_high_speed=0.24751863,
        reynolds_high_speed=4345303,
        mach_low_speed=0.1179,
        reynolds_low_speed=2746999,
    )


def test_cd0_high_speed():
    """Tests drag coefficient @ high speed."""
    cd0_high_speed(
        XML_FILE,
        ENGINE_WRAPPER,
        cd0_wing=0.0066708,
        cd0_fus=0.00499782,
        cd0_ht=0.00133863,
        cd0_vt=0.00066,
        cd0_nac=0.00211851,
        cd0_lg=0.0,
        cd0_other=0.00313656,
        cd0_total=0.02365775,
    )


def test_cd0_low_speed():
    """Tests drag coefficient @ low speed."""
    cd0_low_speed(
        XML_FILE,
        ENGINE_WRAPPER,
        cd0_wing=0.00708011,
        cd0_fus=0.00546624,
        cd0_ht=0.00143676,
        cd0_vt=0.00074,
        cd0_nac=0.00231293,
        cd0_lg=0.01473242,
        cd0_other=0.00313656,
        cd0_total=0.04363234,
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
        cl_alpha_wing=6.4981,
        cl_alpha_htp=6.3412,
        cl_alpha_vtp=6.3412,
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
        cl0_wing=-0.0004732,
        cl_ref_wing=0.84031781,
        cl_alpha_wing=4.764742,
        cm0=-0.0229502,
        coeff_k_wing=0.05120887,
        cl0_htp=-3.51454933e-05,
        cl_alpha_htp=0.56919111,
        cl_alpha_htp_isolated=0.9230808,
        coeff_k_htp=0.37604037,
        cl_alpha_vector=np.array(
            [5.23342164, 5.23342164, 5.29274838, 5.37031882, 5.4644133, 5.57808899]
        ),
        mach_vector=np.array([0.0, 0.15, 0.21372738, 0.27363573, 0.33015704, 0.38365387]),
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
            0.85770879,
            1.37532637,
            1.89294395,
            2.41056153,
            2.92817911,
            3.44579669,
            3.96341426,
            4.33314111,
            4.55497721,
            4.77681332,
            4.99864942,
            5.22048553,
            5.44232163,
            5.66415774,
        ]
    )
    cl_vector_wing = np.array(
        [
            1.00520647,
            1.00403894,
            1.00150662,
            0.99610044,
            0.97692565,
            0.95058388,
            0.91818959,
            0.87954133,
            0.8334678,
            0.77813305,
            0.72702565,
            0.68384063,
            0.63643825,
            0.58090541,
            0.51256445,
            0.4230701,
            0.29162547,
        ]
    )
    chord_vector_wing = np.array(
        [
            1.44720606,
            1.44720606,
            1.44720606,
            1.44720606,
            1.44720606,
            1.44720606,
            1.44720606,
            1.44720606,
            1.44720606,
            1.44720606,
            1.44720606,
            1.44720606,
            1.44720606,
            1.44720606,
            1.44720606,
            1.44720606,
            1.44720606,
        ]
    )
    y_vector_htp = np.array(
        [
            0.05852163,
            0.17556488,
            0.29260813,
            0.40965139,
            0.52669464,
            0.64373789,
            0.76078115,
            0.8778244,
            0.99486765,
            1.11191091,
            1.22895416,
            1.34599741,
            1.46304067,
            1.58008392,
            1.69712718,
            1.81417043,
            1.93121368,
        ]
    )
    cl_vector_htp = np.array(
        [
            0.11823122,
            0.11796622,
            0.11743024,
            0.11661099,
            0.11548919,
            0.1140374,
            0.11221833,
            0.10998224,
            0.10726321,
            0.10397342,
            0.09999428,
            0.09516184,
            0.08924102,
            0.08187565,
            0.07247648,
            0.05991316,
            0.04129893,
        ]
    )
    comp_low_speed(
        XML_FILE,
        use_openvsp=False,
        cl0_wing=-0.00046164,
        cl_ref_wing=0.81979372,
        cl_alpha_wing=4.64836698,
        cm0=-0.02238966,
        coeff_k_wing=0.049191,
        cl0_htp=3.34496579e-05,
        cl_alpha_htp=0.56372144,
        cl_alpha_htp_isolated=0.90053529,
        coeff_k_htp=0.37870826,
        y_vector_wing=y_vector_wing,
        cl_vector_wing=cl_vector_wing,
        chord_vector_wing=chord_vector_wing,
        cl_ref_htp=0.0984214,
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
        cl0_wing=0.02506812,
        cl_ref_wing=0.81641273,
        cl_alpha_wing=4.53407067,
        cm0=-0.02583991,
        coeff_k_wing=0.04808614,
        cl0_htp=-0.0052,
        cl_alpha_htp=0.60332456,
        cl_alpha_htp_isolated=0.943331,
        coeff_k_htp=0.60878183,
        cl_alpha_vector=np.array(
            [5.06923403, 5.06923403, 5.109545, 5.16185769, 5.22444975, 5.29921608]
        ),
        mach_vector=np.array([0.0, 0.15, 0.21372738, 0.27363573, 0.33015704, 0.38365387]),
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
            0.68003,
            0.84278,
            1.00644,
            1.17093,
            1.33613,
            1.50196,
            1.66832,
            1.83509,
            2.00219,
            2.1695,
            2.33692,
            2.50435,
            2.67168,
            2.8388,
            3.00563,
            3.17205,
            3.33795,
            3.50325,
            3.66783,
            3.83161,
            3.99449,
            4.15637,
            4.31716,
            4.47678,
            4.63514,
            4.79216,
            4.94775,
            5.10185,
            5.25437,
            5.40525,
            5.55442,
            5.70182,
        ]
    )

    cl_vector_wing = np.array(
        [
            0.9884616,
            0.98801056,
            0.98684786,
            0.98589566,
            0.98360033,
            0.98132506,
            0.98172599,
            0.97399807,
            0.96588927,
            0.96512751,
            0.96160935,
            0.95513434,
            0.94396845,
            0.9358797,
            0.93050725,
            0.92201756,
            0.91070133,
            0.90051772,
            0.88846979,
            0.87699318,
            0.86246951,
            0.84976005,
            0.83523637,
            0.82151456,
            0.80373334,
            0.78760595,
            0.76766973,
            0.74956776,
            0.7278875,
            0.707831,
            0.68355472,
            0.65946888,
            0.63002061,
            0.60065253,
            0.56061972,
            0.51724917,
            0.45940506,
            0.382597,
            0.31214366,
        ]
    )

    chord_vector_wing = np.array(
        [
            1.43954,
            1.43954,
            1.43954,
            1.43954,
            1.43954,
            1.43954,
            1.43954,
            1.43951,
            1.43945,
            1.4394,
            1.43934,
            1.4393,
            1.43925,
            1.43921,
            1.43918,
            1.43915,
            1.43912,
            1.4391,
            1.43908,
            1.43906,
            1.43905,
            1.43905,
            1.43904,
            1.43905,
            1.43905,
            1.43906,
            1.43908,
            1.43909,
            1.43911,
            1.43914,
            1.43917,
            1.4392,
            1.43923,
            1.43927,
            1.43931,
            1.43936,
            1.43941,
            1.43946,
            1.43951,
        ]
    )

    y_vector_htp = np.array(
        [
            0.04145,
            0.12436,
            0.20726,
            0.29017,
            0.37308,
            0.45598,
            0.53889,
            0.62179,
            0.7047,
            0.7876,
            0.87051,
            0.95341,
            1.03632,
            1.11923,
            1.20213,
            1.28504,
            1.36794,
            1.45085,
            1.53375,
            1.61666,
            1.69957,
            1.78247,
            1.86538,
            1.94828,
        ]
    )

    cl_vector_htp = np.array(
        [
            0.11539581,
            0.11677618,
            0.11650949,
            0.11654653,
            0.11588474,
            0.11519826,
            0.11432411,
            0.11338575,
            0.11240294,
            0.11133618,
            0.11009903,
            0.10848159,
            0.10604433,
            0.10406637,
            0.10161429,
            0.09916222,
            0.09622367,
            0.09289991,
            0.08814145,
            0.08320272,
            0.07707376,
            0.06899895,
            0.05864738,
            0.05189367,
        ]
    )

    comp_low_speed(
        XML_FILE,
        use_openvsp=True,
        cl0_wing=0.02435647,
        cl_ref_wing=0.80189909,
        cl_alpha_wing=4.45499105,
        cm0=-0.0252786,
        coeff_k_wing=0.04803734,
        cl0_htp=-0.00506,
        cl_alpha_htp=0.59971492,
        cl_alpha_htp_isolated=0.93017298,
        coeff_k_htp=0.60672321,
        y_vector_wing=y_vector_wing,
        cl_vector_wing=cl_vector_wing,
        chord_vector_wing=chord_vector_wing,
        cl_ref_htp=0.09961,
        y_vector_htp=y_vector_htp,
        cl_vector_htp=cl_vector_htp,
    )


def test_2d_hinge_moment():
    """Tests tail hinge-moments."""
    hinge_moment_2d(XML_FILE, ch_alpha_2d=-0.4329339, ch_delta_2d=-0.63319245)


def test_3d_hinge_moment():
    """Tests tail hinge-moments."""
    hinge_moment_3d(XML_FILE, ch_alpha=-0.28421252, ch_delta=-0.7115188)


def test_all_hinge_moment():
    """Tests tail hinge-moments full computation."""
    hinge_moments(XML_FILE, ch_alpha=-0.28421252, ch_delta=-0.7115188)


def test_elevator():
    """Tests elevator contribution."""
    elevator(
        XML_FILE,
        cl_delta_elev=0.60450934,
        cd_delta_elev=0.08806163,
    )


def test_high_lift():
    """Tests high-lift contribution."""
    high_lift(
        XML_FILE,
        delta_cl0_landing=0.54551302,
        delta_cl0_landing_2d=1.0673,
        delta_clmax_landing=0.4826221,
        delta_cm_landing=-0.15461104,
        delta_cm_landing_2d=-0.22680618,
        delta_cd_landing=0.00910609,
        delta_cd_landing_2d=0.01452359,
        delta_cl0_takeoff=0.20908197,
        delta_cl0_takeoff_2d=0.4090,
        delta_clmax_takeoff=0.09883286,
        delta_cm_takeoff=-0.05925868,
        delta_cm_takeoff_2d=-0.08692933,
        delta_cd_takeoff=0.0007248,
        delta_cd_takeoff_2d=0.00115601,
    )


@pytest.mark.skipif(
    system() != "Windows",
    reason="No XFOIL executable available: not computed with empty result folder",
)
def test_extreme_cl_wing_clean():
    """Tests maximum/minimum cl component with default result cl=f(y) curve."""
    wing_extreme_cl_clean(
        XML_FILE,
        cl_max_clean_wing=1.39324834,
        cl_min_clean_wing=-1.1170453,
    )


@pytest.mark.skipif(
    system() != "Windows",
    reason="No XFOIL executable available: not computed with empty result folder",
)
def test_extreme_cl_htp_clean():
    """Tests maximum/minimum cl component with default result cl=f(y) curve."""
    htp_extreme_cl_clean(
        XML_FILE,
        cl_max_clean_htp=0.20504416,
        cl_min_clean_htp=-0.20504416,
        alpha_max_clean_htp=20.854251,
        alpha_min_clean_htp=-20.6853228,
    )


@pytest.mark.skipif(
    system() != "Windows",
    reason="No XFOIL executable available: not computed with empty result folder",
)
def test_extreme_cl():
    """Tests maximum/minimum cl component with default result cl=f(y) curve."""
    extreme_cl(
        XML_FILE,
        cl_max_takeoff_wing=1.37058028,
        cl_max_landing_wing=1.75462697,
    )


def test_l_d_max():
    """Tests best lift/drag component."""
    l_d_max(
        XML_FILE,
        l_d_max_=14.07332227,
        optimal_cl=0.69100423,
        optimal_cd=0.04910029,
        optimal_alpha=6.34816277,
    )


def test_cnbeta():
    """Tests cn beta fuselage."""
    cnbeta(XML_FILE, cn_beta_fus=-0.05511128)


@pytest.mark.skipif(
    system() != "Windows" or SKIP_STEPS, reason="OPENVSP is windows dependent platform (or skipped)"
)
def test_slipstream_openvsp_cruise():
    """Compute slipstream @ high speed."""
    y_vector_prop_on = np.array(
        [
            0.04278,
            0.12834,
            0.21389,
            0.29945,
            0.38501,
            0.47056,
            0.55612,
            0.68003,
            0.84278,
            1.00644,
            1.17093,
            1.33613,
            1.50196,
            1.66832,
            1.83509,
            2.00219,
            2.1695,
            2.33692,
            2.50435,
            2.67168,
            2.8388,
            3.00563,
            3.17205,
            3.33795,
            3.50325,
            3.66783,
            3.83161,
            3.99449,
            4.15637,
            4.31716,
            4.47678,
            4.63514,
            4.79216,
            4.94775,
            5.10185,
            5.25437,
            5.40525,
            5.55442,
            5.70182,
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
            1.33494,
            1.3191,
            1.31826,
            1.31758,
            1.31636,
            1.31456,
            1.3175,
            1.30901,
            1.30355,
            1.30994,
            1.34812,
            1.39154,
            1.39564,
            1.38515,
            1.35682,
            1.26453,
            1.22593,
            1.19808,
            1.1754,
            1.16872,
            1.13314,
            1.12842,
            1.11712,
            1.10448,
            1.08419,
            1.06588,
            1.04182,
            1.01949,
            0.99114,
            0.96563,
            0.93414,
            0.90297,
            0.8639,
            0.82513,
            0.77104,
            0.71322,
            0.63644,
            0.53665,
            0.47169,
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
        ct=0.04721,
        delta_cl=0.00628,
    )


@pytest.mark.skipif(
    system() != "Windows" or SKIP_STEPS, reason="OPENVSP is windows dependent platform (or skipped)"
)
def test_slipstream_openvsp_low_speed():
    """Compute slipstream @ low speed."""
    y_vector_prop_on = np.array(
        [
            0.04278,
            0.12834,
            0.21389,
            0.29945,
            0.38501,
            0.47056,
            0.55612,
            0.68003,
            0.84278,
            1.00644,
            1.17093,
            1.33613,
            1.50196,
            1.66832,
            1.83509,
            2.00219,
            2.1695,
            2.33692,
            2.50435,
            2.67168,
            2.8388,
            3.00563,
            3.17205,
            3.33795,
            3.50325,
            3.66783,
            3.83161,
            3.99449,
            4.15637,
            4.31716,
            4.47678,
            4.63514,
            4.79216,
            4.94775,
            5.10185,
            5.25437,
            5.40525,
            5.55442,
            5.70182,
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
            1.36554,
            1.32923,
            1.32861,
            1.3282,
            1.32733,
            1.32607,
            1.32974,
            1.32184,
            1.31783,
            1.32684,
            1.4227,
            1.52317,
            1.55365,
            1.55264,
            1.5146,
            1.3696,
            1.31747,
            1.27612,
            1.2383,
            1.21103,
            1.11936,
            1.12546,
            1.11832,
            1.10834,
            1.08963,
            1.07249,
            1.04933,
            1.02751,
            0.99931,
            0.97422,
            0.94327,
            0.9126,
            0.87398,
            0.83569,
            0.782,
            0.72436,
            0.64784,
            0.54777,
            0.48697,
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
        delta_cl=0.01881,
    )


def test_compute_mach_interpolation_roskam():
    """Tests computation of the mach interpolation vector using Roskam's approach."""
    compute_mach_interpolation_roskam(
        XML_FILE,
        cl_alpha_vector=np.array(
            [5.40562294, 5.42830832, 5.49748396, 5.61664252, 5.79208041, 6.03371761]
        ),
        mach_vector=np.array([0.0, 0.08, 0.15, 0.23, 0.31, 0.39]),
    )


def test_non_equilibrated_cl_cd_polar():
    """Tests computation of the non equilibrated cl/cd polar computation."""
    non_equilibrated_cl_cd_polar(
        XML_FILE,
        cl_polar_ls_=np.array(
            [
                0.15789689,
                0.23976227,
                0.32162765,
                0.40349303,
                0.4853584,
                0.56722378,
                0.64908916,
                0.73095454,
                0.81281991,
                0.89468529,
                0.97655067,
                1.05841605,
                1.14028142,
                1.2221468,
                1.30401218,
            ]
        ),
        cd_polar_ls_=np.array(
            [
                0.04579904,
                0.04740323,
                0.04966792,
                0.05259312,
                0.05617882,
                0.06042503,
                0.06533174,
                0.07089895,
                0.07712667,
                0.08401489,
                0.09156362,
                0.09977285,
                0.10864258,
                0.11817282,
                0.12836356,
            ]
        ),
        cl_polar_cruise_=np.array(
            [
                0.16184995,
                0.24576488,
                0.32967981,
                0.41359475,
                0.49750968,
                0.58142461,
                0.66533954,
                0.74925448,
                0.83316941,
                0.91708434,
                1.00099928,
                1.08491421,
                1.16882914,
                1.25274407,
                1.33665901,
            ]
        ),
        cd_polar_cruise_=np.array(
            [
                0.02589699,
                0.02765565,
                0.03013842,
                0.03334528,
                0.03727626,
                0.04193134,
                0.04731052,
                0.0534138,
                0.06024119,
                0.06779269,
                0.07606829,
                0.08506799,
                0.0947918,
                0.10523971,
                0.11641173,
            ]
        ),
    )


def test_equilibrated_cl_cd_polar():
    """Tests computation of the non equilibrated cl/cd polar computation."""
    equilibrated_cl_cd_polar(
        XML_FILE,
        cl_polar_ls_=np.array(
            [
                0.10408501578219179,
                0.19431327855240987,
                0.28444351751116903,
                0.3744246466040594,
                0.4642061234117144,
                0.5537381004244202,
                0.642971582813649,
                0.7318585396030874,
                0.8203520736378673,
                0.9084065246775324,
                0.9959775908510616,
                1.0830224383528835,
                1.1694997620439176,
            ]
        ),
        cd_polar_ls_=np.array(
            [
                0.04569379377287508,
                0.04745598415699365,
                0.05011385285440968,
                0.05366646918108492,
                0.058112766538959804,
                0.06345169712210266,
                0.06968211957496939,
                0.07680308700575175,
                0.084813463738039,
                0.09371214501433102,
                0.10349798991711547,
                0.11416979469009379,
                0.1257264554071501,
            ]
        ),
        cl_polar_cruise_=np.array(
            [
                0.0347755285812549,
                0.13008991428672945,
                0.22533510856934935,
                0.3204500808220095,
                0.41537437559975005,
                0.5100483267156494,
                0.6044132823058332,
                0.6984117883531852,
                0.7919877916466692,
                0.8850868108850505,
                0.9776560999126512,
                1.0696447815854904,
                1.1610040414361837,
            ]
        ),
        cd_polar_cruise_=np.array(
            [
                0.024945555795540607,
                0.026161063506548884,
                0.02840335611495485,
                0.03167094585490235,
                0.0359621817626931,
                0.04127532908738969,
                0.04760836317075505,
                0.05495923881290456,
                0.06332570449945522,
                0.0727053752574351,
                0.08309572015310841,
                0.09449412994149958,
                0.10689758501156821,
            ]
        ),
    )


def test_cl_alpha_vt():
    """Tests Cl alpha vt."""
    cl_alpha_vt(
        XML_FILE,
        cl_alpha_vt_ls=2.69764915,
        k_ar_effective=1.88000764,
        cl_alpha_vt_cruise=2.74872417,
    )


def test_cy_delta_r():
    """Tests cy delta of the rudder."""
    cy_delta_r(XML_FILE, cy_delta_r_=1.90111266, cy_delta_r_cruise=1.93710672)


def test_effective_efficiency():
    """Tests cy delta of the rudder."""
    effective_efficiency(
        XML_FILE, effective_efficiency_low_speed=0.97760625, effective_efficiency_cruise=0.9846488
    )


def test_cm_alpha_fus():
    """Tests cy delta of the rudder."""
    cm_alpha_fus(XML_FILE, cm_alpha_fus_=-0.18329318)


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
            36.26439392,
            47.75950095,
            70.69231791,
            58.88186728,
            0.0,
            0.0,
            78.34029128,
            78.34029128,
            78.34029128,
            109.60816966,
            109.60816966,
            109.60816966,
            109.60816966,
            98.64735269,
            78.34029128,
            0.0,
            30.87287928,
            43.66084458,
            55.5711827,
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
            -1.53819611,
            3.8,
            0.0,
            2.84299239,
            -0.84299239,
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
        vh=94.24,
        va=70.69,
        vc=78.34,
        vd=109.608,
    )


@pytest.mark.skipif(
    system() != "Windows" and xfoil_path is None,
    reason="No XFOIL executable available",
)
def test_propeller():
    thrust_SL = np.array(
        [
            118.15673631,
            310.5000776,
            502.84341889,
            695.18676018,
            887.53010147,
            1079.87344276,
            1272.21678406,
            1464.56012535,
            1656.90346664,
            1849.24680793,
            2041.59014922,
            2233.93349051,
            2426.2768318,
            2618.62017309,
            2810.96351438,
            3003.30685567,
            3195.65019696,
            3387.99353825,
            3580.33687954,
            3772.68022083,
            3965.02356212,
            4157.36690341,
            4349.7102447,
            4542.05358599,
            4734.39692728,
            4926.74026857,
            5119.08360986,
            5311.42695115,
            5503.77029244,
            5696.11363373,
        ]
    )
    thrust_SL_limit = np.array(
        [
            3907.31825729,
            4177.75124865,
            4414.4531561,
            4623.80209745,
            4814.68370024,
            4992.45378339,
            5160.39575555,
            5330.68703899,
            5508.83419191,
            5696.11363373,
        ]
    )
    efficiency_SL = np.array(
        [
            [
                0.05351056,
                0.12347525,
                0.1655366,
                0.18736827,
                0.19602648,
                0.19775442,
                0.19508082,
                0.19063943,
                0.18487364,
                0.17870561,
                0.1723861,
                0.16619065,
                0.16002567,
                0.15399872,
                0.14811737,
                0.14224821,
                0.1364682,
                0.13062113,
                0.12400109,
                0.11464944,
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
                0.15212887,
                0.31088291,
                0.39613769,
                0.43886415,
                0.45704566,
                0.46225846,
                0.45973704,
                0.4529451,
                0.44392854,
                0.43317449,
                0.42234271,
                0.41085589,
                0.3996228,
                0.38825868,
                0.37701942,
                0.36584575,
                0.35464311,
                0.34360865,
                0.33232289,
                0.31989642,
                0.30413892,
                0.2726115,
                0.26511387,
                0.26511387,
                0.26511387,
                0.26511387,
                0.26511387,
                0.26511387,
                0.26511387,
                0.26511387,
            ],
            [
                0.23023238,
                0.43000268,
                0.5278268,
                0.57425912,
                0.59650032,
                0.60440535,
                0.60412381,
                0.59945007,
                0.59166534,
                0.58203983,
                0.57155435,
                0.56038186,
                0.54903238,
                0.53743479,
                0.52574809,
                0.51393612,
                0.5018895,
                0.49002145,
                0.47807279,
                0.46540016,
                0.45188711,
                0.43499559,
                0.40519931,
                0.38297328,
                0.38297328,
                0.38297328,
                0.38297328,
                0.38297328,
                0.38297328,
                0.38297328,
            ],
            [
                0.28761175,
                0.50166067,
                0.59939487,
                0.6476175,
                0.67085559,
                0.68159472,
                0.68356075,
                0.68210814,
                0.67661848,
                0.66935286,
                0.66111029,
                0.65167786,
                0.64210062,
                0.63186332,
                0.62154945,
                0.61080021,
                0.59999679,
                0.58862685,
                0.57759716,
                0.5663487,
                0.55396148,
                0.54074353,
                0.52398734,
                0.49506023,
                0.46716978,
                0.46716978,
                0.46716978,
                0.46716978,
                0.46716978,
                0.46716978,
            ],
            [
                0.32069764,
                0.541321,
                0.63822562,
                0.6864212,
                0.71207749,
                0.72433804,
                0.72918718,
                0.7296413,
                0.72687497,
                0.72216855,
                0.71606534,
                0.70902618,
                0.70129704,
                0.69300748,
                0.68433384,
                0.67535611,
                0.66600112,
                0.65628052,
                0.64636828,
                0.63653445,
                0.62606511,
                0.61444673,
                0.60166919,
                0.58468842,
                0.55447484,
                0.52878067,
                0.52878067,
                0.52878067,
                0.52878067,
                0.52878067,
            ],
            [
                0.32615364,
                0.55784211,
                0.65564798,
                0.70611297,
                0.73365953,
                0.74829785,
                0.75625535,
                0.75864927,
                0.75813581,
                0.75543376,
                0.75120066,
                0.74612532,
                0.74031364,
                0.73375519,
                0.72685545,
                0.71943779,
                0.71169491,
                0.7034785,
                0.69523726,
                0.68644327,
                0.67758409,
                0.66802486,
                0.65729226,
                0.64480346,
                0.62784666,
                0.59657543,
                0.57476014,
                0.57476014,
                0.57476014,
                0.57476014,
            ],
            [
                0.32561457,
                0.56130587,
                0.6617216,
                0.71441466,
                0.74379777,
                0.76126431,
                0.77113823,
                0.77613669,
                0.77751853,
                0.77725604,
                0.77497526,
                0.77139447,
                0.76703237,
                0.76191999,
                0.75645111,
                0.75055875,
                0.7442092,
                0.7374678,
                0.73053459,
                0.72316607,
                0.71548982,
                0.70743423,
                0.69864038,
                0.68831136,
                0.67648118,
                0.65864525,
                0.62772564,
                0.60942032,
                0.60942032,
                0.60942032,
            ],
            [
                0.33074231,
                0.56005884,
                0.66128982,
                0.71591916,
                0.74805988,
                0.76759916,
                0.77934301,
                0.78623483,
                0.78986265,
                0.79099224,
                0.79069427,
                0.78891854,
                0.78603615,
                0.78244391,
                0.77788408,
                0.77320643,
                0.76806351,
                0.76253505,
                0.75677387,
                0.75063256,
                0.74432472,
                0.73730386,
                0.72988231,
                0.72174948,
                0.71179596,
                0.69985144,
                0.68290923,
                0.64882489,
                0.63602945,
                0.63602945,
            ],
            [
                0.31198127,
                0.5491285,
                0.65730568,
                0.7136968,
                0.74802543,
                0.77002801,
                0.7838479,
                0.79209167,
                0.79747594,
                0.80013123,
                0.80117461,
                0.80101464,
                0.79949339,
                0.79708448,
                0.79415851,
                0.79034036,
                0.78613476,
                0.78160823,
                0.77673881,
                0.77173767,
                0.76637982,
                0.76067214,
                0.75425053,
                0.74738372,
                0.73965203,
                0.73029931,
                0.71848966,
                0.70115195,
                0.6622709,
                0.65675128,
            ],
            [
                0.31671695,
                0.53980013,
                0.64787489,
                0.70905619,
                0.74527768,
                0.76944244,
                0.78515551,
                0.79539174,
                0.80223884,
                0.80628252,
                0.80830042,
                0.80931701,
                0.80913849,
                0.80789866,
                0.80594715,
                0.8033943,
                0.8002341,
                0.7964891,
                0.79243861,
                0.78818161,
                0.78375784,
                0.77894492,
                0.77377229,
                0.76776127,
                0.76139284,
                0.75393309,
                0.74484163,
                0.73315728,
                0.71673281,
                0.67288554,
            ],
        ]
    )
    thrust_CL = np.array(
        [
            90.62542317,
            233.05398451,
            375.48254584,
            517.91110717,
            660.33966851,
            802.76822984,
            945.19679117,
            1087.6253525,
            1230.05391384,
            1372.48247517,
            1514.9110365,
            1657.33959784,
            1799.76815917,
            1942.1967205,
            2084.62528183,
            2227.05384317,
            2369.4824045,
            2511.91096583,
            2654.33952717,
            2796.7680885,
            2939.19664983,
            3081.62521116,
            3224.0537725,
            3366.48233383,
            3508.91089516,
            3651.3394565,
            3793.76801783,
            3936.19657916,
            4078.62514049,
            4221.05370183,
        ]
    )
    thrust_CL_limit = np.array(
        (
            [
                2892.27511752,
                3092.52843453,
                3268.24054129,
                3423.69146545,
                3565.02500064,
                3697.14844501,
                3822.19984334,
                3948.81792225,
                4081.24433611,
                4221.05370183,
            ]
        )
    )
    efficiency_CL = np.array(
        [
            [
                0.05257415,
                0.11979227,
                0.16096558,
                0.18297609,
                0.19214585,
                0.19440941,
                0.192262,
                0.18819478,
                0.18278735,
                0.17687579,
                0.17079108,
                0.16476462,
                0.15874091,
                0.15282559,
                0.14702722,
                0.141222,
                0.13547444,
                0.12963448,
                0.12296689,
                0.11337435,
                0.09681287,
                0.09681287,
                0.09681287,
                0.09681287,
                0.09681287,
                0.09681287,
                0.09681287,
                0.09681287,
                0.09681287,
                0.09681287,
            ],
            [
                0.14926816,
                0.30255758,
                0.3869083,
                0.43039255,
                0.4495968,
                0.455774,
                0.45421453,
                0.44807051,
                0.43969649,
                0.42941572,
                0.41899432,
                0.40781953,
                0.39687339,
                0.3857249,
                0.37467091,
                0.3636227,
                0.35250853,
                0.34153471,
                0.33026362,
                0.31774249,
                0.30173587,
                0.2674251,
                0.26221563,
                0.26221563,
                0.26221563,
                0.26221563,
                0.26221563,
                0.26221563,
                0.26221563,
                0.26221563,
            ],
            [
                0.2272604,
                0.42014606,
                0.51750325,
                0.56501515,
                0.58842143,
                0.59734449,
                0.59801777,
                0.59400213,
                0.58687457,
                0.57772152,
                0.56764364,
                0.5568075,
                0.5457584,
                0.53439755,
                0.5229025,
                0.51124324,
                0.49931821,
                0.48749642,
                0.47559818,
                0.46292757,
                0.44931731,
                0.43214163,
                0.40096745,
                0.37920128,
                0.37920128,
                0.37920128,
                0.37920128,
                0.37920128,
                0.37920128,
                0.37920128,
            ],
            [
                0.28258325,
                0.49152952,
                0.58941955,
                0.6387258,
                0.66303983,
                0.67480876,
                0.67749572,
                0.67675665,
                0.67183388,
                0.66497311,
                0.65715631,
                0.64799263,
                0.63870722,
                0.62869586,
                0.61856311,
                0.60796659,
                0.59728514,
                0.58596093,
                0.57500529,
                0.56378502,
                0.55135094,
                0.53802846,
                0.52095436,
                0.49114681,
                0.46287248,
                0.46287248,
                0.46287248,
                0.46287248,
                0.46287248,
                0.46287248,
            ],
            [
                0.3169378,
                0.53166132,
                0.62885052,
                0.67806778,
                0.70471403,
                0.71790601,
                0.72341201,
                0.72451657,
                0.72224434,
                0.71790192,
                0.71216668,
                0.70539743,
                0.69790546,
                0.68983891,
                0.68133038,
                0.67249712,
                0.6632631,
                0.65361981,
                0.64377054,
                0.63394701,
                0.62348395,
                0.61179762,
                0.59892232,
                0.58155994,
                0.55010639,
                0.52411352,
                0.52411352,
                0.52411352,
                0.52411352,
                0.52411352,
            ],
            [
                0.32353556,
                0.54884734,
                0.64681185,
                0.69816534,
                0.72670374,
                0.74206723,
                0.75071338,
                0.75368305,
                0.75361529,
                0.75128918,
                0.74738236,
                0.74253023,
                0.7369738,
                0.73059401,
                0.72386936,
                0.71658765,
                0.70896539,
                0.7008357,
                0.69263774,
                0.68385797,
                0.67504122,
                0.66544948,
                0.65464611,
                0.641931,
                0.62457278,
                0.59191798,
                0.56983154,
                0.56983154,
                0.56983154,
                0.56983154,
            ],
            [
                0.32338264,
                0.55300397,
                0.65338506,
                0.70692798,
                0.73716818,
                0.75526264,
                0.76572501,
                0.77124892,
                0.77302957,
                0.77312801,
                0.77115456,
                0.76784993,
                0.76369383,
                0.75879055,
                0.75345259,
                0.74770606,
                0.7414678,
                0.7348289,
                0.72792545,
                0.72059014,
                0.71296217,
                0.70490419,
                0.69607696,
                0.68567164,
                0.67359105,
                0.65535684,
                0.62288359,
                0.60425046,
                0.60425046,
                0.60425046,
            ],
            [
                0.32863484,
                0.55167925,
                0.65307233,
                0.70855423,
                0.74143156,
                0.76170944,
                0.77393145,
                0.78132021,
                0.78538012,
                0.78685849,
                0.78683111,
                0.78533997,
                0.78267127,
                0.77927545,
                0.77487478,
                0.77031219,
                0.76529027,
                0.75987303,
                0.75414226,
                0.74805849,
                0.74181818,
                0.73476803,
                0.72737216,
                0.71915874,
                0.70904889,
                0.6969101,
                0.67936002,
                0.64335958,
                0.63060161,
                0.63060161,
            ],
            [
                0.31057148,
                0.54154904,
                0.6496374,
                0.70616132,
                0.74131582,
                0.76409586,
                0.77833761,
                0.78708837,
                0.79288785,
                0.79594933,
                0.79721747,
                0.79730933,
                0.79603582,
                0.79380921,
                0.79105327,
                0.78741428,
                0.78328362,
                0.77888976,
                0.77404434,
                0.76914398,
                0.7638046,
                0.75812719,
                0.75174459,
                0.7448244,
                0.7370901,
                0.72755628,
                0.71551339,
                0.69761567,
                0.65622207,
                0.65105509,
            ],
            [
                0.31571388,
                0.53176652,
                0.639908,
                0.70145019,
                0.73835971,
                0.76331659,
                0.77943871,
                0.79020517,
                0.79750604,
                0.80185271,
                0.80419706,
                0.80545482,
                0.80555951,
                0.80447942,
                0.80270285,
                0.80031786,
                0.79728609,
                0.79368892,
                0.789671,
                0.78551172,
                0.78109309,
                0.77636061,
                0.77120532,
                0.76522801,
                0.75880436,
                0.75126506,
                0.74219109,
                0.73019016,
                0.7129783,
                0.66692487,
            ],
        ]
    )
    speed = np.array(
        [
            5.0,
            15.28207407,
            25.56414815,
            35.84622222,
            46.1282963,
            56.41037037,
            66.69244444,
            76.97451852,
            87.25659259,
            97.53866667,
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
    cy_beta_fus(XML_FILE, cy_beta_fus_=-0.21272941)


def test_downwash_gradient():
    """Tests cy beta of the fuselage."""
    downwash_gradient(
        XML_FILE, downwash_gradient_ls_=0.37165564, downwash_gradient_cruise_=0.38096029
    )


def test_cl_alpha_dot():
    """Tests cl alpha dot of the aircraft."""
    lift_aoa_rate_derivative(
        XML_FILE, cl_aoa_dot_low_speed_=1.38843186, cl_aoa_dot_cruise_=1.43701383
    )


def test_cl_q_ht():
    """Tests cl q of the tail."""
    lift_pitch_velocity_derivative_ht(
        XML_FILE, cl_q_ht_low_speed_=3.73580197, cl_q_ht_cruise_=3.77208303
    )


def test_cl_q_wing():
    """Tests cl q of the wing."""
    lift_pitch_velocity_derivative_wing(
        XML_FILE, cl_q_wing_low_speed_=2.34270419, cl_q_wing_cruise_=2.44889162
    )


def test_cl_q_aircraft():
    """Tests cl q of the aircraft."""
    lift_pitch_velocity_derivative_aircraft(
        XML_FILE, cl_q_low_speed_=6.07850616, cl_q_cruise_=6.22097465
    )


def test_cy_beta_wing():
    """Tests cy beta of the wing."""
    side_force_sideslip_derivative_wing(XML_FILE, cy_beta_wing_=-0.03438)


def test_cy_beta_vt():
    """Tests cy beta of the vertical tail."""
    side_force_sideslip_derivative_vt(
        XML_FILE, cy_beta_vt_low_speed_=-0.29686064, cy_beta_vt_cruise_=-0.30248115
    )


def test_cy_beta_aircraft():
    """Tests cy beta of the aircraft."""
    side_force_sideslip_aircraft(XML_FILE, cy_beta_low_speed_=-0.5437)


def test_cy_r_aircraft():
    """Tests cy beta of the aircraft."""
    side_force_yaw_rate_aircraft(
        XML_FILE, cy_yaw_rate_low_speed_=0.2286, cy_yaw_rate_cruise_=0.22676074
    )


def test_cy_p_aircraft():
    """Tests cy roll rate of the aircraft."""
    side_force_roll_rate_aircraft(
        XML_FILE, cy_roll_rate_low_speed_=-0.07637348, cy_roll_rate_cruise_=-0.09386615
    )


def test_cl_beta_wing():
    """Test cl beta of the wing."""
    roll_moment_side_slip_wing(
        XML_FILE, cl_beta_wing_low_speed_=-0.06372762, cl_beta_wing_cruise_=-0.05045313
    )


def test_cl_beta_ht():
    """Test cl beta of the wing."""
    roll_moment_side_slip_ht(
        XML_FILE, cl_beta_ht_low_speed_=-0.00769295, cl_beta_ht_cruise_=-0.0054336
    )


def test_cl_beta_vt():
    """Test cl beta of the vt."""
    roll_moment_side_slip_vt(
        XML_FILE, cl_beta_vt_low_speed_=-0.03818674, cl_beta_vt_cruise_=-0.04693308
    )


def test_cl_beta_aircraft():
    """Test cl beta of the aircraft."""
    roll_moment_side_slip_aircraft(
        XML_FILE, cl_beta_low_speed_=-0.10960731, cl_beta_cruise_=-0.1028198
    )


def test_cl_p_wing():
    """Test cl p of the wing."""
    roll_moment_roll_rate_wing(
        XML_FILE, cl_p_wing_low_speed_=-0.51408026, cl_p_wing_cruise_=-0.5175
    )


def test_cl_p_ht():
    """Test cl p of the ht."""
    roll_moment_roll_rate_ht(XML_FILE, cl_p_ht_low_speed_=-0.00944413, cl_p_ht_cruise_=-0.00948632)


def test_cl_p_vt():
    """Test cl p of the vt."""
    roll_moment_roll_rate_vt(XML_FILE, cl_p_vt_low_speed_=-0.01552891, cl_p_vt_cruise_=-0.01582293)


def test_cl_p():
    """Test cl p of the aircraft."""
    roll_moment_roll_rate_aircraft(XML_FILE, cl_p_low_speed_=-0.5390533, cl_p_cruise_=-0.54255989)


def test_cl_r_wing():
    """Test cl r of the wing."""
    roll_moment_yaw_rate_wing(
        XML_FILE, cl_r_wing_low_speed_=0.11332605, cl_r_wing_cruise_=0.02436741
    )


def test_cl_r_vt():
    """Test cl r of the vt."""
    roll_moment_yaw_rate_vt(XML_FILE, cl_r_vt_low_speed_=0.02940615, cl_r_vt_cruise_=0.0352)


def test_cl_r_aircraft():
    """Test cl r of the aircraft."""
    roll_moment_yaw_rate_aircraft(XML_FILE, cl_r_low_speed_=0.14272755, cl_r_cruise_=0.05957228)


def test_cl_delta_a_aircraft():
    """Test roll authority of the aileron."""
    roll_authority_aileron(XML_FILE, cl_delta_a_low_speed_=0.400, cl_delta_a_cruise_=0.410)


def test_cl_delta_r_aircraft():
    """Test roll authority of the rudder."""
    roll_moment_rudder(XML_FILE, cl_delta_r_low_speed_=0.02543013, cl_delta_r_cruise_=0.03125468)


def test_cm_q_wing():
    """Test cm q of the wing."""
    pitch_moment_pitch_rate_wing(
        XML_FILE, cm_q_wing_low_speed_=-1.20003104, cm_q_wing_cruise_=-1.26957998
    )


def test_cm_q_ht():
    """Test cm q of the ht."""
    pitch_moment_pitch_rate_ht(XML_FILE, cm_q_ht_low_speed_=-12.39066774, cm_q_ht_cruise_=-12.522)


def test_cm_q_aircraft():
    """Test cm q of the aircraft."""
    pitch_moment_pitch_rate_aircraft(XML_FILE, cm_q_low_speed_=-13.587, cm_q_cruise_=-13.7805824)


def test_cm_alpha_dot():
    """Tests cm alpha dot of the aircraft."""
    pitch_moment_aoa_rate_derivative(
        XML_FILE, cm_aoa_dot_low_speed_=-4.6050615, cm_aoa_dot_cruise_=-4.76619507
    )


def test_cn_beta_vt():
    """Tests cn beta of the vt."""
    yaw_moment_sideslip_derivative_vt(
        XML_FILE, cn_beta_vt_low_speed_=0.11432046, cn_beta_vt_cruise_=0.11338037
    )


def test_cn_beta_aircraft():
    """Tests cn beta of the aircraft."""
    yaw_moment_sideslip_aircraft(XML_FILE, cn_beta_low_speed_=0.05910432)


def test_cn_delta_a_aircraft():
    """Test yaw moment of the aileron."""
    yaw_moment_aileron(XML_FILE, cn_delta_a_low_speed_=-0.04211019, cn_delta_a_cruise_=-0.01874913)


def test_cn_delta_r_aircraft():
    """Test yaw moment of the rudder."""
    yaw_moment_rudder(XML_FILE, cn_delta_r_low_speed_=-0.0760609, cn_delta_r_cruise_=-0.07550468)


def test_cn_p_wing():
    """Test cn p of the wing."""
    yaw_moment_roll_rate_wing(
        XML_FILE, cn_p_wing_low_speed_=0.06312776, cn_p_wing_cruise_=0.02753713
    )


def test_cn_p_vt():
    """Test cn p of the vt."""
    yaw_moment_roll_rate_vt(XML_FILE, cn_p_vt_low_speed_=-0.00755252, cn_p_vt_cruise_=-0.00147892)


def test_cn_p_aircraft():
    """Tests cn p of the aircraft."""
    yaw_moment_roll_rate_aircraft(XML_FILE, cn_p_low_speed_=0.05557524, cn_p_cruise_=0.02605821)


def test_cn_r_wing():
    """Test cn r of the wing."""
    yaw_moment_yaw_rate_wing(
        XML_FILE, cn_r_wing_low_speed_=-0.00820353, cn_r_wing_cruise_=-0.00320297
    )


def test_cn_r_vt():
    """Test cn r of the vt."""
    yaw_moment_yaw_rate_vt(XML_FILE, cn_r_vt_low_speed_=-0.08793917, cn_r_vt_cruise_=-0.08504749)


def test_cn_r_aircraft():
    """Tests cn r of the aircraft."""
    yaw_moment_yaw_rate_aircraft(XML_FILE, cn_r_low_speed_=-0.0961427, cn_r_cruise_=-0.08829304)
