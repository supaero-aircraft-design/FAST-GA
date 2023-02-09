"""Test module for aerodynamics groups"""
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

from .dummy_engines import ENGINE_WRAPPER_SR22 as ENGINE_WRAPPER
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
    elevator,
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
    equilibrated_cl_cd_polar,
    non_equilibrated_cl_cd_polar,
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

XML_FILE = "cirrus_sr22.xml"
SKIP_STEPS = True  # avoid some tests to accelerate validation process (intermediary VLM/OpenVSP)


def test_compute_reynolds():
    """Tests high and low speed reynolds calculation"""
    compute_reynolds(
        XML_FILE,
        mach_high_speed=0.2488,
        reynolds_high_speed=4629639,
        mach_low_speed=0.1194,
        reynolds_low_speed=2782216,
    )


def test_cd0_high_speed():
    """Tests drag coefficient @ high speed."""
    cd0_high_speed(
        XML_FILE,
        ENGINE_WRAPPER,
        cd0_wing=0.00525,
        cd0_fus=0.00579,
        cd0_ht=0.00142,
        cd0_vt=0.00072,
        cd0_nac=0.00000,
        cd0_lg=0.00293,
        cd0_other=0.00446,
        cd0_total=0.02573,
    )


def test_cd0_low_speed():
    """Tests drag coefficient @ low speed."""
    cd0_low_speed(
        XML_FILE,
        ENGINE_WRAPPER,
        cd0_wing=0.00570,
        cd0_fus=0.00639,
        cd0_ht=0.00154,
        cd0_vt=0.00079,
        cd0_nac=0.00000,
        cd0_lg=0.00293,
        cd0_other=0.00446,
        cd0_total=0.02729,
    )


@pytest.mark.skipif(
    system() != "Windows" and xfoil_path is None or SKIP_STEPS,
    reason="No XFOIL executable available (or skipped)",
)
def test_polar():
    """Tests polar execution (XFOIL) @ high and low speed."""
    polar(
        XML_FILE,
        mach_high_speed=0.245,
        reynolds_high_speed=4571770 * 1.549,
        mach_low_speed=0.1179,
        reynolds_low_speed=2746999 * 1.549,
        cdp_1_high_speed=0.00464,
        cl_max_2d=1.6965,
        cdp_1_low_speed=0.00488,
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
    """Tests polar execution (XFOIL) @ low speed!"""
    airfoil_slope_xfoil(
        XML_FILE,
        wing_airfoil_file="roncz.af",
        htp_airfoil_file="naca0012.af",
        vtp_airfoil_file="naca0012.af",
        cl_alpha_wing=6.5755,
        cl_alpha_htp=6.2703,
        cl_alpha_vtp=6.2703,
    )


def test_airfoil_slope_wt_xfoil():
    """Tests polar reading @ low speed!"""
    airfoil_slope_wt_xfoil(
        XML_FILE,
        wing_airfoil_file="roncz.af",
        htp_airfoil_file="naca0012.af",
        vtp_airfoil_file="naca0012.af",
    )


@pytest.mark.skipif(
    system() != "Windows" or SKIP_STEPS,
    reason="No XFOIL executable available: VLM basic function not computed with "
    "empty result folder (or skipped)",
)
def test_vlm_comp_high_speed():
    """Tests vlm components @ high speed!"""
    comp_high_speed(
        XML_FILE,
        use_openvsp=False,
        cl0_wing=0.09622569,
        cl_ref_wing=1.01291563,
        cl_alpha_wing=5.2123498,
        cm0=-0.02801145,
        coeff_k_wing=0.04126501,
        cl0_htp=-0.00537875,
        cl_alpha_htp=0.59284874,
        cl_alpha_htp_isolated=0.88397051,
        coeff_k_htp=0.32674171,
        cl_alpha_vector=np.array(
            [5.69251579, 5.69251579, 5.7581548, 5.84566912, 5.9524415, 6.07945968]
        ),
        mach_vector=np.array([0.0, 0.15, 0.21432613, 0.27476331, 0.33175508, 0.38567317]),
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
            0.106,
            0.318,
            0.53,
            0.83149394,
            1.22248183,
            1.61346972,
            2.00445761,
            2.3954455,
            2.78643339,
            3.17742128,
            3.54627778,
            3.89300289,
            4.239728,
            4.58645311,
            4.93317822,
            5.27990333,
            5.62662844,
        ]
    )
    cl_vector_wing = np.array(
        [
            0.97866667,
            0.97680367,
            0.97267907,
            0.98176565,
            1.00028125,
            1.01487291,
            1.02633147,
            1.03490258,
            1.04053245,
            1.0429698,
            1.0405289,
            1.03236792,
            1.01707211,
            0.98988675,
            0.94154398,
            0.85075402,
            0.65555692,
        ]
    )
    chord_vector_wing = np.array(
        [
            1.47443128,
            1.47443128,
            1.47443128,
            1.44652241,
            1.39070465,
            1.33488689,
            1.27906914,
            1.22325138,
            1.16743363,
            1.11161587,
            1.05895761,
            1.00945885,
            0.95996008,
            0.91046132,
            0.86096255,
            0.81146379,
            0.76196502,
        ]
    )
    y_vector_htp = np.array(
        [
            0.05627362,
            0.16882086,
            0.2813681,
            0.39391534,
            0.50646259,
            0.61900983,
            0.73155707,
            0.84410431,
            0.95665155,
            1.06919879,
            1.18174603,
            1.29429327,
            1.40684051,
            1.51938776,
            1.631935,
            1.74448224,
            1.85702948,
        ]
    )
    cl_vector_htp = np.array(
        [
            0.09902212,
            0.10071269,
            0.10209038,
            0.10319359,
            0.10402982,
            0.1045907,
            0.10485421,
            0.10478287,
            0.10431916,
            0.10337737,
            0.10183014,
            0.09948513,
            0.09604307,
            0.09101606,
            0.08354914,
            0.07195107,
            0.05193693,
        ]
    )
    comp_low_speed(
        XML_FILE,
        use_openvsp=False,
        cl0_wing=0.09386187,
        cl_ref_wing=0.98803303,
        cl_alpha_wing=5.08430673,
        cm0=-0.02732334,
        coeff_k_wing=0.03950543,
        cl0_htp=-0.00511773,
        cl_alpha_htp=0.58526104,
        cl_alpha_htp_isolated=0.86225549,
        coeff_k_htp=0.33214048,
        y_vector_wing=y_vector_wing,
        cl_vector_wing=cl_vector_wing,
        chord_vector_wing=chord_vector_wing,
        cl_ref_htp=0.09702959,
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
        cl0_wing=0.12700925,
        cl_ref_wing=1.01371823,
        cl_alpha_wing=5.08046827,
        cm0=-0.0271461,
        coeff_k_wing=0.03820337,
        cl0_htp=-0.00758,
        cl_alpha_htp=0.50460393,
        cl_alpha_htp_isolated=0.939527,
        coeff_k_htp=0.72237382,
        cl_alpha_vector=np.array(
            [5.50001843, 5.50001843, 5.54969521, 5.61545285, 5.69522352, 5.78895035]
        ),
        mach_vector=np.array([0.0, 0.15, 0.21432613, 0.27476331, 0.33175508, 0.38567317]),
    )


@pytest.mark.skipif(
    system() != "Windows" or SKIP_STEPS, reason="OPENVSP is windows dependent platform (or skipped)"
)
def test_openvsp_comp_low_speed():
    """Tests openvsp components @ low speed."""
    y_vector_wing = np.array(
        [
            0.04543,
            0.13629,
            0.22714,
            0.318,
            0.40886,
            0.49971,
            0.59057,
            0.71673,
            0.87909,
            1.04236,
            1.20645,
            1.37126,
            1.5367,
            1.70265,
            1.86903,
            2.03572,
            2.20263,
            2.36966,
            2.53668,
            2.70362,
            2.87035,
            3.03677,
            3.20279,
            3.36831,
            3.53321,
            3.6974,
            3.86079,
            4.02328,
            4.18478,
            4.34519,
            4.50443,
            4.66242,
            4.81906,
            4.97428,
            5.12801,
            5.28017,
            5.4307,
            5.57952,
            5.72657,
        ]
    )
    cl_vector_wing = np.array(
        [
            0.97349029,
            0.97262819,
            0.97128492,
            0.97054311,
            0.96837784,
            0.96608225,
            0.96469888,
            0.96484925,
            0.97138516,
            0.98527899,
            0.99559411,
            1.00442561,
            1.00683147,
            1.01352777,
            1.02182798,
            1.0287348,
            1.03302524,
            1.03903989,
            1.04196702,
            1.04591663,
            1.04402202,
            1.04731003,
            1.04826234,
            1.05082859,
            1.04789144,
            1.04881369,
            1.04528509,
            1.04364109,
            1.03628318,
            1.03372695,
            1.02551696,
            1.01908129,
            1.00401461,
            0.98681272,
            0.95390259,
            0.91076757,
            0.84101774,
            0.72374219,
            0.56542672,
        ]
    )
    chord_vector_wing = np.array(
        [
            1.46662,
            1.46662,
            1.46662,
            1.46662,
            1.46662,
            1.46662,
            1.46662,
            1.45512,
            1.43207,
            1.40888,
            1.38558,
            1.36217,
            1.33868,
            1.31511,
            1.29149,
            1.26781,
            1.24411,
            1.22039,
            1.19667,
            1.17297,
            1.14929,
            1.12566,
            1.10208,
            1.07858,
            1.05516,
            1.03184,
            1.00864,
            0.98556,
            0.96263,
            0.93985,
            0.91724,
            0.8948,
            0.87256,
            0.85052,
            0.82869,
            0.80708,
            0.7857,
            0.76457,
            0.74369,
        ]
    )
    y_vector_htp = np.array(
        [
            0.03975,
            0.11947,
            0.19919,
            0.27891,
            0.35863,
            0.43835,
            0.51807,
            0.59779,
            0.6775,
            0.75722,
            0.83694,
            0.91666,
            0.99638,
            1.0761,
            1.15581,
            1.23553,
            1.31525,
            1.39497,
            1.47469,
            1.5544,
            1.63412,
            1.71384,
            1.79355,
            1.87327,
        ]
    )
    cl_vector_htp = np.array(
        [
            0.07634316,
            0.07912092,
            0.08011255,
            0.08114038,
            0.08193087,
            0.08268112,
            0.08296674,
            0.08346356,
            0.08524366,
            0.08524165,
            0.08557957,
            0.08625943,
            0.08471667,
            0.08438479,
            0.08455777,
            0.08458995,
            0.08283801,
            0.08142198,
            0.07992549,
            0.0780589,
            0.07492713,
            0.0693535,
            0.06007285,
            0.04775093,
        ]
    )
    comp_low_speed(
        XML_FILE,
        use_openvsp=True,
        cl0_wing=0.1243628,
        cl_ref_wing=0.99389998,
        cl_alpha_wing=4.98208104,
        cm0=-0.02652458,
        coeff_k_wing=0.03816438,
        cl0_htp=-0.00734,
        cl_alpha_htp=0.50042134,
        cl_alpha_htp_isolated=0.92249372,
        coeff_k_htp=0.715625,
        y_vector_wing=y_vector_wing,
        cl_vector_wing=cl_vector_wing,
        chord_vector_wing=chord_vector_wing,
        cl_ref_htp=0.08,
        y_vector_htp=y_vector_htp,
        cl_vector_htp=cl_vector_htp,
    )


def test_2d_hinge_moment():
    """Tests tail hinge-moments."""
    hinge_moment_2d(XML_FILE, ch_alpha_2d=-0.3548, ch_delta_2d=-0.5755)


def test_3d_hinge_moment():
    """Tests tail hinge-moments."""
    hinge_moment_3d(XML_FILE, ch_alpha=-0.2594, ch_delta=-0.6216)


def test_all_hinge_moment():
    """Tests tail hinge-moments full computation."""
    hinge_moments(XML_FILE, ch_alpha=-0.2594, ch_delta=-0.6216)


def test_elevator():
    """Tests elevator contribution."""
    elevator(
        XML_FILE,
        cl_delta_elev=0.4686,
        cd_delta_elev=0.06226,
    )


def test_high_lift():
    """Tests high-lift contribution."""
    high_lift(
        XML_FILE,
        delta_cl0_landing=0.6167,
        delta_cl0_landing_2d=1.5094,
        delta_clmax_landing=0.4194,
        delta_cm_landing=-0.1729,
        delta_cm_landing_2d=-0.2830,
        delta_cd_landing=0.0136,
        delta_cd_landing_2d=0.0258,
        delta_cl0_takeoff=0.2300,
        delta_cl0_takeoff_2d=0.5631,
        delta_clmax_takeoff=0.0858,
        delta_cm_takeoff=-0.0645,
        delta_cm_takeoff_2d=-0.1055,
        delta_cd_takeoff=0.0011,
        delta_cd_takeoff_2d=0.0020,
    )


@pytest.mark.skipif(
    system() != "Windows",
    reason="No XFOIL executable available: not computed with empty result folder",
)
def test_extreme_cl_wing_clean():
    """Tests maximum/minimum cl component with default result cl=f(y) curve."""
    wing_extreme_cl_clean(
        XML_FILE,
        cl_max_clean_wing=1.58443803,
        cl_min_clean_wing=-1.26,
    )


@pytest.mark.skipif(
    system() != "Windows",
    reason="No XFOIL executable available: not computed with empty result folder",
)
def test_extreme_cl_htp_clean():
    """Tests maximum/minimum cl component with default result cl=f(y) curve."""
    htp_extreme_cl_clean(
        XML_FILE,
        cl_max_clean_htp=0.27,
        cl_min_clean_htp=-0.27,
        alpha_max_clean_htp=30.37,
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
        cl_max_takeoff_wing=1.65,
        cl_max_landing_wing=1.98,
    )


def test_l_d_max():
    """Tests best lift/drag component."""
    l_d_max(XML_FILE, l_d_max_=15.86, optimal_cl=0.8166, optimal_cd=0.0514, optimal_alpha=7.74)


def test_cnbeta():
    """Tests cn beta fuselage."""
    cnbeta(XML_FILE, cn_beta_fus=-0.0685)


@pytest.mark.skipif(
    system() != "Windows" or SKIP_STEPS, reason="OPENVSP is windows dependent platform (or skipped)"
)
def test_slipstream_openvsp_cruise():
    """Compute slipstream @ high speed!"""
    y_vector_prop_on = np.array(
        [
            0.04543,
            0.13629,
            0.22714,
            0.318,
            0.40886,
            0.49971,
            0.59057,
            0.71673,
            0.87909,
            1.04236,
            1.20645,
            1.37126,
            1.5367,
            1.70265,
            1.86903,
            2.03572,
            2.20263,
            2.36966,
            2.53668,
            2.70362,
            2.87035,
            3.03677,
            3.20279,
            3.36831,
            3.53321,
            3.6974,
            3.86079,
            4.02328,
            4.18478,
            4.34519,
            4.50443,
            4.66242,
            4.81906,
            4.97428,
            5.12801,
            5.28017,
            5.4307,
            5.57952,
            5.72657,
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
            1.52808,
            1.52873,
            1.52644,
            1.52489,
            1.52235,
            1.51762,
            1.51547,
            1.51508,
            1.52729,
            1.54965,
            1.56709,
            1.58151,
            1.58511,
            1.59648,
            1.61039,
            1.62211,
            1.62841,
            1.63834,
            1.64318,
            1.64979,
            1.6467,
            1.65205,
            1.65331,
            1.65782,
            1.65249,
            1.65446,
            1.64928,
            1.6474,
            1.6343,
            1.63038,
            1.61605,
            1.60545,
            1.57819,
            1.54977,
            1.49446,
            1.42498,
            1.31435,
            1.13807,
            0.93855,
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
        ct=0.06007,
        delta_cl=-0.00041,
    )


@pytest.mark.skipif(
    system() != "Windows" or SKIP_STEPS, reason="OPENVSP is windows dependent platform (or skipped)"
)
def test_slipstream_openvsp_low_speed():
    """Compute slipstream @ low speed!"""
    y_vector_prop_on = np.array(
        [
            0.04543,
            0.13629,
            0.22714,
            0.318,
            0.40886,
            0.49971,
            0.59057,
            0.71673,
            0.87909,
            1.04236,
            1.20645,
            1.37126,
            1.5367,
            1.70265,
            1.86903,
            2.03572,
            2.20263,
            2.36966,
            2.53668,
            2.70362,
            2.87035,
            3.03677,
            3.20279,
            3.36831,
            3.53321,
            3.6974,
            3.86079,
            4.02328,
            4.18478,
            4.34519,
            4.50443,
            4.66242,
            4.81906,
            4.97428,
            5.12801,
            5.28017,
            5.4307,
            5.57952,
            5.72657,
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
            1.52569,
            1.52758,
            1.52533,
            1.52383,
            1.52135,
            1.51655,
            1.51435,
            1.51393,
            1.52619,
            1.54829,
            1.56568,
            1.58007,
            1.58369,
            1.59511,
            1.60896,
            1.62066,
            1.62693,
            1.63688,
            1.64175,
            1.6484,
            1.64535,
            1.65072,
            1.652,
            1.65653,
            1.65125,
            1.65321,
            1.64806,
            1.64616,
            1.633,
            1.62921,
            1.61509,
            1.60485,
            1.57793,
            1.55026,
            1.49586,
            1.42745,
            1.31838,
            1.14331,
            0.94757,
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
        ct=0.04721,
        delta_cl=-0.00116,
    )


def test_compute_mach_interpolation_roskam():
    """Tests computation of the mach interpolation vector using Roskam's approach."""
    compute_mach_interpolation_roskam(
        XML_FILE,
        cl_alpha_vector=np.array([5.48, 5.51, 5.58, 5.72, 5.91, 6.18]),
        mach_vector=np.array([0.0, 0.07, 0.15, 0.23, 0.30, 0.38]),
    )


def test_non_equilibrated_cl_cd_polar():
    """Tests computation of the non equilibrated cl/cd polar computation."""
    non_equilibrated_cl_cd_polar(
        XML_FILE,
        cl_polar_ls_=np.array(
            [0.12, 0.21, 0.3, 0.39, 0.48, 0.56, 0.65, 0.74, 0.83, 0.92, 1.0, 1.09, 1.18, 1.27, 1.35]
        ),
        cd_polar_ls_=np.array(
            [
                0.0279,
                0.029,
                0.0308,
                0.0331,
                0.036,
                0.0396,
                0.0437,
                0.0484,
                0.0537,
                0.0596,
                0.0661,
                0.0732,
                0.0809,
                0.0892,
                0.098,
            ]
        ),
        cl_polar_cruise_=np.array(
            [0.13, 0.22, 0.31, 0.4, 0.49, 0.58, 0.67, 0.75, 0.84, 0.93, 1.02, 1.11, 1.2, 1.29, 1.38]
        ),
        cd_polar_cruise_=np.array(
            [
                0.0264,
                0.0276,
                0.0294,
                0.0318,
                0.0348,
                0.0385,
                0.0428,
                0.0477,
                0.0532,
                0.0594,
                0.0662,
                0.0735,
                0.0815,
                0.0902,
                0.0994,
            ]
        ),
    )


def test_equilibrated_cl_cd_polar():
    """Tests computation of the non equilibrated cl/cd polar computation."""
    equilibrated_cl_cd_polar(
        XML_FILE,
        cl_polar_ls_=np.array(
            [0.12, 0.23, 0.34, 0.45, 0.57, 0.68, 0.79, 0.9, 1.01, 1.12, 1.23, 1.33, 1.44]
        ),
        cd_polar_ls_=np.array(
            [
                0.028,
                0.0298,
                0.0329,
                0.0372,
                0.0426,
                0.0493,
                0.0572,
                0.0662,
                0.0764,
                0.0878,
                0.1004,
                0.1142,
                0.1291,
            ]
        ),
        cl_polar_cruise_=np.array(
            [0.04, 0.15, 0.27, 0.39, 0.51, 0.62, 0.74, 0.86, 0.97, 1.09, 1.2, 1.31, 1.43]
        ),
        cd_polar_cruise_=np.array(
            [
                0.0258,
                0.0269,
                0.0293,
                0.033,
                0.0381,
                0.0444,
                0.0521,
                0.061,
                0.0713,
                0.0828,
                0.0956,
                0.1097,
                0.1251,
            ]
        ),
    )


def test_cl_alpha_vt():
    """Tests Cl alpha vt."""
    cl_alpha_vt(XML_FILE, cl_alpha_vt_ls=3.0253, k_ar_effective=1.2612, cl_alpha_vt_cruise=3.0889)


def test_cy_delta_r():
    """Tests cy delta of the rudder."""
    cy_delta_r(XML_FILE, cy_delta_r_=1.8858, cy_delta_r_cruise=1.9255)


def test_effective_efficiency():
    """Tests cy delta of the rudder."""
    effective_efficiency(
        XML_FILE, effective_efficiency_low_speed=0.9662, effective_efficiency_cruise=0.9759
    )


def test_cm_alpha_fus():
    """Tests cy delta of the rudder."""
    cm_alpha_fus(XML_FILE, cm_alpha_fus_=-0.2895)


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
            35.19288485,
            39.24079728,
            68.6035622,
            48.37930405,
            0.0,
            0.0,
            79.8413954,
            79.8413954,
            79.8413954,
            111.53546056,
            111.53546056,
            111.53546056,
            111.53546056,
            100.38191451,
            83.95891628,
            0.0,
            31.26200054,
            44.21114515,
            56.27160098,
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
            -1.79136195,
            3.8,
            0.0,
            3.02923506,
            -1.02923506,
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
        vh=88.71,
        va=68.603,
        vc=79.841,
        vd=111.535,
    )


def test_cy_beta_fus():
    """Tests cy beta of the fuselage."""
    cy_beta_fus(XML_FILE, cy_beta_fus_=-0.2949)


def test_downwash_gradient():
    """Tests cy beta of the fuselage."""
    downwash_gradient(XML_FILE, downwash_gradient_ls_=0.3145, downwash_gradient_cruise_=0.3207)


def test_cl_alpha_dot():
    """Tests cl alpha dot of the aircraft."""
    lift_aoa_rate_derivative(XML_FILE, cl_aoa_dot_low_speed_=0.998, cl_aoa_dot_cruise_=1.026)


def test_cl_q_ht():
    """Tests cl q of the tail."""
    lift_pitch_velocity_derivative_ht(XML_FILE, cl_q_ht_low_speed_=3.174, cl_q_ht_cruise_=3.199)


def test_cl_q_wing():
    """Tests cl q of the wing."""
    lift_pitch_velocity_derivative_wing(
        XML_FILE, cl_q_wing_low_speed_=2.515, cl_q_wing_cruise_=2.618
    )


def test_cl_q_aircraft():
    """Tests cl q of the aircraft."""
    lift_pitch_velocity_derivative_aircraft(XML_FILE, cl_q_low_speed_=5.689, cl_q_cruise_=5.818)


def test_cy_beta_wing():
    """Tests cy beta of the wing."""
    side_force_sideslip_derivative_wing(XML_FILE, cy_beta_wing_=-0.02865)


def test_cy_beta_vt():
    """Tests cy beta of the vertical tail."""
    side_force_sideslip_derivative_vt(
        XML_FILE, cy_beta_vt_low_speed_=-0.3445, cy_beta_vt_cruise_=-0.3518
    )


def test_cy_beta_aircraft():
    """Tests cy beta of the aircraft."""
    side_force_sideslip_aircraft(XML_FILE, cy_beta_low_speed_=-0.6681)


def test_cy_r_aircraft():
    """Tests cy beta of the aircraft."""
    side_force_yaw_rate_aircraft(
        XML_FILE, cy_yaw_rate_low_speed_=0.2469, cy_yaw_rate_cruise_=0.24468695
    )


def test_cy_p_aircraft():
    """Tests cy beta of the aircraft."""
    side_force_roll_rate_aircraft(
        XML_FILE, cy_roll_rate_low_speed_=-0.09683, cy_roll_rate_cruise_=-0.11623686
    )


def test_cl_beta_wing():
    """Test cl beta of the wing."""
    roll_moment_side_slip_wing(
        XML_FILE, cl_beta_wing_low_speed_=-0.01422404, cl_beta_wing_cruise_=-0.01499659
    )


def test_cl_beta_ht():
    """Test cl beta of the wing."""
    roll_moment_side_slip_ht(
        XML_FILE, cl_beta_ht_low_speed_=-0.00079034, cl_beta_ht_cruise_=-5.08832572e-05
    )


def test_cl_beta_vt():
    """Test cl beta of the vt."""
    roll_moment_side_slip_vt(
        XML_FILE, cl_beta_vt_low_speed_=-0.04841665, cl_beta_vt_cruise_=-0.05811843
    )


def test_cl_beta_aircraft():
    """Test cl beta of the vt."""
    roll_moment_side_slip_aircraft(
        XML_FILE, cl_beta_low_speed_=-0.06343, cl_beta_cruise_=-0.0731659
    )


def test_cl_p_wing():
    """Test cl p of the wing."""
    roll_moment_roll_rate_wing(
        XML_FILE, cl_p_wing_low_speed_=-0.52850071, cl_p_wing_cruise_=-0.53237904
    )


def test_cl_p_ht():
    """Test cl p of the ht."""
    roll_moment_roll_rate_ht(XML_FILE, cl_p_ht_low_speed_=-0.00859419, cl_p_ht_cruise_=-0.0086555)


def test_cl_p_vt():
    """Test cl p of the vt."""
    roll_moment_roll_rate_vt(XML_FILE, cl_p_vt_low_speed_=-0.02020534, cl_p_vt_cruise_=-0.02063349)


def test_cl_p():
    """Test cl p of the aircraft."""
    roll_moment_roll_rate_aircraft(XML_FILE, cl_p_low_speed_=-0.55730024, cl_p_cruise_=-0.5616597)


def test_cl_r_wing():
    """Test cl r of the wing."""
    roll_moment_yaw_rate_wing(
        XML_FILE, cl_r_wing_low_speed_=0.13962554, cl_r_wing_cruise_=0.05445163
    )


def test_cl_r_vt():
    """Test cl r of the vt."""
    roll_moment_yaw_rate_vt(XML_FILE, cl_r_vt_low_speed_=0.03470904, cl_r_vt_cruise_=0.04042303)


def test_cl_r_aircraft():
    """Test cl r of the aircraft."""
    roll_moment_yaw_rate_aircraft(XML_FILE, cl_r_low_speed_=0.17433458, cl_r_cruise_=0.09487465)


def test_cl_delta_a_aircraft():
    """Test roll authority of the aileron."""
    roll_authority_aileron(XML_FILE, cl_delta_a_low_speed_=0.222, cl_delta_a_cruise_=0.2277)


def test_cl_delta_r_aircraft():
    """Test roll authority of the rudder."""
    roll_moment_rudder(XML_FILE, cl_delta_r_low_speed_=0.02815446, cl_delta_r_cruise_=0.03379028)


def test_cm_q_wing():
    """Test cm q of the wing."""
    pitch_moment_pitch_rate_wing(
        XML_FILE, cm_q_wing_low_speed_=-1.31910395, cm_q_wing_cruise_=-1.38898117
    )


def test_cm_q_ht():
    """Test cm q of the ht."""
    pitch_moment_pitch_rate_ht(XML_FILE, cm_q_ht_low_speed_=-10.643, cm_q_ht_cruise_=-10.726)


def test_cm_q_aircraft():
    """Test cm q of the aircraft."""
    pitch_moment_pitch_rate_aircraft(XML_FILE, cm_q_low_speed_=-11.962, cm_q_cruise_=-12.115)


def test_cm_alpha_dot():
    """Tests cm alpha dot of the aircraft."""
    pitch_moment_aoa_rate_derivative(
        XML_FILE, cm_aoa_dot_low_speed_=-3.347, cm_aoa_dot_cruise_=-3.440
    )


def test_cn_beta_vt():
    """Tests cn beta of the vt."""
    yaw_moment_sideslip_derivative_vt(
        XML_FILE, cn_beta_vt_low_speed_=0.12348297, cn_beta_vt_cruise_=0.12234347
    )


def test_cn_beta_aircraft():
    """Tests cn beta of the aircraft."""
    yaw_moment_sideslip_aircraft(XML_FILE, cn_beta_low_speed_=0.05499728)


def test_cn_delta_a_aircraft():
    """Test yaw moment of the aileron."""
    yaw_moment_aileron(XML_FILE, cn_delta_a_low_speed_=-0.01690296, cn_delta_a_cruise_=-0.0066846)


def test_cn_delta_r_aircraft():
    """Test yaw moment of the rudder."""
    yaw_moment_rudder(XML_FILE, cn_delta_r_low_speed_=-0.07180581, cn_delta_r_cruise_=-0.07113097)


def test_cn_p_wing():
    """Test cn p of the wing."""
    yaw_moment_roll_rate_wing(
        XML_FILE, cn_p_wing_low_speed_=0.0668911, cn_p_wing_cruise_=0.02562311
    )


def test_cn_p_vt():
    """Test cn p of the vt."""
    yaw_moment_roll_rate_vt(XML_FILE, cn_p_vt_low_speed_=-0.00758319, cn_p_vt_cruise_=-0.00147893)


def test_cn_p_aircraft():
    """Tests cn p of the aircraft."""
    yaw_moment_roll_rate_aircraft(XML_FILE, cn_p_low_speed_=0.05930792, cn_p_cruise_=0.02414418)


def test_cn_r_wing():
    """Test cn r of the wing."""
    yaw_moment_yaw_rate_wing(
        XML_FILE, cn_r_wing_low_speed_=-0.00527219, cn_r_wing_cruise_=-0.00218132
    )


def test_cn_r_vt():
    """Test cn r of the vt."""
    yaw_moment_yaw_rate_vt(XML_FILE, cn_r_vt_low_speed_=-0.08852275, cn_r_vt_cruise_=-0.08509338)


def test_cn_r_aircraft():
    """Tests cn r of the aircraft."""
    yaw_moment_yaw_rate_aircraft(XML_FILE, cn_r_low_speed_=-0.09379494, cn_r_cruise_=-0.0872747)
