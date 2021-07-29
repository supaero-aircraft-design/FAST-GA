"""Test module for aerodynamics groups"""
#  This file is part of FAST : A framework for rapid Overall Aircraft Design
#  Copyright (C) 2020  ONERA & ISAE-SUPAERO
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

import pytest

from platform import system
import numpy as np

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
    high_lift,
    extreme_cl,
    l_d_max,
    cnbeta,
    slipstream_openvsp_cruise,
    slipstream_openvsp_low_speed,
    compute_mach_interpolation_roskam,
    cl_alpha_vt,
    cy_delta_r,
    high_speed_connection,
    low_speed_connection,
    v_n_diagram,
    load_factor,
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
    """Tests drag coefficient @ high speed"""
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
    """Tests drag coefficient @ low speed"""
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
    """Tests polar execution (XFOIL) @ high and low speed"""
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
        cl0_wing=0.0969,
        cl_alpha_wing=5.248,
        cm0=-0.0280,
        coef_k_wing=0.0412,
        cl0_htp=-0.0053,
        cl_alpha_htp=0.5928,
        cl_alpha_htp_isolated=0.8839,
        coef_k_htp=0.3267,
        cl_alpha_vector=np.array([5.238, 5.238, 5.301, 5.384, 5.487, 5.609]),
        mach_vector=np.array([0.0, 0.15, 0.214, 0.274, 0.331, 0.385]),
    )


@pytest.mark.skipif(
    system() != "Windows" or SKIP_STEPS,
    reason="No XFOIL executable available: VLM basic function not computed with "
    "empty result folder (or skipped)",
)
def test_vlm_comp_low_speed():
    """Tests vlm components @ low speed"""
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
            0.09400661,
            0.09382848,
            0.09342488,
            0.0942291,
            0.09584674,
            0.09709421,
            0.09806479,
            0.09878542,
            0.09925102,
            0.0994386,
            0.09919319,
            0.09843692,
            0.09704609,
            0.09459343,
            0.09024482,
            0.08204578,
            0.06402934,
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
        cl0_wing=0.0945,
        cl_alpha_wing=5.119,
        cm0=-0.0273,
        coef_k_wing=0.03950,
        cl0_htp=-0.0051,
        cl_alpha_htp=0.5852,
        cl_alpha_htp_isolated=0.8622,
        coef_k_htp=0.3321,
        y_vector_wing=y_vector_wing,
        cl_vector_wing=cl_vector_wing,
        chord_vector_wing=chord_vector_wing,
        cl_ref_htp=0.0970,
        y_vector_htp=y_vector_htp,
        cl_vector_htp=cl_vector_htp,
    )


@pytest.mark.skipif(
    system() != "Windows" or SKIP_STEPS, reason="OPENVSP is windows dependent platform (or skipped)"
)
def test_openvsp_comp_high_speed():
    """Tests openvsp components @ high speed"""
    comp_high_speed(
        XML_FILE,
        use_openvsp=True,
        cl0_wing=0.1269,
        cl_alpha_wing=5.079,
        cm0=-0.0271,
        coef_k_wing=0.0382,
        cl0_htp=-0.0074,
        cl_alpha_htp=0.5093,
        cl_alpha_htp_isolated=0.9395,
        coef_k_htp=0.6778,
        cl_alpha_vector=np.array([5.50, 5.50, 5.55, 5.61, 5.69, 5.79]),
        mach_vector=np.array([0.0, 0.15, 0.21, 0.27, 0.33, 0.38]),
    )


@pytest.mark.skipif(
    system() != "Windows" or SKIP_STEPS, reason="OPENVSP is windows dependent platform (or skipped)"
)
def test_openvsp_comp_low_speed():
    """Tests openvsp components @ low speed"""
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
            2.03573,
            2.20264,
            2.36966,
            2.53669,
            2.70362,
            2.87035,
            3.03678,
            3.2028,
            3.36831,
            3.53321,
            3.69741,
            3.8608,
            4.02329,
            4.18478,
            4.3452,
            4.50444,
            4.66242,
            4.81906,
            4.97429,
            5.12802,
            5.28018,
            5.4307,
            5.57952,
            5.72657,
        ]
    )
    cl_vector_wing = np.array(
        [
            0.12269875,
            0.12276892,
            0.12268873,
            0.12255841,
            0.12246819,
            0.12229777,
            0.12196697,
            0.1218166,
            0.1223479,
            0.12371122,
            0.12456329,
            0.12536524,
            0.12565595,
            0.12618724,
            0.126909,
            0.12751047,
            0.12801169,
            0.12857305,
            0.12884371,
            0.12921462,
            0.1291244,
            0.12943515,
            0.12956547,
            0.12975593,
            0.12960557,
            0.12959554,
            0.12916449,
            0.12888381,
            0.12829237,
            0.12807183,
            0.12748039,
            0.12691903,
            0.12596671,
            0.1244029,
            0.12146575,
            0.11731564,
            0.11077973,
            0.09798859,
            0.07717792,
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
            0.07706928,
            0.08017691,
            0.08122285,
            0.08215212,
            0.08300295,
            0.08383165,
            0.08411727,
            0.08472874,
            0.08636402,
            0.08635799,
            0.08664361,
            0.08733754,
            0.08604823,
            0.0858853,
            0.08624535,
            0.08599995,
            0.08462214,
            0.08302709,
            0.08063552,
            0.07846721,
            0.07551848,
            0.06980205,
            0.05951367,
            0.04758801,
        ]
    )
    comp_low_speed(
        XML_FILE,
        use_openvsp=True,
        cl0_wing=0.1243,
        cl_alpha_wing=4.981,
        cm0=-0.0265,
        coef_k_wing=0.0381,
        cl0_htp=-0.0071,
        cl_alpha_htp=0.5052,
        cl_alpha_htp_isolated=0.9224,
        coef_k_htp=0.6736,
        y_vector_wing=y_vector_wing,
        cl_vector_wing=cl_vector_wing,
        chord_vector_wing=chord_vector_wing,
        cl_ref_htp=0.081,
        y_vector_htp=y_vector_htp,
        cl_vector_htp=cl_vector_htp,
    )


def test_2d_hinge_moment():
    """Tests tail hinge-moments"""
    hinge_moment_2d(XML_FILE, ch_alpha_2d=-0.3548, ch_delta_2d=-0.5755)


def test_3d_hinge_moment():
    """Tests tail hinge-moments"""
    hinge_moment_3d(XML_FILE, ch_alpha=-0.2594, ch_delta=-0.6216)


def test_high_lift():
    """Tests high-lift contribution"""
    high_lift(
        XML_FILE,
        delta_cl0_landing=0.6167,
        delta_clmax_landing=0.4194,
        delta_cm_landing=-0.1049,
        delta_cd_landing=0.0136,
        delta_cl0_takeoff=0.2300,
        delta_clmax_takeoff=0.0858,
        delta_cm_takeoff=-0.0391,
        delta_cd_takeoff=0.0011,
        cl_delta_elev=0.4686,
        cd_delta_elev=0.06226,
    )


@pytest.mark.skipif(
    system() != "Windows",
    reason="No XFOIL executable available: not computed with empty result folder",
)
def test_extreme_cl():
    """Tests maximum/minimum cl component with default result cl=f(y) curve"""
    extreme_cl(
        XML_FILE,
        cl_max_clean_wing=1.56,
        cl_min_clean_wing=-1.25,
        cl_max_takeoff_wing=1.65,
        cl_max_landing_wing=1.98,
        cl_max_clean_htp=0.27,
        cl_min_clean_htp=-0.27,
        alpha_max_clean_htp=30.37,
        alpha_min_clean_htp=-30.35,
    )


def test_l_d_max():
    """Tests best lift/drag component"""
    l_d_max(XML_FILE, l_d_max_=15.86, optimal_cl=0.8166, optimal_cd=0.0514, optimal_alpha=7.74)


def test_cnbeta():
    """Tests cn beta fuselage"""
    cnbeta(XML_FILE, cn_beta_fus=-0.0684)


@pytest.mark.skipif(
    system() != "Windows" or SKIP_STEPS, reason="OPENVSP is windows dependent platform (or skipped)"
)
def test_slipstream_openvsp_cruise():
    """Compute slipstream @ high speed!"""
    y_vector_prop_on = np.array(
        [
            0.05,
            0.14,
            0.23,
            0.32,
            0.41,
            0.5,
            0.59,
            0.72,
            0.88,
            1.04,
            1.21,
            1.37,
            1.54,
            1.7,
            1.87,
            2.04,
            2.2,
            2.37,
            2.54,
            2.7,
            2.87,
            3.04,
            3.2,
            3.37,
            3.53,
            3.7,
            3.86,
            4.02,
            4.18,
            4.35,
            4.5,
            4.66,
            4.82,
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
            1.53,
            1.53,
            1.53,
            1.52,
            1.52,
            1.52,
            1.52,
            1.51,
            1.53,
            1.55,
            1.57,
            1.58,
            1.59,
            1.6,
            1.61,
            1.62,
            1.63,
            1.64,
            1.64,
            1.65,
            1.65,
            1.65,
            1.65,
            1.66,
            1.65,
            1.65,
            1.65,
            1.65,
            1.63,
            1.63,
            1.62,
            1.61,
            1.58,
            1.55,
            1.49,
            1.43,
            1.31,
            1.14,
            0.94,
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
        ct=0.9231,
        delta_cl=-0.0002,
    )


@pytest.mark.skipif(
    system() != "Windows" or SKIP_STEPS, reason="OPENVSP is windows dependent platform (or skipped)"
)
def test_slipstream_openvsp_low_speed():
    """Compute slipstream @ low speed!"""
    y_vector_prop_on = np.array(
        [
            0.05,
            0.14,
            0.23,
            0.32,
            0.41,
            0.5,
            0.59,
            0.72,
            0.88,
            1.04,
            1.21,
            1.37,
            1.54,
            1.7,
            1.87,
            2.04,
            2.2,
            2.37,
            2.54,
            2.7,
            2.87,
            3.04,
            3.2,
            3.37,
            3.53,
            3.7,
            3.86,
            4.02,
            4.18,
            4.35,
            4.5,
            4.66,
            4.82,
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
            1.53,
            1.53,
            1.53,
            1.52,
            1.52,
            1.52,
            1.51,
            1.51,
            1.53,
            1.55,
            1.57,
            1.58,
            1.58,
            1.6,
            1.61,
            1.62,
            1.63,
            1.64,
            1.64,
            1.65,
            1.65,
            1.65,
            1.65,
            1.66,
            1.65,
            1.65,
            1.65,
            1.65,
            1.63,
            1.63,
            1.62,
            1.6,
            1.58,
            1.55,
            1.5,
            1.43,
            1.32,
            1.14,
            0.95,
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
        ct=0.7256,
        delta_cl=-0.0007,
    )


def test_compute_mach_interpolation_roskam():
    """Tests computation of the mach interpolation vector using Roskam's approach"""
    compute_mach_interpolation_roskam(
        XML_FILE,
        cl_alpha_vector=np.array([5.48, 5.51, 5.58, 5.72, 5.91, 6.18]),
        mach_vector=np.array([0.0, 0.07, 0.15, 0.23, 0.30, 0.38]),
    )


def test_cl_alpha_vt():
    """Tests Cl alpha vt"""
    cl_alpha_vt(XML_FILE, cl_alpha_vt_ls=3.0253, k_ar_effective=1.2612, cl_alpha_vt_cruise=3.0889)


def test_cy_delta_r():
    """Tests cy delta of the rudder"""
    cy_delta_r(XML_FILE, cy_delta_r_=1.8858)


@pytest.mark.skipif(
    system() != "Windows" or SKIP_STEPS, reason="OPENVSP is windows dependent platform (or skipped)"
)
def test_high_speed_connection_openvsp():
    """Tests high speed components connection"""
    high_speed_connection(XML_FILE, ENGINE_WRAPPER, use_openvsp=True)


@pytest.mark.skipif(SKIP_STEPS, reason="Skip test because already performed on Cirrus")
def test_high_speed_connection_vlm():
    """Tests high speed components connection"""
    high_speed_connection(XML_FILE, ENGINE_WRAPPER, use_openvsp=False)


@pytest.mark.skipif(
    system() != "Windows" or SKIP_STEPS, reason="OPENVSP is windows dependent platform (or skipped)"
)
def test_low_speed_connection_openvsp():
    """Tests low speed components connection"""
    low_speed_connection(XML_FILE, ENGINE_WRAPPER, use_openvsp=True)


@pytest.mark.skipif(SKIP_STEPS, reason="Skip test because already performed on Cirrus")
def test_low_speed_connection_vlm():
    """Tests low speed components connection"""
    low_speed_connection(XML_FILE, ENGINE_WRAPPER, use_openvsp=False)


@pytest.mark.skipif(SKIP_STEPS, reason="Skip test because already performed on Cirrus")
def test_v_n_diagram():
    # load all inputs
    velocity_vect = np.array(
        [
            35.193,
            39.241,
            68.604,
            48.379,
            0.0,
            0.0,
            83.959,
            83.959,
            83.959,
            117.287,
            117.287,
            117.287,
            117.287,
            105.559,
            83.959,
            0.0,
            31.262,
            44.211,
            56.272,
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
            3.948,
            -1.948,
            3.8,
            0.0,
            3.153,
            -1.153,
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
    load_factor(XML_FILE, ENGINE_WRAPPER, load_factor_ultimate=5.9, vh=88.71)
