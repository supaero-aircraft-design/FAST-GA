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
        cl0_wing=0.10589994,
        cl_alpha_wing=5.71596997,
        cm0=-0.03776567,
        coeff_k_wing=0.06576841,
        cl0_htp=-0.01089999,
        cl_alpha_htp=0.74395071,
        cl_alpha_htp_isolated=1.33227995,
        coeff_k_htp=0.26627819,
        cl_alpha_vector=np.array(
            [4.98907042, 4.98907042, 5.30378261, 5.88820249, 6.88161884, 8.82257634]
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
            0.09243095,
            0.09226693,
            0.09189049,
            0.09287295,
            0.09439806,
            0.09530078,
            0.09563392,
            0.0953067,
            0.09408894,
            0.09160488,
            0.08858024,
            0.08504006,
            0.08099342,
            0.0758463,
            0.06885986,
            0.05867507,
            0.04190078,
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
            0.12244236,
            0.12400364,
            0.12519612,
            0.12605197,
            0.1265721,
            0.12673994,
            0.12652302,
            0.12587054,
            0.12470799,
            0.12292806,
            0.12037578,
            0.11682345,
            0.11192635,
            0.10513754,
            0.09552302,
            0.08127223,
            0.05783497,
        ]
    )
    comp_low_speed(
        XML_FILE,
        use_openvsp=False,
        cl0_wing=0.0900082,
        cl_alpha_wing=4.8582102,
        cm0=-0.03209841,
        coeff_k_wing=0.04762593,
        cl0_htp=-0.00787406,
        cl_alpha_htp=0.70734892,
        cl_alpha_htp_isolated=1.13235305,
        coeff_k_htp=0.26409816,
        y_vector_wing=y_vector_wing,
        cl_vector_wing=cl_vector_wing,
        chord_vector_wing=chord_vector_wing,
        cl_ref_htp=0.11558161,
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
        cl0_wing=0.1353942,
        cl_alpha_wing=5.29562763,
        cm0=-0.03079601,
        coeff_k_wing=0.0469921,
        cl0_htp=-0.01164,
        cl_alpha_htp=0.67053251,
        cl_alpha_htp_isolated=1.35235907,
        coeff_k_htp=0.63023023,
        cl_alpha_vector=np.array(
            [5.37384776, 5.37384776, 5.60076567, 6.00159713, 6.62379745, 7.62815533]
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
            1.10696,
            1.27934,
            1.45249,
            1.62628,
            1.80063,
            1.97542,
            2.15054,
            2.32589,
            2.50136,
            2.67684,
            2.85221,
            3.02737,
            3.20221,
            3.37663,
            3.55051,
            3.72375,
            3.89624,
            4.06789,
            4.2386,
            4.40826,
            4.57679,
            4.74408,
            4.91005,
            5.07462,
            5.23769,
            5.39919,
            5.55905,
            5.71718,
            5.87353,
            6.02801,
        ]
    )
    cl_vector_wing = np.array(
        [
            0.12021673,
            0.12034706,
            0.12015659,
            0.12003629,
            0.11996612,
            0.1197957,
            0.11944483,
            0.11950498,
            0.11966537,
            0.12087837,
            0.12180064,
            0.1224723,
            0.12245225,
            0.1227129,
            0.12298356,
            0.1234447,
            0.12359507,
            0.12389582,
            0.12398604,
            0.12412638,
            0.12397601,
            0.12404619,
            0.12395596,
            0.12382564,
            0.12325423,
            0.12281314,
            0.12204124,
            0.12126933,
            0.12018666,
            0.1194749,
            0.11841228,
            0.11724941,
            0.11528456,
            0.11327961,
            0.10986117,
            0.1050092,
            0.09780141,
            0.08506998,
            0.06568213,
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
            1.7763,
            1.75438,
            1.73235,
            1.7102,
            1.68795,
            1.66562,
            1.64322,
            1.62076,
            1.59826,
            1.57573,
            1.55319,
            1.53064,
            1.50811,
            1.4856,
            1.46314,
            1.44073,
            1.41839,
            1.39613,
            1.37397,
            1.35191,
            1.32998,
            1.30818,
            1.28653,
            1.26503,
            1.24371,
            1.22257,
            1.20161,
            1.18086,
            1.16032,
            1.14001,
            1.11992,
            1.10007,
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
            0.12145233,
            0.10679457,
            0.10847123,
            0.10292158,
            0.10089755,
            0.10467755,
            0.10886783,
            0.11188745,
            0.10992634,
            0.11028191,
            0.11181087,
            0.11265603,
            0.10993454,
            0.10901279,
            0.10732519,
            0.10602599,
            0.10566495,
            0.10404573,
            0.10046813,
            0.09726252,
            0.09223255,
            0.08523326,
            0.07205796,
            0.05736191,
        ]
    )
    comp_low_speed(
        XML_FILE,
        use_openvsp=True,
        cl0_wing=0.11883332,
        cl_alpha_wing=4.709,
        cm0=-0.02646533,
        coeff_k_wing=0.0466655,
        cl0_htp=-0.00952,
        cl_alpha_htp=0.65357296,
        cl_alpha_htp_isolated=1.20916995,
        coeff_k_htp=0.58733645,
        y_vector_wing=y_vector_wing,
        cl_vector_wing=cl_vector_wing,
        chord_vector_wing=chord_vector_wing,
        cl_ref_htp=0.10455,
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
        cl_max_clean_wing=1.60,
        cl_min_clean_wing=-1.27,
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
            1.10696,
            1.27934,
            1.45249,
            1.62628,
            1.80063,
            1.97542,
            2.15054,
            2.32589,
            2.50136,
            2.67684,
            2.85221,
            3.02737,
            3.20221,
            3.37663,
            3.55051,
            3.72375,
            3.89624,
            4.06789,
            4.2386,
            4.40826,
            4.57679,
            4.74408,
            4.91005,
            5.07462,
            5.23769,
            5.39919,
            5.55905,
            5.71718,
            5.87353,
            6.02801,
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
            1.33121,
            1.31188,
            1.30964,
            1.3083,
            1.3059,
            1.30202,
            1.29806,
            1.30067,
            1.30467,
            1.32451,
            1.34027,
            1.35118,
            1.35134,
            1.35515,
            1.36099,
            1.36904,
            1.37066,
            1.37541,
            1.37618,
            1.3789,
            1.37461,
            1.37591,
            1.37396,
            1.37269,
            1.36203,
            1.35667,
            1.34536,
            1.33612,
            1.31793,
            1.30616,
            1.28616,
            1.26678,
            1.23169,
            1.19773,
            1.14053,
            1.07249,
            0.97409,
            0.83596,
            0.72512,
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
        delta_cl=-0.00025,
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
            1.10696,
            1.27934,
            1.45249,
            1.62628,
            1.80063,
            1.97542,
            2.15054,
            2.32589,
            2.50136,
            2.67684,
            2.85221,
            3.02737,
            3.20221,
            3.37663,
            3.55051,
            3.72375,
            3.89624,
            4.06789,
            4.2386,
            4.40826,
            4.57679,
            4.74408,
            4.91005,
            5.07462,
            5.23769,
            5.39919,
            5.55905,
            5.71718,
            5.87353,
            6.02801,
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
            1.35247,
            1.33074,
            1.32915,
            1.32781,
            1.32547,
            1.32147,
            1.31715,
            1.31933,
            1.32361,
            1.34202,
            1.35784,
            1.36843,
            1.36865,
            1.37289,
            1.37858,
            1.38667,
            1.38808,
            1.39305,
            1.394,
            1.39692,
            1.39231,
            1.39378,
            1.392,
            1.39096,
            1.38008,
            1.37498,
            1.36387,
            1.35459,
            1.33565,
            1.32456,
            1.30589,
            1.28807,
            1.25365,
            1.22251,
            1.16916,
            1.10492,
            1.01155,
            0.87554,
            0.7785,
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
        delta_cl=-0.00068,
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
            502.17475854,
            1988.68527573,
            3475.19579292,
            4961.70631011,
            6448.2168273,
            7934.72734449,
            9421.23786168,
            10907.74837887,
            12394.25889605,
            13880.76941324,
            15367.27993043,
            16853.79044762,
            18340.30096481,
            19826.811482,
            21313.32199919,
            22799.83251638,
            24286.34303357,
            25772.85355076,
            27259.36406795,
            28745.87458514,
            30232.38510232,
            31718.89561951,
            33205.4061367,
            34691.91665389,
            36178.42717108,
            37664.93768827,
            39151.44820546,
            40637.95872265,
            42124.46923984,
            43610.97975703,
        ]
    )
    thrust_SL_limit = np.array(
        [
            13371.66677748,
            15324.35126122,
            17645.18950182,
            19943.85676657,
            22370.06281802,
            25051.53857605,
            28208.88946904,
            32032.41265602,
            36870.82254847,
            43610.97975703,
        ]
    )
    efficiency_SL = np.array(
        [
            [
                0.06328424,
                0.14583494,
                0.14939457,
                0.13720595,
                0.1238086,
                0.11181988,
                0.10130154,
                0.0919852,
                0.08316432,
                0.07564219,
                0.07564219,
                0.07564219,
                0.07564219,
                0.07564219,
                0.07564219,
                0.07564219,
                0.07564219,
                0.07564219,
                0.07564219,
                0.07564219,
                0.07564219,
                0.07564219,
                0.07564219,
                0.07564219,
                0.07564219,
                0.07564219,
                0.07564219,
                0.07564219,
                0.07564219,
                0.07564219,
            ],
            [
                0.26897434,
                0.51917817,
                0.53819213,
                0.51395802,
                0.48207913,
                0.4502241,
                0.42081057,
                0.3937125,
                0.36853617,
                0.34436215,
                0.31593656,
                0.31593656,
                0.31593656,
                0.31593656,
                0.31593656,
                0.31593656,
                0.31593656,
                0.31593656,
                0.31593656,
                0.31593656,
                0.31593656,
                0.31593656,
                0.31593656,
                0.31593656,
                0.31593656,
                0.31593656,
                0.31593656,
                0.31593656,
                0.31593656,
                0.31593656,
            ],
            [
                0.39751464,
                0.66625014,
                0.69537987,
                0.6807549,
                0.65550398,
                0.62716467,
                0.59892505,
                0.57174835,
                0.54566383,
                0.52050358,
                0.49592642,
                0.46894004,
                0.44559474,
                0.44559474,
                0.44559474,
                0.44559474,
                0.44559474,
                0.44559474,
                0.44559474,
                0.44559474,
                0.44559474,
                0.44559474,
                0.44559474,
                0.44559474,
                0.44559474,
                0.44559474,
                0.44559474,
                0.44559474,
                0.44559474,
                0.44559474,
            ],
            [
                0.43960366,
                0.71961147,
                0.75881179,
                0.75604695,
                0.74016274,
                0.71955112,
                0.69746535,
                0.67497871,
                0.65269337,
                0.63068439,
                0.6090584,
                0.58754725,
                0.56426366,
                0.52660026,
                0.5205537,
                0.5205537,
                0.5205537,
                0.5205537,
                0.5205537,
                0.5205537,
                0.5205537,
                0.5205537,
                0.5205537,
                0.5205537,
                0.5205537,
                0.5205537,
                0.5205537,
                0.5205537,
                0.5205537,
                0.5205537,
            ],
            [
                0.44508481,
                0.73436238,
                0.78429436,
                0.79123668,
                0.78404724,
                0.77066722,
                0.75452356,
                0.73724262,
                0.71937455,
                0.70131264,
                0.6831244,
                0.6650381,
                0.64675929,
                0.62743775,
                0.60395891,
                0.56564042,
                0.56564042,
                0.56564042,
                0.56564042,
                0.56564042,
                0.56564042,
                0.56564042,
                0.56564042,
                0.56564042,
                0.56564042,
                0.56564042,
                0.56564042,
                0.56564042,
                0.56564042,
                0.56564042,
            ],
            [
                0.43294473,
                0.73253426,
                0.79365088,
                0.80866473,
                0.80820634,
                0.80056948,
                0.78969291,
                0.77674726,
                0.76279692,
                0.74827702,
                0.73341132,
                0.71828685,
                0.70311984,
                0.68767062,
                0.67173692,
                0.65403235,
                0.63036527,
                0.6003384,
                0.6003384,
                0.6003384,
                0.6003384,
                0.6003384,
                0.6003384,
                0.6003384,
                0.6003384,
                0.6003384,
                0.6003384,
                0.6003384,
                0.6003384,
                0.6003384,
            ],
            [
                0.40896212,
                0.72587974,
                0.79553724,
                0.81767173,
                0.82221645,
                0.81933684,
                0.81223547,
                0.80310611,
                0.7923607,
                0.78079055,
                0.76862815,
                0.75619273,
                0.74341302,
                0.73052858,
                0.71746491,
                0.70400806,
                0.69003895,
                0.67439424,
                0.6536205,
                0.62123949,
                0.62123949,
                0.62123949,
                0.62123949,
                0.62123949,
                0.62123949,
                0.62123949,
                0.62123949,
                0.62123949,
                0.62123949,
                0.62123949,
            ],
            [
                0.40211454,
                0.71675727,
                0.79395501,
                0.82121257,
                0.83034296,
                0.83112635,
                0.82753481,
                0.82093763,
                0.81310546,
                0.80393255,
                0.79404952,
                0.78372103,
                0.77307287,
                0.76218044,
                0.75104617,
                0.7397727,
                0.72839343,
                0.71663727,
                0.70446026,
                0.69127952,
                0.67625401,
                0.65154351,
                0.63407446,
                0.63407446,
                0.63407446,
                0.63407446,
                0.63407446,
                0.63407446,
                0.63407446,
                0.63407446,
            ],
            [
                0.37071692,
                0.70633326,
                0.79003149,
                0.82171728,
                0.83443445,
                0.83846354,
                0.8373724,
                0.83360423,
                0.82748761,
                0.8206024,
                0.81260619,
                0.80404715,
                0.7951867,
                0.78581368,
                0.77643729,
                0.76676165,
                0.75696874,
                0.74706921,
                0.73704836,
                0.72677269,
                0.7164086,
                0.70570263,
                0.69411478,
                0.68143978,
                0.66331704,
                0.64500377,
                0.64500377,
                0.64500377,
                0.64500377,
                0.64500377,
            ],
            [
                0.33815496,
                0.69562495,
                0.78371091,
                0.81945247,
                0.83536249,
                0.84221975,
                0.84352598,
                0.84163381,
                0.83772947,
                0.83226181,
                0.82593879,
                0.81900725,
                0.81153612,
                0.80362204,
                0.79556597,
                0.78719338,
                0.77878028,
                0.77023156,
                0.76156353,
                0.75283316,
                0.74399218,
                0.7351074,
                0.72609187,
                0.71705059,
                0.70784168,
                0.69842751,
                0.68844147,
                0.67779498,
                0.66462354,
                0.63427787,
            ],
        ]
    )
    thrust_CL = np.array(
        [
            215.08456094,
            1326.58095228,
            2438.07734362,
            3549.57373496,
            4661.0701263,
            5772.56651764,
            6884.06290898,
            7995.55930032,
            9107.05569166,
            10218.55208299,
            11330.04847433,
            12441.54486567,
            13553.04125701,
            14664.53764835,
            15776.03403969,
            16887.53043103,
            17999.02682237,
            19110.52321371,
            20222.01960505,
            21333.51599639,
            22445.01238773,
            23556.50877907,
            24668.00517041,
            25779.50156175,
            26890.99795309,
            28002.49434443,
            29113.99073577,
            30225.48712711,
            31336.98351845,
            32448.47990979,
        ]
    )
    thrust_CL_limit = np.array(
        [
            5497.58624111,
            6326.26391825,
            7321.6686703,
            8359.42976009,
            9499.53339241,
            10827.00486952,
            12473.98515666,
            14695.32669699,
            18248.39573702,
            32448.47990979,
        ]
    )
    efficiency_CL = np.array(
        [
            [
                0.06608959,
                0.15047776,
                0.12733181,
                0.10553593,
                0.08837112,
                0.07501886,
                0.07501886,
                0.07501886,
                0.07501886,
                0.07501886,
                0.07501886,
                0.07501886,
                0.07501886,
                0.07501886,
                0.07501886,
                0.07501886,
                0.07501886,
                0.07501886,
                0.07501886,
                0.07501886,
                0.07501886,
                0.07501886,
                0.07501886,
                0.07501886,
                0.07501886,
                0.07501886,
                0.07501886,
                0.07501886,
                0.07501886,
                0.07501886,
            ],
            [
                0.28093102,
                0.53989152,
                0.49085948,
                0.43264689,
                0.38258392,
                0.33853101,
                0.3131095,
                0.3131095,
                0.3131095,
                0.3131095,
                0.3131095,
                0.3131095,
                0.3131095,
                0.3131095,
                0.3131095,
                0.3131095,
                0.3131095,
                0.3131095,
                0.3131095,
                0.3131095,
                0.3131095,
                0.3131095,
                0.3131095,
                0.3131095,
                0.3131095,
                0.3131095,
                0.3131095,
                0.3131095,
                0.3131095,
                0.3131095,
            ],
            [
                0.41105672,
                0.69521299,
                0.66277288,
                0.61002172,
                0.56003562,
                0.51381481,
                0.46874911,
                0.44173118,
                0.44173118,
                0.44173118,
                0.44173118,
                0.44173118,
                0.44173118,
                0.44173118,
                0.44173118,
                0.44173118,
                0.44173118,
                0.44173118,
                0.44173118,
                0.44173118,
                0.44173118,
                0.44173118,
                0.44173118,
                0.44173118,
                0.44173118,
                0.44173118,
                0.44173118,
                0.44173118,
                0.44173118,
                0.44173118,
            ],
            [
                0.45422695,
                0.75652551,
                0.74434082,
                0.70595362,
                0.66464386,
                0.62435271,
                0.58548311,
                0.54279234,
                0.51599028,
                0.51599028,
                0.51599028,
                0.51599028,
                0.51599028,
                0.51599028,
                0.51599028,
                0.51599028,
                0.51599028,
                0.51599028,
                0.51599028,
                0.51599028,
                0.51599028,
                0.51599028,
                0.51599028,
                0.51599028,
                0.51599028,
                0.51599028,
                0.51599028,
                0.51599028,
                0.51599028,
                0.51599028,
            ],
            [
                0.45932242,
                0.7799628,
                0.78556536,
                0.76032145,
                0.72845137,
                0.69541537,
                0.66255535,
                0.62980483,
                0.59124543,
                0.56030453,
                0.56030453,
                0.56030453,
                0.56030453,
                0.56030453,
                0.56030453,
                0.56030453,
                0.56030453,
                0.56030453,
                0.56030453,
                0.56030453,
                0.56030453,
                0.56030453,
                0.56030453,
                0.56030453,
                0.56030453,
                0.56030453,
                0.56030453,
                0.56030453,
                0.56030453,
                0.56030453,
            ],
            [
                0.4472304,
                0.78689695,
                0.80719297,
                0.79257359,
                0.76916367,
                0.74272788,
                0.71553568,
                0.68805674,
                0.66022901,
                0.62832271,
                0.58644032,
                0.58644032,
                0.58644032,
                0.58644032,
                0.58644032,
                0.58644032,
                0.58644032,
                0.58644032,
                0.58644032,
                0.58644032,
                0.58644032,
                0.58644032,
                0.58644032,
                0.58644032,
                0.58644032,
                0.58644032,
                0.58644032,
                0.58644032,
                0.58644032,
                0.58644032,
            ],
            [
                0.41934895,
                0.78668829,
                0.81887978,
                0.81303039,
                0.79624643,
                0.77530997,
                0.75291677,
                0.72985481,
                0.70649043,
                0.6827954,
                0.65719996,
                0.61358238,
                0.60239799,
                0.60239799,
                0.60239799,
                0.60239799,
                0.60239799,
                0.60239799,
                0.60239799,
                0.60239799,
                0.60239799,
                0.60239799,
                0.60239799,
                0.60239799,
                0.60239799,
                0.60239799,
                0.60239799,
                0.60239799,
                0.60239799,
                0.60239799,
            ],
            [
                0.40894564,
                0.78234285,
                0.82456403,
                0.82551002,
                0.81429403,
                0.79823607,
                0.77975866,
                0.76028494,
                0.74041457,
                0.72031593,
                0.69995912,
                0.6793641,
                0.65691738,
                0.61810408,
                0.61389441,
                0.61389441,
                0.61389441,
                0.61389441,
                0.61389441,
                0.61389441,
                0.61389441,
                0.61389441,
                0.61389441,
                0.61389441,
                0.61389441,
                0.61389441,
                0.61389441,
                0.61389441,
                0.61389441,
                0.61389441,
            ],
            [
                0.36722062,
                0.7755179,
                0.82536911,
                0.83254873,
                0.82603006,
                0.81356787,
                0.79872665,
                0.78228809,
                0.76517823,
                0.7478836,
                0.73042384,
                0.7128732,
                0.69517182,
                0.67765403,
                0.65957222,
                0.64045288,
                0.61292665,
                0.59593472,
                0.59593472,
                0.59593472,
                0.59593472,
                0.59593472,
                0.59593472,
                0.59593472,
                0.59593472,
                0.59593472,
                0.59593472,
                0.59593472,
                0.59593472,
                0.59593472,
            ],
            [
                0.38070599,
                0.76284297,
                0.82141611,
                0.83360185,
                0.83124872,
                0.8223593,
                0.81026366,
                0.79644667,
                0.78174538,
                0.76649525,
                0.75103058,
                0.73554111,
                0.72002932,
                0.70459873,
                0.68920961,
                0.673931,
                0.65862655,
                0.64362236,
                0.62869643,
                0.61351418,
                0.59857786,
                0.58363579,
                0.56837193,
                0.55313803,
                0.53801086,
                0.52137432,
                0.50293234,
                0.48449035,
                0.46604837,
                0.44760639,
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
