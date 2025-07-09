"""Test module for aerodynamics groups."""
#  This file is part of FAST-OAD_CS23 : A framework for rapid Overall Aircraft Design
#  Copyright (C) 2025  ONERA & ISAE-SUPAERO
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
    polar_xfoil,
    polar_neuralfoil,
    polar_single_aoa_xfoil,
    polar_single_aoa_neuralfoil,
    polar_interpolation,
    polar_single_aoa_inv,
    polar_ext_folder_inv,
    polar_ext_folder_neuralfoil,
    airfoil_slope_wt_xfoil,
    airfoil_slope_wt_neuralfoil,
    airfoil_slope_xfoil,
    airfoil_slope_neuralfoil,
    comp_high_speed_xfoil,
    comp_high_speed_neuralfoil,
    comp_low_speed_xfoil,
    comp_low_speed_neuralfoil,
    comp_low_speed_input_aoa_xfoil,
    comp_low_speed_input_aoa_neuralfoil,
    comp_high_speed_input_aoa_xfoil,
    comp_high_speed_input_aoa_neuralfoil,
    hinge_moment_2d,
    hinge_moment_3d,
    hinge_moments,
    high_lift,
    extreme_cl,
    wing_extreme_cl_clean_xfoil,
    wing_extreme_cl_clean_neuralfoil,
    htp_extreme_cl_clean_xfoil,
    htp_extreme_cl_clean_neuralfoil,
    l_d_max,
    cnbeta,
    slipstream_openvsp_cruise,
    slipstream_openvsp_low_speed,
    compute_mach_interpolation_roskam_xfoil,
    compute_mach_interpolation_roskam_neuralfoil,
    cl_alpha_vt,
    cy_delta_r,
    effective_efficiency,
    cm_alpha_fus,
    high_speed_connection,
    low_speed_connection,
    v_n_diagram,
    load_factor,
    propeller_xfoil,
    propeller_neuralfoil,
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
def test_polar_xfoil():
    """Tests polar execution (XFOIL) @ high and low speed."""
    polar_xfoil(
        XML_FILE,
        mach_high_speed=0.245,
        reynolds_high_speed=4571770 * 1.549,
        mach_low_speed=0.1179,
        reynolds_low_speed=2746999 * 1.549,
        cdp_1_high_speed=0.0046,
        cl_max_2d=1.6965,
        cdp_1_low_speed=0.0049,
    )


def test_polar_neuralfoil():
    """Tests polar execution (NeuralFOIL) @ high and low speed."""
    polar_neuralfoil(
        XML_FILE,
        mach_high_speed=0.245,
        reynolds_high_speed=4571770 * 1.549,
        mach_low_speed=0.1179,
        reynolds_low_speed=2746999 * 1.549,
        cdp_1_high_speed=0.0,
        cl_max_2d=1.60120558,
        cdp_1_low_speed=0.0,
    )


@pytest.mark.skipif(
    system() != "Windows" and xfoil_path is None or SKIP_STEPS,
    reason="No XFOIL executable available (or skipped)",
)
def test_polar_single_aoa():
    """Tests polar execution (XFOIL) @ low speed."""
    polar_single_aoa_xfoil(
        XML_FILE,
        mach_low_speed=0.1179,
        reynolds_low_speed=2746999 * 1.549,
    )


def test_polar_single_aoa_neuralfoil():
    """Tests polar execution (NeuralFoil) @ low speed."""
    polar_single_aoa_neuralfoil(
        XML_FILE,
        mach_low_speed=0.1179,
        reynolds_low_speed=2746999 * 1.549,
        alpha=5.0,
        cl=0.72787,
    )


@pytest.mark.skipif(
    system() != "Windows" and xfoil_path is None or SKIP_STEPS,
    reason="No XFOIL executable available (or skipped)",
)
def test_polar_interpolation():
    """Tests polar execution interpolation (XFOIL)."""
    polar_interpolation(
        mach=0.1179,
    )


@pytest.mark.skipif(
    system() != "Windows" and xfoil_path is None or SKIP_STEPS,
    reason="No XFOIL executable available (or skipped)",
)
def test_polar_single_aoa_inv():
    """Tests polar execution (XFOIL) @ high and low speed."""
    polar_single_aoa_inv(
        XML_FILE,
        mach_low_speed=0.1179,
        reynolds_low_speed=2746999 * 1.549,
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


def test_polar_with_ext_folder_neuralfoil():
    """Tests polar execution (NeuralFoil) @ high and low speed."""
    polar_ext_folder_neuralfoil(
        XML_FILE,
        mach_high_speed=0.53835122,
        reynolds_high_speed=5381384,
        mach_low_speed=0.1284,
        reynolds_low_speed=2993524,
        cdp_1_high_speed=0.0,
        cl_max_2d=1.6316,
        cdp_1_low_speed=0.0,
    )


@pytest.mark.skipif(
    system() != "Windows" and xfoil_path is None or SKIP_STEPS,
    reason="No XFOIL executable available",
)
def test_polar_with_ext_folder_inv():
    """Tests polar execution (XFOIL) @ high and low speed."""
    polar_ext_folder_inv(
        XML_FILE,
        mach_low_speed=0.1284,
        reynolds_low_speed=2993524,
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


def test_airfoil_slope_neuralfoil():
    """Tests polar execution (Neuralfoil) @ low speed."""
    airfoil_slope_neuralfoil(
        XML_FILE,
        wing_airfoil_file="naca63_415.af",
        htp_airfoil_file="naca0012.af",
        vtp_airfoil_file="naca0012.af",
        cl_alpha_wing=7.1564,
        cl_alpha_htp=6.2973,
        cl_alpha_vtp=6.2973,
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


def test_airfoil_slope_wt_neuralfoil():
    """Tests polar reading @ low speed."""
    airfoil_slope_wt_neuralfoil(
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
    comp_high_speed_xfoil(
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


@pytest.mark.skipif(system() != "Windows", reason="May result in unexpected values")
def test_vlm_comp_high_speed_neuralfoil():
    """Tests vlm components @ high speed."""
    comp_high_speed_neuralfoil(
        XML_FILE,
        cl0_wing=-0.0004732,
        cl_ref_wing=0.84031781,
        cl_alpha_wing=4.764742,
        cm0=-0.0229502,
        coeff_k_wing=0.04560849,
        cl0_htp=-3.51454933e-05,
        cl_alpha_htp=0.56919111,
        cl_alpha_htp_isolated=0.9230808,
        coeff_k_htp=0.34084416,
        cl_alpha_vector=np.array(
            [5.23342236, 5.23342236, 5.2927491, 5.37031955, 5.46441404, 5.57808973]
        ),
        mach_vector=np.array([0.0, 0.15, 0.21372738, 0.27363573, 0.33015704, 0.38365387]),
    )


@pytest.mark.skipif(
    system() != "Windows" or SKIP_STEPS,
    reason="No XFOIL executable available: VLM basic function not computed with "
    "empty result folder (or skipped)",
)
def test_vlm_comp_high_speed_input_aoa():
    """Tests vlm components @ low speed."""

    comp_high_speed_input_aoa_xfoil(
        XML_FILE,
        use_openvsp=False,
    )


@pytest.mark.skipif(system() != "Windows", reason="May result in unexpected values")
def test_vlm_comp_high_speed_input_aoa_neuralfoil():
    """Tests vlm components @ low speed."""

    comp_high_speed_input_aoa_neuralfoil(XML_FILE)


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
    comp_low_speed_xfoil(
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


@pytest.mark.skipif(system() != "Windows", reason="May result in unexpected values")
def test_vlm_comp_low_speed_neuralfoil():
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
    comp_low_speed_neuralfoil(
        XML_FILE,
        cl0_wing=-0.00046164,
        cl_ref_wing=0.81979372,
        cl_alpha_wing=4.64836698,
        cm0=-0.02238966,
        coeff_k_wing=0.0434078,
        cl0_htp=3.34496579e-05,
        cl_alpha_htp=0.56372144,
        cl_alpha_htp_isolated=0.90053529,
        coeff_k_htp=0.34084416,
        y_vector_wing=y_vector_wing,
        cl_vector_wing=cl_vector_wing,
        chord_vector_wing=chord_vector_wing,
        cl_ref_htp=0.0984214,
        y_vector_htp=y_vector_htp,
        cl_vector_htp=cl_vector_htp,
    )


@pytest.mark.skipif(
    system() != "Windows" or SKIP_STEPS,
    reason="No XFOIL executable available: VLM basic function not computed with "
    "empty result folder (or skipped)",
)
def test_vlm_comp_low_speed_input_aoa():
    """Tests vlm components @ low speed."""

    comp_low_speed_input_aoa_xfoil(
        XML_FILE,
        use_openvsp=False,
    )


@pytest.mark.skipif(system() != "Windows", reason="May result in unexpected values")
def test_vlm_comp_low_speed_input_aoa_neuralfoil():
    """Tests vlm components @ low speed."""

    comp_low_speed_input_aoa_neuralfoil(XML_FILE)


@pytest.mark.skipif(
    system() != "Windows" or SKIP_STEPS, reason="OPENVSP is windows dependent platform (or skipped)"
)
def test_openvsp_comp_high_speed():
    """Tests openvsp components @ high speed."""
    comp_high_speed_xfoil(
        XML_FILE,
        use_openvsp=True,
        cl0_wing=0.02506812,
        cl_ref_wing=0.81641273,
        cl_alpha_wing=4.53407067,
        cm0=-0.02583991,
        coeff_k_wing=0.04361555,
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
def test_openvsp_comp_high_speed_input_aoa():
    """Tests openvsp components @ low speed."""

    comp_high_speed_input_aoa_xfoil(
        XML_FILE,
        use_openvsp=True,
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

    comp_low_speed_xfoil(
        XML_FILE,
        use_openvsp=True,
        cl0_wing=0.02435647,
        cl_ref_wing=0.80189909,
        cl_alpha_wing=4.45499105,
        cm0=-0.0252786,
        coeff_k_wing=0.04357129,
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


@pytest.mark.skipif(
    system() != "Windows" or SKIP_STEPS, reason="OPENVSP is windows dependent platform (or skipped)"
)
def test_openvsp_comp_low_speed_input_aoa():
    """Tests openvsp components @ low speed."""

    comp_low_speed_input_aoa_xfoil(
        XML_FILE,
        use_openvsp=True,
    )


def test_2d_hinge_moment():
    """Tests tail hinge-moments."""
    hinge_moment_2d(XML_FILE, ch_alpha_2d=-0.4329339, ch_delta_2d=-0.63319245)


def test_3d_hinge_moment():
    """Tests tail hinge-moments."""
    hinge_moment_3d(XML_FILE, ch_alpha=-0.28421252, ch_delta=-0.71255199)


def test_all_hinge_moment():
    """Tests tail hinge-moments full computation."""
    hinge_moments(XML_FILE, ch_alpha=-0.28421252, ch_delta=-0.71255199)


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
    wing_extreme_cl_clean_xfoil(
        XML_FILE,
        cl_max_clean_wing=1.39324834,
        cl_min_clean_wing=-1.1170453,
    )


def test_extreme_cl_wing_clean_neuralfoil():
    """Tests maximum/minimum cl component with default result cl=f(y) curve."""
    wing_extreme_cl_clean_neuralfoil(
        XML_FILE,
        cl_max_clean_wing=1.325348,
        cl_min_clean_wing=-0.80819,
    )


@pytest.mark.skipif(
    system() != "Windows",
    reason="No XFOIL executable available: not computed with empty result folder",
)
def test_extreme_cl_htp_clean():
    """Tests maximum/minimum cl component with default result cl=f(y) curve."""
    htp_extreme_cl_clean_xfoil(
        XML_FILE,
        cl_max_clean_htp=0.20504416,
        cl_min_clean_htp=-0.20504416,
        alpha_max_clean_htp=20.854251,
        alpha_min_clean_htp=-20.6853228,
    )


def test_extreme_cl_htp_clean_neuralfoil():
    """Tests maximum/minimum cl component with default result cl=f(y) curve."""
    htp_extreme_cl_clean_neuralfoil(
        XML_FILE,
        cl_max_clean_htp=0.20611143,
        cl_min_clean_htp=-0.21311707,
        alpha_max_clean_htp=20.96279896,
        alpha_min_clean_htp=-21.67531568,
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
    compute_mach_interpolation_roskam_xfoil(
        XML_FILE,
        cl_alpha_vector=np.array(
            [5.40562294, 5.42830832, 5.49748396, 5.61664252, 5.79208041, 6.03371761]
        ),
        mach_vector=np.array([0.0, 0.08, 0.15, 0.23, 0.31, 0.39]),
    )

    compute_mach_interpolation_roskam_neuralfoil(
        XML_FILE,
        cl_alpha_vector=np.array(
            [5.3536284, 5.37615249, 5.44484109, 5.56317807, 5.73744773, 5.97756079]
        ),
        mach_vector=np.array([0.0, 0.07673077, 0.15346155, 0.23019232, 0.3069231, 0.38365387]),
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
    cm_alpha_fus(XML_FILE, cm_alpha_fus_=-0.42322479)


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
            281.50813308,
            468.21866758,
            654.92920209,
            841.63973659,
            1028.3502711,
            1215.0608056,
            1401.77134011,
            1588.48187461,
            1775.19240912,
            1961.90294363,
            2148.61347813,
            2335.32401264,
            2522.03454714,
            2708.74508165,
            2895.45561615,
            3082.16615066,
            3268.87668516,
            3455.58721967,
            3642.29775417,
            3829.00828868,
            4015.71882318,
            4202.42935769,
            4389.13989219,
            4575.8504267,
            4762.56096121,
            4949.27149571,
            5135.98203022,
            5322.69256472,
            5509.40309923,
            5696.11363373,
        ]
    )
    thrust_SL_limit = np.array(
        [
            3909.96732951,
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
                0.11481833,
                0.15851086,
                0.18298579,
                0.19407307,
                0.19732919,
                0.19595566,
                0.19206767,
                0.1869762,
                0.18108355,
                0.17501583,
                0.16894467,
                0.16293479,
                0.15703208,
                0.15119315,
                0.14554806,
                0.1399436,
                0.13430438,
                0.12834843,
                0.12147371,
                0.1100042,
                0.09598577,
                0.09598577,
                0.09598577,
                0.09598577,
                0.09598577,
                0.09598577,
                0.09598577,
                0.09598577,
                0.09598577,
                0.09598577,
            ],
            [
                0.29228526,
                0.38491963,
                0.43201356,
                0.45386967,
                0.4612585,
                0.46072763,
                0.45519724,
                0.44713986,
                0.43729253,
                0.42685449,
                0.41594434,
                0.40493836,
                0.39394536,
                0.38302247,
                0.37202289,
                0.36133046,
                0.3505104,
                0.33969028,
                0.32849101,
                0.31590828,
                0.29852058,
                0.26511387,
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
                0.39445204,
                0.50817379,
                0.56513528,
                0.5925015,
                0.60272736,
                0.6041827,
                0.60091575,
                0.59442654,
                0.58575312,
                0.57589411,
                0.56537669,
                0.55444607,
                0.54329212,
                0.53195226,
                0.52055016,
                0.5090512,
                0.49734605,
                0.48583131,
                0.47401405,
                0.46158032,
                0.44770676,
                0.42972617,
                0.39260997,
                0.38297328,
                0.38297328,
                0.38297328,
                0.38297328,
                0.38297328,
                0.38297328,
                0.38297328,
            ],
            [
                0.46022042,
                0.57754496,
                0.63586353,
                0.6657085,
                0.67959102,
                0.68288622,
                0.68218717,
                0.67853964,
                0.67221994,
                0.66473891,
                0.65574861,
                0.64662482,
                0.6369957,
                0.62704078,
                0.61684956,
                0.60636335,
                0.59567063,
                0.58475111,
                0.57406769,
                0.56277604,
                0.55051411,
                0.536896,
                0.51896606,
                0.48536666,
                0.46716978,
                0.46716978,
                0.46716978,
                0.46716978,
                0.46716978,
                0.46716978,
            ],
            [
                0.51603977,
                0.61654254,
                0.67373395,
                0.70553702,
                0.72217537,
                0.72755174,
                0.72920635,
                0.7278833,
                0.72412463,
                0.71859143,
                0.71211405,
                0.70494927,
                0.69722768,
                0.68894076,
                0.68041033,
                0.67153613,
                0.66231291,
                0.65286483,
                0.64327429,
                0.6334608,
                0.62311275,
                0.61157338,
                0.59848769,
                0.58085838,
                0.54694408,
                0.52878067,
                0.52878067,
                0.52878067,
                0.52878067,
                0.52878067,
            ],
            [
                0.51376066,
                0.64339978,
                0.69213795,
                0.72576516,
                0.74479881,
                0.75350293,
                0.75756459,
                0.75822471,
                0.75651653,
                0.75286044,
                0.74835052,
                0.74313423,
                0.73703541,
                0.73052087,
                0.72358086,
                0.71625161,
                0.70859018,
                0.70057034,
                0.69243174,
                0.68397381,
                0.67507798,
                0.66555512,
                0.65471747,
                0.64238183,
                0.62453374,
                0.59106659,
                0.57476014,
                0.57476014,
                0.57476014,
                0.57476014,
            ],
            [
                0.50196347,
                0.64329711,
                0.70140893,
                0.7349125,
                0.75642549,
                0.76770742,
                0.77393902,
                0.77711801,
                0.77738243,
                0.77579919,
                0.77300154,
                0.76905762,
                0.76440795,
                0.75935107,
                0.75389504,
                0.74799043,
                0.74166119,
                0.73504854,
                0.72812576,
                0.72089568,
                0.71337285,
                0.70537766,
                0.69654622,
                0.68623366,
                0.67429586,
                0.65585544,
                0.62134408,
                0.60942032,
                0.60942032,
                0.60942032,
            ],
            [
                0.49190493,
                0.63667359,
                0.70619582,
                0.73788804,
                0.76209538,
                0.77540647,
                0.78342062,
                0.78877331,
                0.79039305,
                0.79064225,
                0.78986781,
                0.78733897,
                0.78415126,
                0.78038538,
                0.77587299,
                0.77104945,
                0.76592566,
                0.76048152,
                0.75482113,
                0.74886346,
                0.74240654,
                0.73564663,
                0.72825023,
                0.71996313,
                0.71032554,
                0.69802261,
                0.68043235,
                0.6422306,
                0.63602945,
                0.63602945,
            ],
            [
                0.49631709,
                0.62596998,
                0.70307145,
                0.73681636,
                0.76358553,
                0.77918978,
                0.78882839,
                0.79582596,
                0.79876075,
                0.80062211,
                0.80116018,
                0.80007094,
                0.7983172,
                0.79574362,
                0.7923783,
                0.78859366,
                0.78448534,
                0.77984613,
                0.77507494,
                0.77015508,
                0.76492467,
                0.75927385,
                0.75282898,
                0.74601885,
                0.73813678,
                0.72905906,
                0.71675553,
                0.69981493,
                0.65675128,
                0.65675128,
            ],
            [
                0.52067183,
                0.61547868,
                0.69526392,
                0.73309225,
                0.76200406,
                0.77970568,
                0.79133394,
                0.80003056,
                0.80428981,
                0.80751428,
                0.80877268,
                0.80908696,
                0.80855083,
                0.80688716,
                0.80481125,
                0.80206918,
                0.79873532,
                0.79508155,
                0.79113451,
                0.78688405,
                0.78243226,
                0.77772652,
                0.77261264,
                0.76665099,
                0.76031542,
                0.75298855,
                0.74386389,
                0.73205485,
                0.71524196,
                0.67288554,
            ],
        ]
    )
    thrust_CL = np.array(
        [
            214.78415226,
            352.9313781,
            491.07860395,
            629.2258298,
            767.37305564,
            905.52028149,
            1043.66750734,
            1181.81473319,
            1319.96195903,
            1458.10918488,
            1596.25641073,
            1734.40363658,
            1872.55086242,
            2010.69808827,
            2148.84531412,
            2286.99253996,
            2425.13976581,
            2563.28699166,
            2701.43421751,
            2839.58144335,
            2977.7286692,
            3115.87589505,
            3254.0231209,
            3392.17034674,
            3530.31757259,
            3668.46479844,
            3806.61202428,
            3944.75925013,
            4082.90647598,
            4221.05370183,
        ]
    )
    thrust_CL_limit = np.array(
        [
            2893.90572693,
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
    efficiency_CL = np.array(
        [
            [
                0.11264201,
                0.15464073,
                0.17883864,
                0.19018175,
                0.1938572,
                0.1929139,
                0.18943307,
                0.18466698,
                0.17906085,
                0.17323259,
                0.16734975,
                0.16150264,
                0.1557252,
                0.14998988,
                0.14442439,
                0.13887608,
                0.13325483,
                0.12729098,
                0.12031881,
                0.10818041,
                0.09463903,
                0.09463903,
                0.09463903,
                0.09463903,
                0.09463903,
                0.09463903,
                0.09463903,
                0.09463903,
                0.09463903,
                0.09463903,
            ],
            [
                0.28765872,
                0.37654177,
                0.42447955,
                0.44667293,
                0.45466556,
                0.45491239,
                0.44999877,
                0.44254,
                0.4331936,
                0.42319176,
                0.41261619,
                0.40190184,
                0.39115897,
                0.38044669,
                0.36960001,
                0.35903619,
                0.34828187,
                0.33752925,
                0.32630265,
                0.31362749,
                0.29559765,
                0.26221563,
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
                0.38870381,
                0.49928912,
                0.55623818,
                0.58437298,
                0.59560881,
                0.59779913,
                0.59517731,
                0.589261,
                0.58109912,
                0.57165915,
                0.5615182,
                0.55087169,
                0.53998958,
                0.52886476,
                0.51764602,
                0.5062978,
                0.49468411,
                0.48321499,
                0.47144341,
                0.45897933,
                0.44498075,
                0.42661665,
                0.38700306,
                0.37920128,
                0.37920128,
                0.37920128,
                0.37920128,
                0.37920128,
                0.37920128,
                0.37920128,
            ],
            [
                0.45656603,
                0.56929127,
                0.6274439,
                0.65798458,
                0.67274706,
                0.67663431,
                0.67656013,
                0.67347415,
                0.66759399,
                0.66046872,
                0.6518014,
                0.64296386,
                0.63358328,
                0.62382533,
                0.61380437,
                0.60346078,
                0.59286734,
                0.58201317,
                0.57139602,
                0.56010787,
                0.54779089,
                0.53402689,
                0.51563085,
                0.47988445,
                0.46287248,
                0.46287248,
                0.46287248,
                0.46287248,
                0.46287248,
                0.46287248,
            ],
            [
                0.50876069,
                0.60935459,
                0.66598059,
                0.69832916,
                0.71575158,
                0.72167346,
                0.7238545,
                0.72301644,
                0.71964616,
                0.71444896,
                0.70826051,
                0.70133926,
                0.69384331,
                0.68574675,
                0.67737123,
                0.66862466,
                0.65951821,
                0.65012897,
                0.6405707,
                0.63078975,
                0.62043663,
                0.60882453,
                0.59554171,
                0.57750498,
                0.5418194,
                0.52411352,
                0.52411352,
                0.52411352,
                0.52411352,
                0.52411352,
            ],
            [
                0.5086281,
                0.63577424,
                0.68500802,
                0.71903777,
                0.73872235,
                0.74790903,
                0.75246848,
                0.75355133,
                0.75215935,
                0.74882911,
                0.74459332,
                0.73961608,
                0.73370068,
                0.72735333,
                0.72056863,
                0.71336973,
                0.70581091,
                0.69784471,
                0.68978114,
                0.68135209,
                0.67245494,
                0.66291037,
                0.65196617,
                0.6394427,
                0.62108936,
                0.58546456,
                0.56983154,
                0.56983154,
                0.56983154,
                0.56983154,
            ],
            [
                0.49904387,
                0.63643231,
                0.69465593,
                0.72850979,
                0.75054973,
                0.76225725,
                0.7689854,
                0.7725744,
                0.77309678,
                0.77182345,
                0.76929998,
                0.7655531,
                0.76110058,
                0.75621659,
                0.75091091,
                0.74510802,
                0.73889308,
                0.73234744,
                0.7254713,
                0.71828569,
                0.71078927,
                0.70278288,
                0.6939,
                0.68345516,
                0.67137799,
                0.65236627,
                0.61560986,
                0.60425046,
                0.60425046,
                0.60425046,
            ],
            [
                0.49075864,
                0.63035812,
                0.69945409,
                0.73165165,
                0.75634336,
                0.76999841,
                0.77849992,
                0.78425478,
                0.78611765,
                0.78667526,
                0.78617227,
                0.78382325,
                0.78084366,
                0.77725398,
                0.77285514,
                0.7681551,
                0.76314379,
                0.75777834,
                0.75217291,
                0.74624524,
                0.73982982,
                0.73306533,
                0.72565307,
                0.71732414,
                0.70754858,
                0.6950077,
                0.67675746,
                0.63527555,
                0.63060161,
                0.63060161,
            ],
            [
                0.495229,
                0.62000354,
                0.69678401,
                0.73055538,
                0.75784792,
                0.77369513,
                0.78387097,
                0.79127123,
                0.79441528,
                0.79660353,
                0.7973098,
                0.79649126,
                0.79494974,
                0.79249243,
                0.78930383,
                0.78565044,
                0.78162948,
                0.77708359,
                0.77237175,
                0.7675294,
                0.7623227,
                0.75668361,
                0.75026337,
                0.74341128,
                0.73547675,
                0.72625209,
                0.71369156,
                0.69627214,
                0.65105509,
                0.65105509,
            ],
            [
                0.51663834,
                0.60945865,
                0.68897227,
                0.72661533,
                0.7561824,
                0.77400467,
                0.78622124,
                0.79508451,
                0.79980267,
                0.80337701,
                0.80478667,
                0.80537813,
                0.80498079,
                0.80353581,
                0.80164508,
                0.79899565,
                0.79581385,
                0.79226857,
                0.78835197,
                0.7841637,
                0.77976851,
                0.7751211,
                0.77004135,
                0.7640504,
                0.7576655,
                0.7502525,
                0.74112058,
                0.72893318,
                0.71158236,
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
    propeller_xfoil(
        XML_FILE,
        thrust_SL=thrust_SL,
        thrust_SL_limit=thrust_SL_limit,
        efficiency_SL=efficiency_SL,
        thrust_CL=thrust_CL,
        thrust_CL_limit=thrust_CL_limit,
        efficiency_CL=efficiency_CL,
        speed=speed,
    )


def test_propeller_neuralfoil():
    thrust_SL = np.array(
        [
            222.78794375,
            410.51750646,
            598.24706916,
            785.97663187,
            973.70619457,
            1161.43575727,
            1349.16531998,
            1536.89488268,
            1724.62444539,
            1912.35400809,
            2100.0835708,
            2287.8131335,
            2475.54269621,
            2663.27225891,
            2851.00182161,
            3038.73138432,
            3226.46094702,
            3414.19050973,
            3601.92007243,
            3789.64963514,
            3977.37919784,
            4165.10876055,
            4352.83832325,
            4540.56788595,
            4728.29744866,
            4916.02701136,
            5103.75657407,
            5291.48613677,
            5479.21569948,
            5666.94526218,
        ]
    )
    thrust_SL_limit = np.array(
        [
            3939.02698473,
            4246.83873887,
            4495.04974057,
            4704.17743203,
            4884.28076164,
            5040.47420499,
            5195.65204178,
            5349.79851467,
            5502.58540557,
            5666.94526218,
        ]
    )
    efficiency_SL = np.array(
        [
            [
                0.09373318,
                0.14615483,
                0.17674872,
                0.19165858,
                0.19688012,
                0.19652821,
                0.19310031,
                0.18803395,
                0.18218374,
                0.17602838,
                0.16983885,
                0.16376417,
                0.15787159,
                0.15217663,
                0.14666059,
                0.14124918,
                0.13583271,
                0.1303012,
                0.124478,
                0.11795436,
                0.10881214,
                0.10881214,
                0.10881214,
                0.10881214,
                0.10881214,
                0.10881214,
                0.10881214,
                0.10881214,
                0.10881214,
                0.10881214,
            ],
            [
                0.24082499,
                0.35538169,
                0.41666042,
                0.44677894,
                0.45894282,
                0.46072812,
                0.45650065,
                0.44881713,
                0.43920305,
                0.42857984,
                0.41750527,
                0.40631108,
                0.39521822,
                0.38435009,
                0.37372192,
                0.36329707,
                0.35292177,
                0.3424572,
                0.331681,
                0.32033203,
                0.30816265,
                0.29362086,
                0.2808051,
                0.2808051,
                0.2808051,
                0.2808051,
                0.2808051,
                0.2808051,
                0.2808051,
                0.2808051,
            ],
            [
                0.34683571,
                0.48021118,
                0.54870366,
                0.58333213,
                0.59908167,
                0.60381126,
                0.60191708,
                0.59597539,
                0.58739519,
                0.57751179,
                0.56682045,
                0.55569195,
                0.54437992,
                0.53303687,
                0.52176832,
                0.51063697,
                0.49954114,
                0.48831462,
                0.4768695,
                0.46508422,
                0.45241317,
                0.43885818,
                0.42326208,
                0.40293906,
                0.40293906,
                0.40293906,
                0.40293906,
                0.40293906,
                0.40293906,
                0.40293906,
            ],
            [
                0.4063063,
                0.5589716,
                0.62256286,
                0.65676786,
                0.67452029,
                0.68196175,
                0.68290932,
                0.67918572,
                0.67313411,
                0.66556868,
                0.65695227,
                0.64763091,
                0.63785321,
                0.6278118,
                0.61765676,
                0.60747673,
                0.59727449,
                0.58695007,
                0.5762828,
                0.56537984,
                0.55400974,
                0.54165103,
                0.528449,
                0.51320144,
                0.488842,
                0.488842,
                0.488842,
                0.488842,
                0.488842,
                0.488842,
            ],
            [
                0.41989907,
                0.58886935,
                0.66451634,
                0.69676397,
                0.71572743,
                0.725558,
                0.72917801,
                0.72780685,
                0.72430575,
                0.71919021,
                0.71285802,
                0.70558531,
                0.6976789,
                0.68941836,
                0.68089834,
                0.67220658,
                0.66338992,
                0.65439227,
                0.6450893,
                0.6354579,
                0.62548839,
                0.61494847,
                0.60348278,
                0.59099831,
                0.57636593,
                0.55078614,
                0.55078614,
                0.55078614,
                0.55078614,
                0.55078614,
            ],
            [
                0.43428388,
                0.60151169,
                0.68060835,
                0.71879141,
                0.73832764,
                0.75033577,
                0.75642556,
                0.75720834,
                0.75595482,
                0.75303131,
                0.74872115,
                0.74326864,
                0.73723695,
                0.7307263,
                0.72384453,
                0.71666557,
                0.70927253,
                0.70165307,
                0.69372818,
                0.68548826,
                0.67691939,
                0.66791266,
                0.65827122,
                0.6478015,
                0.63598802,
                0.62173181,
                0.59624548,
                0.59624548,
                0.59624548,
                0.59624548,
            ],
            [
                0.49009831,
                0.60677195,
                0.68712018,
                0.73108474,
                0.75060126,
                0.76444554,
                0.77272454,
                0.77539215,
                0.77615165,
                0.77519544,
                0.77244502,
                0.76873686,
                0.76436332,
                0.75941985,
                0.75389135,
                0.74809249,
                0.74203975,
                0.73573176,
                0.72907999,
                0.72209262,
                0.71481549,
                0.70717984,
                0.69901199,
                0.69026617,
                0.68058969,
                0.66941084,
                0.65505604,
                0.6410014,
                0.6410014,
                0.6410014,
            ],
            [
                0.46508986,
                0.60846578,
                0.68774968,
                0.73477663,
                0.75691277,
                0.77245502,
                0.78264241,
                0.78682724,
                0.78935792,
                0.79004953,
                0.78865938,
                0.78651967,
                0.78362895,
                0.77981528,
                0.77561298,
                0.77106161,
                0.76616567,
                0.76091454,
                0.7553667,
                0.74952131,
                0.74339016,
                0.73697639,
                0.73009779,
                0.72266137,
                0.71467192,
                0.70564131,
                0.69502305,
                0.68002546,
                0.66712857,
                0.66712857,
            ],
            [
                0.43386088,
                0.61117797,
                0.68382787,
                0.73434434,
                0.75914494,
                0.77666997,
                0.78838775,
                0.79408233,
                0.79817513,
                0.79991775,
                0.80000961,
                0.79927652,
                0.7973702,
                0.79480012,
                0.79178674,
                0.78822345,
                0.78429717,
                0.78008611,
                0.77555918,
                0.77059605,
                0.76540179,
                0.75995353,
                0.75418422,
                0.74794626,
                0.74119853,
                0.7338364,
                0.72541051,
                0.71525906,
                0.69892756,
                0.68698149,
            ],
            [
                0.40124329,
                0.61368338,
                0.67728837,
                0.73100378,
                0.75827414,
                0.77798119,
                0.79107642,
                0.79850852,
                0.8041872,
                0.80670446,
                0.80812242,
                0.80843212,
                0.80748281,
                0.8060728,
                0.80392176,
                0.80127942,
                0.79833245,
                0.79494832,
                0.79118451,
                0.78714422,
                0.78280647,
                0.77815606,
                0.77324654,
                0.76800776,
                0.76233097,
                0.75614018,
                0.74933056,
                0.74145031,
                0.73162223,
                0.71297178,
            ],
        ]
    )
    thrust_CL = np.array(
        [
            168.51098772,
            307.64251133,
            446.77403494,
            585.90555855,
            725.03708217,
            864.16860578,
            1003.30012939,
            1142.43165301,
            1281.56317662,
            1420.69470023,
            1559.82622384,
            1698.95774746,
            1838.08927107,
            1977.22079468,
            2116.35231829,
            2255.48384191,
            2394.61536552,
            2533.74688913,
            2672.87841275,
            2812.00993636,
            2951.14145997,
            3090.27298358,
            3229.4045072,
            3368.53603081,
            3507.66755442,
            3646.79907804,
            3785.93060165,
            3925.06212526,
            4064.19364887,
            4203.32517249,
        ]
    )
    thrust_CL_limit = np.array(
        [
            2915.67850099,
            3143.8262655,
            3328.09258476,
            3483.57742697,
            3617.88524363,
            3734.70650605,
            3850.24144517,
            3965.91116029,
            4080.80918969,
            4203.32517249,
        ]
    )
    efficiency_CL = np.array(
        [
            [
                0.09115915,
                0.14210053,
                0.17240049,
                0.18759361,
                0.19328473,
                0.19341074,
                0.19040576,
                0.18568562,
                0.1801271,
                0.17420979,
                0.16820929,
                0.16228761,
                0.15651986,
                0.15092539,
                0.14548945,
                0.1401378,
                0.13475348,
                0.12922461,
                0.12336988,
                0.11668046,
                0.10778867,
                0.10778867,
                0.10778867,
                0.10778867,
                0.10778867,
                0.10778867,
                0.10778867,
                0.10778867,
                0.10778867,
                0.10778867,
            ],
            [
                0.23498757,
                0.3467371,
                0.40790277,
                0.43880161,
                0.45193286,
                0.45462434,
                0.45116855,
                0.44412337,
                0.43503092,
                0.42483297,
                0.4141069,
                0.4032014,
                0.39234061,
                0.38166465,
                0.37119618,
                0.36089549,
                0.35061693,
                0.34020636,
                0.329423,
                0.31801848,
                0.30568498,
                0.29050196,
                0.27847034,
                0.27847034,
                0.27847034,
                0.27847034,
                0.27847034,
                0.27847034,
                0.27847034,
                0.27847034,
            ],
            [
                0.34109987,
                0.47071409,
                0.53918658,
                0.57471735,
                0.59149199,
                0.59714732,
                0.59603339,
                0.59072733,
                0.58266103,
                0.57320909,
                0.56287312,
                0.55203662,
                0.54096624,
                0.5298274,
                0.51872756,
                0.50773395,
                0.49675352,
                0.48560361,
                0.47419921,
                0.46238322,
                0.44962054,
                0.43587286,
                0.41984453,
                0.39994734,
                0.39994734,
                0.39994734,
                0.39994734,
                0.39994734,
                0.39994734,
                0.39994734,
            ],
            [
                0.39808517,
                0.54876819,
                0.61337287,
                0.64848204,
                0.66720608,
                0.67550444,
                0.67716598,
                0.67398598,
                0.66840706,
                0.66123255,
                0.65294053,
                0.64389049,
                0.63434278,
                0.6244868,
                0.61448442,
                0.60443747,
                0.59434191,
                0.58410389,
                0.57348901,
                0.56258326,
                0.55117801,
                0.53869182,
                0.52528069,
                0.50956363,
                0.48543626,
                0.48543626,
                0.48543626,
                0.48543626,
                0.48543626,
                0.48543626,
            ],
            [
                0.4139307,
                0.57952833,
                0.65583599,
                0.68897351,
                0.7088438,
                0.7194572,
                0.7237223,
                0.72279679,
                0.71972748,
                0.71497215,
                0.70894177,
                0.70189794,
                0.69419511,
                0.68610909,
                0.67773473,
                0.66916191,
                0.66044698,
                0.65153811,
                0.64228929,
                0.6326802,
                0.62269101,
                0.61208669,
                0.60047593,
                0.58775643,
                0.57261994,
                0.54707256,
                0.54707256,
                0.54707256,
                0.54707256,
                0.54707256,
            ],
            [
                0.43017822,
                0.59294832,
                0.67250886,
                0.71135656,
                0.73177543,
                0.74451778,
                0.75120683,
                0.75237088,
                0.75152753,
                0.74893894,
                0.74486692,
                0.73964619,
                0.73381198,
                0.72746741,
                0.7207222,
                0.71365283,
                0.70635516,
                0.69881522,
                0.69095369,
                0.68274629,
                0.67417705,
                0.66514204,
                0.65540649,
                0.64477669,
                0.63271025,
                0.61790756,
                0.59224807,
                0.59224807,
                0.59224807,
                0.59224807,
            ],
            [
                0.48301077,
                0.59862602,
                0.6793768,
                0.72388269,
                0.74423233,
                0.75880332,
                0.76766245,
                0.77064772,
                0.77180634,
                0.77117657,
                0.7686369,
                0.76516059,
                0.76097846,
                0.75617236,
                0.75078241,
                0.74509785,
                0.73914461,
                0.73292903,
                0.72632947,
                0.71938243,
                0.71211631,
                0.70447396,
                0.69624783,
                0.68739896,
                0.67755547,
                0.66609958,
                0.6511449,
                0.63718501,
                0.63718501,
                0.63718501,
            ],
            [
                0.45956282,
                0.6002631,
                0.68016539,
                0.72805808,
                0.75057002,
                0.76685813,
                0.77742429,
                0.78208619,
                0.78502583,
                0.78594488,
                0.78484184,
                0.7829374,
                0.78022245,
                0.77655498,
                0.77249534,
                0.76806251,
                0.76325298,
                0.75808987,
                0.75261807,
                0.74682563,
                0.7407138,
                0.73430235,
                0.72739145,
                0.71989595,
                0.71179721,
                0.70258448,
                0.69168224,
                0.67603774,
                0.66307585,
                0.66307585,
            ],
            [
                0.42994488,
                0.60212472,
                0.67621461,
                0.72762404,
                0.75267574,
                0.77100522,
                0.78300673,
                0.78926582,
                0.79378639,
                0.79572932,
                0.7961252,
                0.79564248,
                0.79387384,
                0.79148653,
                0.78862343,
                0.78515399,
                0.78133937,
                0.77722908,
                0.77277228,
                0.76786721,
                0.7627091,
                0.75727846,
                0.75150605,
                0.74523947,
                0.73841302,
                0.73093383,
                0.7223383,
                0.71189565,
                0.69476773,
                0.68265794,
            ],
            [
                0.39880503,
                0.60642086,
                0.66943334,
                0.72415184,
                0.75155602,
                0.77215922,
                0.78547506,
                0.79353192,
                0.79966916,
                0.80236986,
                0.80411893,
                0.80458147,
                0.80387248,
                0.80266127,
                0.80061698,
                0.79812152,
                0.79529834,
                0.79199376,
                0.78833099,
                0.78436614,
                0.78006018,
                0.77543735,
                0.77054279,
                0.76529965,
                0.75958035,
                0.75331504,
                0.74640309,
                0.73834614,
                0.72820336,
                0.70882678,
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
    propeller_neuralfoil(
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
    roll_authority_aileron(XML_FILE, cl_delta_a_low_speed_=0.3852, cl_delta_a_cruise_=0.3948)


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
