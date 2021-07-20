"""
    Test module for aerodynamics groups
"""
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

import os.path as pth
import os
import shutil
import glob
import numpy as np
import pytest
import time
import tempfile
from platform import system
from pathlib import Path
from tempfile import TemporaryDirectory

from tests.testing_utilities import run_system, get_indep_var_comp, list_inputs
from tests.xfoil_exe.get_xfoil import get_xfoil_path

from ..components.cd0 import Cd0
from ..external.xfoil.xfoil_polar import XfoilPolar
from ..external.xfoil import resources
from ..external.vlm import ComputeAEROvlm
from ..external.openvsp import ComputeAEROopenvsp
from ..external.openvsp.compute_aero_slipstream import ComputeSlipstreamOpenvsp
from ..components import (
    ComputeExtremeCL,
    ComputeUnitReynolds,
    ComputeCnBetaFuselage,
    ComputeLDMax,
    ComputeDeltaHighLift,
    Compute2DHingeMomentsTail,
    Compute3DHingeMomentsTail,
    ComputeMachInterpolation,
    ComputeCyDeltaRudder,
    ComputeClAlphaVT,
    ComputeAirfoilLiftCurveSlope,
    ComputeVNAndVH,
)
from ..aerodynamics_high_speed import AerodynamicsHighSpeed
from ..aerodynamics_low_speed import AerodynamicsLowSpeed
from ..load_factor import LoadFactor
from ..components.compute_propeller_aero import ComputePropellerPerformance

from .dummy_engines import ENGINE_WRAPPER_BE76 as ENGINE_WRAPPER

RESULTS_FOLDER = pth.join(pth.dirname(__file__), "results")
TMP_SAVE_FOLDER = "test_save"
xfoil_path = None if system() == "Windows" else get_xfoil_path()

XML_FILE = "beechcraft_76.xml"
SKIP_STEPS = True


def _create_tmp_directory() -> TemporaryDirectory:
    """Provide temporary directory for calculation."""
    for tmp_base_path in [None, pth.join(str(Path.home()), ".fast")]:
        if tmp_base_path is not None:
            os.makedirs(tmp_base_path, exist_ok=True)
        tmp_directory = tempfile.TemporaryDirectory(prefix="x", dir=tmp_base_path)
        break

    return tmp_directory


def reshape_curve(y, cl):
    """ Reshape data from openvsp/vlm lift curve """
    for idx in range(len(y)):
        if np.sum(y[idx : len(y)] == 0) == (len(y) - idx):
            y = y[0:idx]
            cl = cl[0:idx]
            break

    return y, cl


def reshape_polar(cl, cdp):
    """ Reshape data from xfoil polar vectors """
    for idx in range(len(cl)):
        if np.sum(cl[idx : len(cl)] == 0) == (len(cl) - idx):
            cl = cl[0:idx]
            cdp = cdp[0:idx]
            break
    return cl, cdp


def polar_result_transfer():
    # Put saved polar results in a temporary folder to activate Xfoil run and have repeatable results  [need writing
    # permission]

    tmp_folder = _create_tmp_directory()

    files = glob.iglob(pth.join(resources.__path__[0], "*.csv"))

    for file in files:
        if os.path.isfile(file):
            shutil.copy(file, tmp_folder.name)
            # noinspection PyBroadException
            try:
                os.remove(file)
            except:
                pass

    return tmp_folder


def polar_result_retrieve(tmp_folder):
    # Retrieve the polar results set aside during the test duration if there are some [need writing permission]

    files = glob.iglob(pth.join(tmp_folder.name, "*.csv"))

    for file in files:
        if os.path.isfile(file):
            # noinspection PyBroadException
            try:
                shutil.copy(file, resources.__path__[0])
            except:
                pass

    tmp_folder.cleanup()


def test_compute_reynolds():
    """ Tests high and low speed reynolds calculation """
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(ComputeUnitReynolds()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeUnitReynolds(), ivc)
    mach = problem["data:aerodynamics:cruise:mach"]
    assert mach == pytest.approx(0.2488, abs=1e-4)
    reynolds = problem.get_val("data:aerodynamics:cruise:unit_reynolds", units="m**-1")
    assert reynolds == pytest.approx(4629639, abs=1)

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(ComputeUnitReynolds(low_speed_aero=True)), __file__, XML_FILE
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeUnitReynolds(low_speed_aero=True), ivc)
    mach = problem["data:aerodynamics:low_speed:mach"]
    assert mach == pytest.approx(0.1179, abs=1e-4)
    reynolds = problem.get_val("data:aerodynamics:low_speed:unit_reynolds", units="m**-1")
    assert reynolds == pytest.approx(2746999, abs=1)


def test_cd0_high_speed():
    """ Tests drag coefficient @ high speed """
    # Research independent input value in .xml file
    # noinspection PyTypeChecker
    ivc = get_indep_var_comp(list_inputs(Cd0(propulsion_id=ENGINE_WRAPPER)), __file__, XML_FILE)

    # noinspection PyTypeChecker
    problem = run_system(Cd0(propulsion_id=ENGINE_WRAPPER), ivc)
    cd0_wing = problem["data:aerodynamics:wing:cruise:CD0"]
    assert cd0_wing == pytest.approx(0.00541, abs=1e-5)
    cd0_fus = problem["data:aerodynamics:fuselage:cruise:CD0"]
    assert cd0_fus == pytest.approx(0.00490, abs=1e-5)
    cd0_ht = problem["data:aerodynamics:horizontal_tail:cruise:CD0"]
    assert cd0_ht == pytest.approx(0.00119, abs=1e-5)
    cd0_vt = problem["data:aerodynamics:vertical_tail:cruise:CD0"]
    assert cd0_vt == pytest.approx(0.00066, abs=1e-5)
    cd0_nac = problem["data:aerodynamics:nacelles:cruise:CD0"]
    assert cd0_nac == pytest.approx(0.00209, abs=1e-5)
    cd0_lg = problem["data:aerodynamics:landing_gear:cruise:CD0"]
    assert cd0_lg == pytest.approx(0.0, abs=1e-5)
    cd0_other = problem["data:aerodynamics:other:cruise:CD0"]
    assert cd0_other == pytest.approx(0.00205, abs=1e-5)
    cd0_total = 1.25 * (cd0_wing + cd0_fus + cd0_ht + cd0_vt + cd0_nac + cd0_lg + cd0_other)
    assert cd0_total == pytest.approx(0.02040, abs=1e-5)


def test_cd0_low_speed():
    """ Tests drag coefficient @ low speed """
    # Research independent input value in .xml file
    # noinspection PyTypeChecker
    ivc = get_indep_var_comp(
        list_inputs(Cd0(propulsion_id=ENGINE_WRAPPER, low_speed_aero=True)), __file__, XML_FILE
    )

    # noinspection PyTypeChecker
    problem = run_system(Cd0(propulsion_id=ENGINE_WRAPPER, low_speed_aero=True), ivc)
    cd0_wing = problem["data:aerodynamics:wing:low_speed:CD0"]
    assert cd0_wing == pytest.approx(0.00587, abs=1e-5)
    cd0_fus = problem["data:aerodynamics:fuselage:low_speed:CD0"]
    assert cd0_fus == pytest.approx(0.00543, abs=1e-5)
    cd0_ht = problem["data:aerodynamics:horizontal_tail:low_speed:CD0"]
    assert cd0_ht == pytest.approx(0.00129, abs=1e-5)
    cd0_vt = problem["data:aerodynamics:vertical_tail:low_speed:CD0"]
    assert cd0_vt == pytest.approx(0.00074, abs=1e-5)
    cd0_nac = problem["data:aerodynamics:nacelles:low_speed:CD0"]
    assert cd0_nac == pytest.approx(0.00229, abs=1e-5)
    cd0_lg = problem["data:aerodynamics:landing_gear:low_speed:CD0"]
    assert cd0_lg == pytest.approx(0.01459, abs=1e-5)
    cd0_other = problem["data:aerodynamics:other:low_speed:CD0"]
    assert cd0_other == pytest.approx(0.00205, abs=1e-5)
    cd0_total = 1.25 * (cd0_wing + cd0_fus + cd0_ht + cd0_vt + cd0_nac + cd0_lg + cd0_other)
    assert cd0_total == pytest.approx(0.04036, abs=1e-5)


@pytest.mark.skipif(
    system() != "Windows" and xfoil_path is None or SKIP_STEPS,
    reason="No XFOIL executable available",
)
def test_polar():
    """ Tests polar execution (XFOIL) @ high and low speed """
    # Transfer saved polar results to temporary folder
    tmp_folder = polar_result_transfer()

    # Define high-speed parameters (with .xml file and additional inputs)
    ivc = get_indep_var_comp(list_inputs(XfoilPolar()), __file__, XML_FILE)
    ivc.add_output("xfoil:mach", 0.245)
    ivc.add_output("xfoil:reynolds", 4571770 * 1.549)

    # Run problem
    xfoil_comp = XfoilPolar(
        alpha_start=0.0, alpha_end=25.0, iter_limit=20, xfoil_exe_path=xfoil_path
    )
    problem = run_system(xfoil_comp, ivc)

    # Retrieve polar results from temporary folder
    polar_result_retrieve(tmp_folder)

    # Check obtained value(s) is/(are) correct
    cl = problem["xfoil:CL"]
    cdp = problem["xfoil:CDp"]
    cl, cdp = reshape_polar(cl, cdp)
    cdp_1 = np.interp(1.0, cl, cdp)
    assert cdp_1 == pytest.approx(0.0046, abs=1e-4)

    # Transfer saved polar results to temporary folder
    tmp_folder = polar_result_transfer()

    # Define low-speed parameters (with .xml file and additional inputs)
    ivc = get_indep_var_comp(list_inputs(XfoilPolar()), __file__, XML_FILE)
    ivc.add_output("xfoil:mach", 0.1179)
    ivc.add_output("xfoil:reynolds", 2746999 * 1.549)

    # Run problem
    xfoil_comp = XfoilPolar(
        alpha_start=0.0, alpha_end=25.0, iter_limit=20, xfoil_exe_path=xfoil_path
    )
    problem = run_system(xfoil_comp, ivc)

    # Retrieve polar results from temporary folder
    polar_result_retrieve(tmp_folder)

    # Check obtained value(s) is/(are) correct
    cl = problem["xfoil:CL"]
    cdp = problem["xfoil:CDp"]
    cl_max_2d = problem["xfoil:CL_max_2D"]
    assert cl_max_2d == pytest.approx(1.6965, abs=1e-4)
    cl, cdp = reshape_polar(cl, cdp)
    cdp_1 = np.interp(1.0, cl, cdp)
    assert cdp_1 == pytest.approx(0.0049, abs=1e-4)


@pytest.mark.skipif(
    system() != "Windows" and xfoil_path is None or SKIP_STEPS,
    reason="No XFOIL executable available",
)
def test_airfoil_slope():
    """ Tests polar execution (XFOIL) @ high speed """
    # Transfer saved polar results to temporary folder
    tmp_folder = polar_result_transfer()

    # Define high-speed parameters (with .xml file and additional inputs)
    ivc = get_indep_var_comp(
        list_inputs(
            ComputeAirfoilLiftCurveSlope(
                wing_airfoil_file="naca63_415.af",
                htp_airfoil_file="naca0012.af",
                vtp_airfoil_file="naca0012.af",
            )
        ),
        __file__,
        XML_FILE,
    )

    # Run problem
    problem = run_system(
        ComputeAirfoilLiftCurveSlope(
            wing_airfoil_file="naca63_415.af",
            htp_airfoil_file="naca0012.af",
            vtp_airfoil_file="naca0012.af",
        ),
        ivc,
    )

    # Retrieve polar results from temporary folder
    polar_result_retrieve(tmp_folder)

    # Check obtained value(s) is/(are) correct
    cl_alpha_wing = problem.get_val("data:aerodynamics:wing:airfoil:CL_alpha", units="rad**-1")
    assert cl_alpha_wing == pytest.approx(6.4975, abs=1e-4)
    cl_alpha_htp = problem.get_val(
        "data:aerodynamics:horizontal_tail:airfoil:CL_alpha", units="rad**-1"
    )
    assert cl_alpha_htp == pytest.approx(6.3321, abs=1e-4)
    cl_alpha_vtp = problem.get_val(
        "data:aerodynamics:vertical_tail:airfoil:CL_alpha", units="rad**-1"
    )
    assert cl_alpha_vtp == pytest.approx(6.3321, abs=1e-4)


def test_airfoil_slope_wt_xfoil():
    """ Tests polar reading @ high speed """
    # Define high-speed parameters (with .xml file and additional inputs)
    ivc = get_indep_var_comp(
        list_inputs(
            ComputeAirfoilLiftCurveSlope(
                wing_airfoil_file="naca63_415.af",
                htp_airfoil_file="naca0012.af",
                vtp_airfoil_file="naca0012.af",
            )
        ),
        __file__,
        XML_FILE,
    )

    # Run problem
    run_system(
        ComputeAirfoilLiftCurveSlope(
            wing_airfoil_file="naca63_415.af",
            htp_airfoil_file="naca0012.af",
            vtp_airfoil_file="naca0012.af",
        ),
        ivc,
    )


@pytest.mark.skipif(
    system() != "Windows" or SKIP_STEPS,
    reason="No XFOIL executable available: VLM basic function not computed with "
    "empty result folder",
)
def test_vlm_comp_high_speed():
    """ Tests vlm components @ high speed """
    for mach_interpolation in [True, False]:

        # Create result temporary directory
        results_folder = _create_tmp_directory()

        # Transfer saved polar results to temporary folder
        tmp_folder = polar_result_transfer()

        # Research independent input value in .xml file
        # noinspection PyTypeChecker
        ivc = get_indep_var_comp(list_inputs(ComputeAEROvlm()), __file__, XML_FILE)

        # Run problem twice
        start = time.time()
        # noinspection PyTypeChecker
        problem = run_system(
            ComputeAEROvlm(
                result_folder_path=results_folder.name,
                compute_mach_interpolation=mach_interpolation,
            ),
            ivc,
        )
        stop = time.time()
        duration_1st_run = stop - start
        start = time.time()
        # noinspection PyTypeChecker
        run_system(
            ComputeAEROvlm(
                result_folder_path=results_folder.name,
                compute_mach_interpolation=mach_interpolation,
            ),
            ivc,
        )
        stop = time.time()
        duration_2nd_run = stop - start

        # Retrieve polar results from temporary folder
        polar_result_retrieve(tmp_folder)

        # Check obtained value(s) is/(are) correct
        cl0_wing = problem["data:aerodynamics:wing:cruise:CL0_clean"]
        assert cl0_wing == pytest.approx(0.0894, abs=1e-4)
        cl_alpha_wing = problem.get_val("data:aerodynamics:wing:cruise:CL_alpha", units="rad**-1")
        assert cl_alpha_wing == pytest.approx(4.820, abs=1e-3)
        cm0 = problem["data:aerodynamics:wing:cruise:CM0_clean"]
        assert cm0 == pytest.approx(-0.0247, abs=1e-4)
        coef_k_wing = problem["data:aerodynamics:wing:cruise:induced_drag_coefficient"]
        assert coef_k_wing == pytest.approx(0.0522, abs=1e-4)
        cl0_htp = problem["data:aerodynamics:horizontal_tail:cruise:CL0"]
        assert cl0_htp == pytest.approx(-0.0058, abs=1e-4)
        cl_alpha_htp = problem.get_val(
            "data:aerodynamics:horizontal_tail:cruise:CL_alpha", units="rad**-1"
        )
        assert cl_alpha_htp == pytest.approx(0.5068, abs=1e-4)
        cl_alpha_htp_isolated = problem.get_val(
            "data:aerodynamics:horizontal_tail:cruise:CL_alpha_isolated", units="rad**-1"
        )
        assert cl_alpha_htp_isolated == pytest.approx(0.8223, abs=1e-4)
        coef_k_htp = problem["data:aerodynamics:horizontal_tail:cruise:induced_drag_coefficient"]
        assert coef_k_htp == pytest.approx(0.4252, abs=1e-4)
        if mach_interpolation:
            cl_alpha_vector = problem[
                "data:aerodynamics:aircraft:mach_interpolation:CL_alpha_vector"
            ]
            assert cl_alpha_vector == pytest.approx([4.8, 4.8, 4.86, 4.94, 5.03, 5.14], abs=1e-2)
            mach_vector = problem["data:aerodynamics:aircraft:mach_interpolation:mach_vector"]
            assert mach_vector == pytest.approx([0.0, 0.15, 0.21, 0.27, 0.33, 0.39], abs=1e-2)

        # Run problem 2nd time to check time reduction

        assert (duration_2nd_run / duration_1st_run) <= 0.1

        # Remove existing result files
        results_folder.cleanup()


@pytest.mark.skipif(
    system() != "Windows" or SKIP_STEPS,
    reason="No XFOIL executable available: VLM basic function not computed with "
    "empty result folder",
)
def test_vlm_comp_low_speed():
    """ Tests vlm components @ low speed """
    # Create result temporary directory
    results_folder = _create_tmp_directory()

    # Transfer saved polar results to temporary folder
    tmp_folder = polar_result_transfer()

    # Research independent input value in .xml file
    # noinspection PyTypeChecker
    ivc = get_indep_var_comp(list_inputs(ComputeAEROvlm(low_speed_aero=True)), __file__, XML_FILE)

    # Run problem twice
    start = time.time()
    # noinspection PyTypeChecker
    problem = run_system(
        ComputeAEROvlm(low_speed_aero=True, result_folder_path=results_folder.name), ivc
    )
    stop = time.time()
    duration_1st_run = stop - start
    start = time.time()
    # noinspection PyTypeChecker
    run_system(ComputeAEROvlm(low_speed_aero=True, result_folder_path=results_folder.name), ivc)
    stop = time.time()
    duration_2nd_run = stop - start

    # Retrieve polar results from temporary folder
    polar_result_retrieve(tmp_folder)

    # Check obtained value(s) is/(are) correct
    cl0_wing = problem["data:aerodynamics:wing:low_speed:CL0_clean"]
    assert cl0_wing == pytest.approx(0.0872, abs=1e-4)
    cl_alpha_wing = problem.get_val("data:aerodynamics:wing:low_speed:CL_alpha", units="rad**-1")
    assert cl_alpha_wing == pytest.approx(4.701, abs=1e-3)
    cm0 = problem["data:aerodynamics:wing:low_speed:CM0_clean"]
    assert cm0 == pytest.approx(-0.0241, abs=1e-4)
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
    y, cl = reshape_curve(
        problem.get_val("data:aerodynamics:wing:low_speed:Y_vector", "m"),
        problem["data:aerodynamics:wing:low_speed:CL_vector"],
    )
    chord = problem.get_val("data:aerodynamics:wing:low_speed:chord_vector", "m")
    assert np.max(np.abs(y_vector_wing - y)) <= 1e-3
    assert np.max(np.abs(cl_vector_wing - cl)) <= 1e-3
    assert np.max(np.abs(chord_vector_wing - chord)) <= 1e-3
    coef_k_wing = problem["data:aerodynamics:wing:low_speed:induced_drag_coefficient"]
    assert coef_k_wing == pytest.approx(0.0500, abs=1e-4)
    cl0_htp = problem["data:aerodynamics:horizontal_tail:low_speed:CL0"]
    assert cl0_htp == pytest.approx(-0.0055, abs=1e-4)
    cl_ref_htp = problem["data:aerodynamics:horizontal_tail:low_speed:CL_ref"]
    assert cl_ref_htp == pytest.approx(0.0820, abs=1e-4)
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
    y, cl = reshape_curve(
        problem.get_val("data:aerodynamics:horizontal_tail:low_speed:Y_vector", "m"),
        problem["data:aerodynamics:horizontal_tail:low_speed:CL_vector"],
    )
    assert np.max(np.abs(y_vector_htp - y)) <= 1e-3
    assert np.max(np.abs(cl_vector_htp - cl)) <= 1e-3
    cl_alpha_htp = problem.get_val(
        "data:aerodynamics:horizontal_tail:low_speed:CL_alpha", units="rad**-1"
    )
    assert cl_alpha_htp == pytest.approx(0.5019, abs=1e-4)
    cl_alpha_htp_isolated = problem.get_val(
        "data:aerodynamics:horizontal_tail:low_speed:CL_alpha_isolated", units="rad**-1"
    )
    assert cl_alpha_htp_isolated == pytest.approx(0.8020, abs=1e-4)
    coef_k_htp = problem["data:aerodynamics:horizontal_tail:low_speed:induced_drag_coefficient"]
    assert coef_k_htp == pytest.approx(0.4287, abs=1e-4)
    assert (duration_2nd_run / duration_1st_run) <= 0.1

    # Remove existing result files
    results_folder.cleanup()


@pytest.mark.skipif(
    system() != "Windows" or SKIP_STEPS, reason="OPENVSP is windows dependent platform"
)
def test_openvsp_comp_high_speed():
    """ Tests openvsp components @ high speed """
    for mach_interpolation in [True, False]:

        # Create result temporary directory
        results_folder = _create_tmp_directory()

        # Research independent input value in .xml file
        # noinspection PyTypeChecker
        ivc = get_indep_var_comp(list_inputs(ComputeAEROopenvsp()), __file__, XML_FILE)

        # Run problem twice
        start = time.time()
        # noinspection PyTypeChecker
        problem = run_system(
            ComputeAEROopenvsp(
                result_folder_path=results_folder.name,
                compute_mach_interpolation=mach_interpolation,
            ),
            ivc,
        )
        stop = time.time()
        duration_1st_run = stop - start
        start = time.time()
        # noinspection PyTypeChecker
        run_system(
            ComputeAEROopenvsp(
                result_folder_path=results_folder.name,
                compute_mach_interpolation=mach_interpolation,
            ),
            ivc,
        )
        stop = time.time()
        duration_2nd_run = stop - start

        # Check obtained value(s) is/(are) correct
        cl0_wing = problem["data:aerodynamics:wing:cruise:CL0_clean"]
        assert cl0_wing == pytest.approx(0.1170, abs=1e-4)
        cl_alpha_wing = problem.get_val("data:aerodynamics:wing:cruise:CL_alpha", units="rad**-1")
        assert cl_alpha_wing == pytest.approx(4.591, abs=1e-3)
        cm0 = problem["data:aerodynamics:wing:cruise:CM0_clean"]
        assert cm0 == pytest.approx(-0.0264, abs=1e-4)
        coef_k_wing = problem["data:aerodynamics:wing:cruise:induced_drag_coefficient"]
        assert coef_k_wing == pytest.approx(0.0483, abs=1e-4)
        if mach_interpolation:
            cl_alpha_vector = problem[
                "data:aerodynamics:aircraft:mach_interpolation:CL_alpha_vector"
            ]
            assert cl_alpha_vector == pytest.approx([5.06, 5.06, 5.10, 5.15, 5.22, 5.29], abs=1e-2)
            mach_vector = problem["data:aerodynamics:aircraft:mach_interpolation:mach_vector"]
            assert mach_vector == pytest.approx([0.0, 0.15, 0.21, 0.27, 0.33, 0.38], abs=1e-2)
        cl0_htp = problem["data:aerodynamics:horizontal_tail:cruise:CL0"]
        assert cl0_htp == pytest.approx(-0.0046, abs=1e-4)
        cl_alpha_htp = problem.get_val(
            "data:aerodynamics:horizontal_tail:cruise:CL_alpha", units="rad**-1"
        )
        assert cl_alpha_htp == pytest.approx(0.5433, abs=1e-4)
        cl_alpha_htp_isolated = problem.get_val(
            "data:aerodynamics:horizontal_tail:cruise:CL_alpha_isolated", units="rad**-1"
        )
        assert cl_alpha_htp_isolated == pytest.approx(0.8438, abs=1e-4)
        coef_k_htp = problem["data:aerodynamics:horizontal_tail:cruise:induced_drag_coefficient"]
        assert coef_k_htp == pytest.approx(0.6684, abs=1e-4)
        assert (duration_2nd_run / duration_1st_run) <= 0.01

        # Remove existing result files
        results_folder.cleanup()


@pytest.mark.skipif(
    system() != "Windows" or SKIP_STEPS, reason="OPENVSP is windows dependent platform"
)
def test_openvsp_comp_low_speed():
    """ Tests openvsp components @ low speed """
    # Create result temporary directory
    results_folder = _create_tmp_directory()

    # Research independent input value in .xml file
    # noinspection PyTypeChecker
    ivc = get_indep_var_comp(
        list_inputs(ComputeAEROopenvsp(low_speed_aero=True)), __file__, XML_FILE
    )

    # Run problem twice
    start = time.time()
    # noinspection PyTypeChecker
    problem = run_system(
        ComputeAEROopenvsp(low_speed_aero=True, result_folder_path=results_folder.name), ivc
    )
    stop = time.time()
    duration_1st_run = stop - start
    start = time.time()
    # noinspection PyTypeChecker
    run_system(ComputeAEROopenvsp(low_speed_aero=True, result_folder_path=results_folder.name), ivc)
    stop = time.time()
    duration_2nd_run = stop - start

    # Check obtained value(s) is/(are) correct
    cl0_wing = problem["data:aerodynamics:wing:low_speed:CL0_clean"]
    assert cl0_wing == pytest.approx(0.1147, abs=1e-4)
    cl_alpha_wing = problem.get_val("data:aerodynamics:wing:low_speed:CL_alpha", units="rad**-1")
    assert cl_alpha_wing == pytest.approx(4.510, abs=1e-3)
    cm0 = problem["data:aerodynamics:wing:low_speed:CM0_clean"]
    assert cm0 == pytest.approx(-0.0258, abs=1e-4)
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
    y, cl = reshape_curve(
        problem.get_val("data:aerodynamics:wing:low_speed:Y_vector", "m"),
        problem["data:aerodynamics:wing:low_speed:CL_vector"],
    )
    assert np.max(np.abs(y_vector_wing - y)) <= 1e-3
    assert np.max(np.abs(cl_vector_wing - cl)) <= 1e-3
    coef_k_wing = problem["data:aerodynamics:wing:low_speed:induced_drag_coefficient"]
    assert coef_k_wing == pytest.approx(0.0483, abs=1e-4)
    cl0_htp = problem["data:aerodynamics:horizontal_tail:low_speed:CL0"]
    assert cl0_htp == pytest.approx(-0.0044, abs=1e-4)
    cl_ref_htp = problem["data:aerodynamics:horizontal_tail:low_speed:CL_ref"]
    assert cl_ref_htp == pytest.approx(0.0897, abs=1e-4)
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
    y, cl = reshape_curve(
        problem.get_val("data:aerodynamics:horizontal_tail:low_speed:Y_vector", "m"),
        problem["data:aerodynamics:horizontal_tail:low_speed:CL_vector"],
    )
    assert np.max(np.abs(y_vector_htp - y)) <= 1e-3
    assert np.max(np.abs(cl_vector_htp - cl)) <= 1e-3
    cl_alpha_htp = problem.get_val(
        "data:aerodynamics:horizontal_tail:low_speed:CL_alpha", units="rad**-1"
    )
    assert cl_alpha_htp == pytest.approx(0.5401, abs=1e-4)
    cl_alpha_htp_isolated = problem.get_val(
        "data:aerodynamics:horizontal_tail:low_speed:CL_alpha_isolated", units="rad**-1"
    )
    assert cl_alpha_htp_isolated == pytest.approx(0.8318, abs=1e-4)
    coef_k_htp = problem["data:aerodynamics:horizontal_tail:low_speed:induced_drag_coefficient"]
    assert coef_k_htp == pytest.approx(0.6648, abs=1e-4)
    assert (duration_2nd_run / duration_1st_run) <= 0.1

    # Remove existing result files
    results_folder.cleanup()


def test_2d_hinge_moment():
    """ Tests tail hinge-moments """
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(Compute2DHingeMomentsTail()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(Compute2DHingeMomentsTail(), ivc)
    ch_alpha_2d = problem.get_val(
        "data:aerodynamics:horizontal_tail:cruise:hinge_moment:CH_alpha_2D", units="rad**-1"
    )
    assert ch_alpha_2d == pytest.approx(-0.3998, abs=1e-4)
    ch_delta_2d = problem.get_val(
        "data:aerodynamics:horizontal_tail:cruise:hinge_moment:CH_delta_2D", units="rad**-1"
    )
    assert ch_delta_2d == pytest.approx(-0.6146, abs=1e-4)


def test_3d_hinge_moment():
    """ Tests tail hinge-moments """
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(Compute3DHingeMomentsTail()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(Compute3DHingeMomentsTail(), ivc)
    ch_alpha = problem.get_val(
        "data:aerodynamics:horizontal_tail:cruise:hinge_moment:CH_alpha", units="rad**-1"
    )
    assert ch_alpha == pytest.approx(-0.2625, abs=1e-4)
    ch_delta = problem.get_val(
        "data:aerodynamics:horizontal_tail:cruise:hinge_moment:CH_delta", units="rad**-1"
    )
    assert ch_delta == pytest.approx(-0.6822, abs=1e-4)


def test_high_lift():
    """ Tests high-lift contribution """
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(ComputeDeltaHighLift()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeDeltaHighLift(), ivc)
    delta_cl0_landing = problem["data:aerodynamics:flaps:landing:CL"]
    assert delta_cl0_landing == pytest.approx(0.5037, abs=1e-4)
    delta_clmax_landing = problem["data:aerodynamics:flaps:landing:CL_max"]
    assert delta_clmax_landing == pytest.approx(0.3613, abs=1e-4)
    delta_cm_landing = problem["data:aerodynamics:flaps:landing:CM"]
    assert delta_cm_landing == pytest.approx(-0.0680, abs=1e-4)
    delta_cd_landing = problem["data:aerodynamics:flaps:landing:CD"]
    assert delta_cd_landing == pytest.approx(0.005, abs=1e-4)
    delta_cl0_takeoff = problem["data:aerodynamics:flaps:takeoff:CL"]
    assert delta_cl0_takeoff == pytest.approx(0.1930, abs=1e-4)
    delta_clmax_takeoff = problem["data:aerodynamics:flaps:takeoff:CL_max"]
    assert delta_clmax_takeoff == pytest.approx(0.0740, abs=1e-4)
    delta_cm_takeoff = problem["data:aerodynamics:flaps:takeoff:CM"]
    assert delta_cm_takeoff == pytest.approx(-0.0260, abs=1e-4)
    delta_cd_takeoff = problem["data:aerodynamics:flaps:takeoff:CD"]
    assert delta_cd_takeoff == pytest.approx(0.0004, abs=1e-4)
    cl_delta_elev = problem.get_val(
        "data:aerodynamics:elevator:low_speed:CL_delta", units="rad**-1"
    )
    assert cl_delta_elev == pytest.approx(0.5115, abs=1e-4)
    cd_delta_elev = problem.get_val(
        "data:aerodynamics:elevator:low_speed:CD_delta", units="rad**-2"
    )
    assert cd_delta_elev == pytest.approx(0.0680, abs=1e-4)


@pytest.mark.skipif(
    system() != "Windows",
    reason="No XFOIL executable available: not computed with empty result folder",
)
def test_extreme_cl():
    """ Tests maximum/minimum cl component with default result cl=f(y) curve"""
    # Transfer saved polar results to temporary folder
    tmp_folder = polar_result_transfer()

    # Research independent input value in .xml file for Openvsp test
    ivc = get_indep_var_comp(list_inputs(ComputeExtremeCL()), __file__, XML_FILE)

    # Run problem
    problem = run_system(ComputeExtremeCL(), ivc)

    # Retrieve polar results from temporary folder
    polar_result_retrieve(tmp_folder)

    # Check obtained value(s) is/(are) correct
    cl_max_clean_wing = problem["data:aerodynamics:wing:low_speed:CL_max_clean"]
    assert cl_max_clean_wing == pytest.approx(1.50, abs=1e-2)
    cl_min_clean_wing = problem["data:aerodynamics:wing:low_speed:CL_min_clean"]
    assert cl_min_clean_wing == pytest.approx(-1.20, abs=1e-2)
    cl_max_takeoff_wing = problem["data:aerodynamics:aircraft:takeoff:CL_max"]
    assert cl_max_takeoff_wing == pytest.approx(1.58, abs=1e-2)
    cl_max_landing_wing = problem["data:aerodynamics:aircraft:landing:CL_max"]
    assert cl_max_landing_wing == pytest.approx(1.87, abs=1e-2)
    cl_max_clean_htp = problem["data:aerodynamics:horizontal_tail:low_speed:CL_max_clean"]
    assert cl_max_clean_htp == pytest.approx(0.30, abs=1e-2)
    cl_min_clean_htp = problem["data:aerodynamics:horizontal_tail:low_speed:CL_min_clean"]
    assert cl_min_clean_htp == pytest.approx(-0.30, abs=1e-2)
    alpha_max_clean_htp = problem[
        "data:aerodynamics:horizontal_tail:low_speed:clean:alpha_aircraft_max"
    ]
    assert alpha_max_clean_htp == pytest.approx(30.39, abs=1e-2)
    alpha_min_clean_htp = problem[
        "data:aerodynamics:horizontal_tail:low_speed:clean:alpha_aircraft_min"
    ]
    assert alpha_min_clean_htp == pytest.approx(-30.36, abs=1e-2)


def test_l_d_max():
    """ Tests best lift/drag component """
    # Define independent input value (openVSP)
    ivc = get_indep_var_comp(list_inputs(ComputeLDMax()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeLDMax(), ivc)
    l_d_max = problem["data:aerodynamics:aircraft:cruise:L_D_max"]
    assert l_d_max == pytest.approx(15.422, abs=1e-1)
    optimal_cl = problem["data:aerodynamics:aircraft:cruise:optimal_CL"]
    assert optimal_cl == pytest.approx(0.6475, abs=1e-4)
    optimal_cd = problem["data:aerodynamics:aircraft:cruise:optimal_CD"]
    assert optimal_cd == pytest.approx(0.0419, abs=1e-4)
    optimal_alpha = problem.get_val("data:aerodynamics:aircraft:cruise:optimal_alpha", units="deg")
    assert optimal_alpha == pytest.approx(4.92, abs=1e-2)


def test_cnbeta():
    """ Tests cn beta fuselage """
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(ComputeCnBetaFuselage()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeCnBetaFuselage(), ivc)
    cn_beta_fus = problem["data:aerodynamics:fuselage:cruise:CnBeta"]
    assert cn_beta_fus == pytest.approx(-0.0557, abs=1e-4)


@pytest.mark.skipif(
    system() != "Windows" or SKIP_STEPS, reason="OPENVSP is windows dependent platform"
)
def test_slipstream_openvsp_cruise():
    # Create result temporary directory
    results_folder = _create_tmp_directory()

    # Research independent input value in .xml file
    # noinspection PyTypeChecker
    ivc = get_indep_var_comp(
        list_inputs(
            ComputeSlipstreamOpenvsp(
                propulsion_id=ENGINE_WRAPPER, result_folder_path=results_folder.name,
            )
        ),
        __file__,
        XML_FILE,
    )
    # Run problem and check obtained value(s) is/(are) correct
    # noinspection PyTypeChecker
    problem = run_system(
        ComputeSlipstreamOpenvsp(
            propulsion_id=ENGINE_WRAPPER,
            result_folder_path=results_folder.name,
            low_speed_aero=False,
        ),
        ivc,
    )
    y_vector_prop_on = problem.get_val(
        "data:aerodynamics:slipstream:wing:cruise:prop_on:Y_vector", units="m"
    )
    y_result_prop_on = np.array(
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
    assert np.max(np.abs(y_vector_prop_on - y_result_prop_on)) <= 1e-2
    cl_vector_prop_on = problem.get_val(
        "data:aerodynamics:slipstream:wing:cruise:prop_on:CL_vector"
    )
    cl_result_prop_on = np.array(
        [
            1.43,
            1.42,
            1.42,
            1.42,
            1.42,
            1.41,
            1.42,
            1.41,
            1.4,
            1.4,
            1.41,
            1.4,
            1.4,
            1.46,
            1.5,
            1.38,
            1.35,
            1.36,
            1.35,
            1.34,
            1.33,
            1.32,
            1.31,
            1.3,
            1.28,
            1.26,
            1.24,
            1.21,
            1.19,
            1.16,
            1.13,
            1.09,
            1.05,
            1.01,
            0.95,
            0.88,
            0.79,
            0.67,
            0.63,
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
    assert np.max(np.abs(cl_vector_prop_on - cl_result_prop_on)) <= 1e-2
    ct = problem.get_val("data:aerodynamics:slipstream:wing:cruise:prop_on:CT_ref")
    assert ct == pytest.approx(0.6154, abs=1e-4)
    delta_cl = problem.get_val(
        "data:aerodynamics:slipstream:wing:cruise:prop_on:CL"
    ) - problem.get_val("data:aerodynamics:slipstream:wing:cruise:prop_off:CL")
    assert delta_cl == pytest.approx(0.0004, abs=1e-4)


@pytest.mark.skipif(
    system() != "Windows" or SKIP_STEPS, reason="OPENVSP is windows dependent platform"
)
def test_slipstream_openvsp_low_speed():
    # Create result temporary directory
    results_folder = _create_tmp_directory()

    # Research independent input value in .xml file
    # noinspection PyTypeChecker
    ivc = get_indep_var_comp(
        list_inputs(
            ComputeSlipstreamOpenvsp(
                propulsion_id=ENGINE_WRAPPER,
                result_folder_path=results_folder.name,
                low_speed_aero=True,
            )
        ),
        __file__,
        XML_FILE,
    )

    # Run problem and check obtained value(s) is/(are) correct
    # noinspection PyTypeChecker
    problem = run_system(
        ComputeSlipstreamOpenvsp(
            propulsion_id=ENGINE_WRAPPER,
            result_folder_path=results_folder.name,
            low_speed_aero=True,
        ),
        ivc,
    )
    y_vector_prop_on = problem.get_val(
        "data:aerodynamics:slipstream:wing:low_speed:prop_on:Y_vector", units="m"
    )
    y_result_prop_on = np.array(
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
    assert np.max(np.abs(y_vector_prop_on - y_result_prop_on)) <= 1e-2
    cl_vector_prop_on = problem.get_val(
        "data:aerodynamics:slipstream:wing:low_speed:prop_on:CL_vector"
    )
    cl_result_prop_on = np.array(
        [
            1.45,
            1.42,
            1.42,
            1.41,
            1.41,
            1.41,
            1.41,
            1.4,
            1.4,
            1.4,
            1.4,
            1.4,
            1.39,
            1.66,
            1.79,
            1.56,
            1.49,
            1.34,
            1.33,
            1.33,
            1.32,
            1.31,
            1.3,
            1.29,
            1.27,
            1.26,
            1.23,
            1.21,
            1.18,
            1.16,
            1.13,
            1.1,
            1.05,
            1.01,
            0.95,
            0.88,
            0.79,
            0.68,
            0.63,
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
    assert np.max(np.abs(cl_vector_prop_on - cl_result_prop_on)) <= 1e-2
    ct = problem.get_val("data:aerodynamics:slipstream:wing:low_speed:prop_on:CT_ref")
    assert ct == pytest.approx(0.4837, abs=1e-4)
    delta_cl = problem.get_val(
        "data:aerodynamics:slipstream:wing:low_speed:prop_on:CL"
    ) - problem.get_val("data:aerodynamics:slipstream:wing:low_speed:prop_off:CL")
    assert delta_cl == pytest.approx(0.0088, abs=1e-4)


def test_compute_mach_interpolation_roskam():
    """ Tests computation of the mach interpolation vector using Roskam's approach """
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(ComputeMachInterpolation()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeMachInterpolation(), ivc)
    cl_alpha_vector = problem["data:aerodynamics:aircraft:mach_interpolation:CL_alpha_vector"]
    cl_alpha_result = np.array([5.33, 5.35, 5.42, 5.54, 5.72, 5.96])
    assert np.max(np.abs(cl_alpha_vector - cl_alpha_result)) <= 1e-2
    mach_vector = problem["data:aerodynamics:aircraft:mach_interpolation:mach_vector"]
    mach_result = np.array([0.0, 0.08, 0.15, 0.23, 0.31, 0.39])
    assert np.max(np.abs(mach_vector - mach_result)) <= 1e-2


def test_cl_alpha_vt():
    """ Tests Cl alpha vt """
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(ComputeClAlphaVT(low_speed_aero=True)), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeClAlphaVT(low_speed_aero=True), ivc)
    cl_alpha_vt_ls = problem.get_val(
        "data:aerodynamics:vertical_tail:low_speed:CL_alpha", units="rad**-1"
    )
    assert cl_alpha_vt_ls == pytest.approx(2.6812, abs=1e-4)
    k_ar_effective = problem.get_val("data:aerodynamics:vertical_tail:k_ar_effective")
    assert k_ar_effective == pytest.approx(1.8630, abs=1e-4)

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(ComputeClAlphaVT()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeClAlphaVT(), ivc)
    cl_alpha_vt_cruise = problem.get_val(
        "data:aerodynamics:vertical_tail:cruise:CL_alpha", units="rad**-1"
    )
    assert cl_alpha_vt_cruise == pytest.approx(2.7321, abs=1e-4)


def test_cy_delta_r():
    """ Tests cy delta of the rudder """
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(ComputeCyDeltaRudder()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeCyDeltaRudder(), ivc)
    cy_delta_r = problem.get_val("data:aerodynamics:rudder:low_speed:Cy_delta_r", units="rad**-1")
    assert cy_delta_r == pytest.approx(1.8882, abs=1e-4)


@pytest.mark.skipif(
    system() != "Windows" or SKIP_STEPS, reason="OPENVSP is windows dependent platform"
)
def test_high_speed_connection_openvsp():
    """ Tests high speed components connection """
    # load all inputs
    ivc = get_indep_var_comp(
        list_inputs(AerodynamicsHighSpeed(propulsion_id=ENGINE_WRAPPER, use_openvsp=True)),
        __file__,
        XML_FILE,
    )

    # noinspection PyTypeChecker
    run_system(AerodynamicsHighSpeed(propulsion_id=ENGINE_WRAPPER, use_openvsp=True), ivc)


@pytest.mark.skipif(SKIP_STEPS, reason="Skip test because already performed on Cirrus")
def test_high_speed_connection_vlm():
    """ Tests high speed components connection """
    # load all inputs
    ivc = get_indep_var_comp(
        list_inputs(AerodynamicsHighSpeed(propulsion_id=ENGINE_WRAPPER, use_openvsp=False)),
        __file__,
        XML_FILE,
    )

    # noinspection PyTypeChecker
    run_system(AerodynamicsHighSpeed(propulsion_id=ENGINE_WRAPPER, use_openvsp=False), ivc)


@pytest.mark.skipif(
    system() != "Windows" or SKIP_STEPS, reason="OPENVSP is windows dependent platform"
)
def test_low_speed_connection_openvsp():
    """ Tests low speed components connection """
    # load all inputs
    ivc = get_indep_var_comp(
        list_inputs(AerodynamicsLowSpeed(propulsion_id=ENGINE_WRAPPER, use_openvsp=True)),
        __file__,
        XML_FILE,
    )

    # noinspection PyTypeChecker
    run_system(AerodynamicsLowSpeed(propulsion_id=ENGINE_WRAPPER, use_openvsp=True), ivc)


@pytest.mark.skipif(SKIP_STEPS, reason="Skip test because already performed on Cirrus")
def test_low_speed_connection_vlm():
    """ Tests low speed components connection """
    # load all inputs
    ivc = get_indep_var_comp(
        list_inputs(AerodynamicsLowSpeed(propulsion_id=ENGINE_WRAPPER, use_openvsp=False)),
        __file__,
        XML_FILE,
    )

    # noinspection PyTypeChecker
    run_system(AerodynamicsLowSpeed(propulsion_id=ENGINE_WRAPPER, use_openvsp=False), ivc)


@pytest.mark.skipif(SKIP_STEPS, reason="Skip test because already performed on Cirrus")
def test_v_n_diagram():
    # load all inputs
    ivc = get_indep_var_comp(
        list_inputs(ComputeVNAndVH(propulsion_id=ENGINE_WRAPPER)), __file__, XML_FILE
    )
    # Run problem with VLM and check obtained value(s) is/(are) correct
    # noinspection PyTypeChecker
    problem = run_system(ComputeVNAndVH(propulsion_id=ENGINE_WRAPPER), ivc)
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
    velocity_array = problem.get_val("data:flight_domain:velocity", units="m/s")
    load_factor_array = problem["data:flight_domain:load_factor"]
    assert np.max(np.abs(velocity_vect - velocity_array)) <= 1e-3
    assert np.max(np.abs(load_factor_vect - load_factor_array)) <= 1e-3


def test_load_factor():
    # load all inputs
    ivc = get_indep_var_comp(
        list_inputs(LoadFactor(propulsion_id=ENGINE_WRAPPER)), __file__, XML_FILE
    )

    problem = run_system(LoadFactor(propulsion_id=ENGINE_WRAPPER), ivc)

    load_factor_ultimate = problem.get_val("data:mission:sizing:cs23:sizing_factor_ultimate")
    vh = problem.get_val("data:TLAR:v_max_sl", units="m/s")
    assert load_factor_ultimate == pytest.approx(5.7, abs=1e-1)
    assert vh == pytest.approx(102.09, abs=1e-2)


@pytest.mark.skipif(
    system() != "Windows" and xfoil_path is None, reason="No XFOIL executable available",
)
def test_propeller():
    # Transfer saved polar results to temporary folder
    # tmp_folder = polar_result_transfer()

    # load all inputs and add missing ones
    ivc = get_indep_var_comp(
        list_inputs(
            ComputePropellerPerformance(
                sections_profile_name_list=["naca4430"],
                sections_profile_position_list=[0],
                elements_number=3,
                vectors_length=7,
            )
        ),
        __file__,
        XML_FILE,
    )

    # Run problem
    problem = run_system(
        ComputePropellerPerformance(
            sections_profile_name_list=["naca4430"],
            sections_profile_position_list=[0],
            elements_number=3,
            vectors_length=7,
        ),
        ivc,
    )

    # Retrieve polar results from temporary folder
    # polar_result_retrieve(tmp_folder)

    # Check obtained value(s) is/(are) correct
    thrust_SL = np.array(
        [
            118.28404737,
            343.04728219,
            567.810517,
            792.57375182,
            1017.33698664,
            1242.10022145,
            1466.86345627,
            1691.62669108,
            1916.3899259,
            2141.15316071,
            2365.91639553,
            2590.67963035,
            2815.44286516,
            3040.20609998,
            3264.96933479,
            3489.73256961,
            3714.49580442,
            3939.25903924,
            4164.02227405,
            4388.78550887,
            4613.54874369,
            4838.3119785,
            5063.07521332,
            5287.83844813,
            5512.60168295,
            5737.36491776,
            5962.12815258,
            6186.89138739,
            6411.65462221,
            6636.41785703,
        ]
    )
    trust_SL_limit = np.array(
        [
            4208.4616783,
            4505.40962551,
            4789.77591159,
            5058.48426464,
            5306.73819484,
            5543.13586359,
            5794.74759939,
            6060.43907064,
            6338.4260971,
            6636.41785703,
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
    efficiency_SL = np.array(
        [
            [
                0.05822727,
                0.13977398,
                0.18159523,
                0.19849126,
                0.2018458,
                0.1990279,
                0.19295559,
                0.18574411,
                0.1781099,
                0.17047836,
                0.16317614,
                0.15608156,
                0.14922972,
                0.14273312,
                0.13636042,
                0.1301907,
                0.12361888,
                0.11615645,
                0.10360319,
                0.09637868,
                0.09637868,
                0.09637868,
                0.09637868,
                0.09637868,
                0.09637868,
                0.09637868,
                0.09637868,
                0.09637868,
                0.09637868,
                0.09637868,
            ],
            [
                0.16440952,
                0.34718785,
                0.43041829,
                0.4636416,
                0.47226257,
                0.4698733,
                0.46045339,
                0.44883606,
                0.43531374,
                0.42188159,
                0.40813311,
                0.39476957,
                0.38164161,
                0.36868179,
                0.35626102,
                0.34406412,
                0.33189893,
                0.31952329,
                0.30582993,
                0.28635943,
                0.26619418,
                0.26619418,
                0.26619418,
                0.26619418,
                0.26619418,
                0.26619418,
                0.26619418,
                0.26619418,
                0.26619418,
                0.26619418,
            ],
            [
                0.24784447,
                0.47373779,
                0.56486852,
                0.60220775,
                0.61475448,
                0.61476364,
                0.60799142,
                0.5977694,
                0.58543972,
                0.57223204,
                0.55854393,
                0.54485208,
                0.53113312,
                0.51753348,
                0.5039672,
                0.4905188,
                0.4774016,
                0.46415346,
                0.45053412,
                0.43546808,
                0.41555585,
                0.38462898,
                0.38462898,
                0.38462898,
                0.38462898,
                0.38462898,
                0.38462898,
                0.38462898,
                0.38462898,
                0.38462898,
            ],
            [
                0.30608854,
                0.54429969,
                0.63616786,
                0.67504266,
                0.69043073,
                0.6929904,
                0.69014147,
                0.6827371,
                0.67307863,
                0.66239424,
                0.65074542,
                0.63888842,
                0.6265992,
                0.6143684,
                0.60191776,
                0.58944273,
                0.57698492,
                0.56462733,
                0.55200457,
                0.53874821,
                0.52423645,
                0.50503998,
                0.46943816,
                0.46943816,
                0.46943816,
                0.46943816,
                0.46943816,
                0.46943816,
                0.46943816,
                0.46943816,
            ],
            [
                0.34007528,
                0.58054599,
                0.67376707,
                0.71356005,
                0.73165884,
                0.73727781,
                0.73684619,
                0.73268604,
                0.72597082,
                0.71786017,
                0.70872569,
                0.69906156,
                0.68881949,
                0.67841664,
                0.66773992,
                0.65688926,
                0.64579144,
                0.63476471,
                0.62374823,
                0.61226421,
                0.59997916,
                0.58634737,
                0.56873011,
                0.53566428,
                0.5315517,
                0.5315517,
                0.5315517,
                0.5315517,
                0.5315517,
                0.5315517,
            ],
            [
                0.34307711,
                0.59743354,
                0.68946849,
                0.73294957,
                0.75358771,
                0.76278853,
                0.76519866,
                0.7635913,
                0.75924055,
                0.75331617,
                0.74643723,
                0.73881224,
                0.73061866,
                0.72206102,
                0.71313835,
                0.70394178,
                0.69452939,
                0.68501267,
                0.6752966,
                0.66543289,
                0.65516064,
                0.64401444,
                0.63164348,
                0.61538335,
                0.58596082,
                0.57801417,
                0.57801417,
                0.57801417,
                0.57801417,
                0.57801417,
            ],
            [
                0.34112256,
                0.59956891,
                0.69591324,
                0.74168853,
                0.76453618,
                0.77660405,
                0.78191473,
                0.78259502,
                0.78106457,
                0.777386,
                0.772183,
                0.766279,
                0.759598,
                0.75272135,
                0.7453902,
                0.73775181,
                0.72985757,
                0.72171251,
                0.71344196,
                0.70489371,
                0.69620055,
                0.68691217,
                0.67678873,
                0.66564501,
                0.6507914,
                0.62568955,
                0.61369045,
                0.61369045,
                0.61369045,
                0.61369045,
            ],
            [
                0.34829032,
                0.59717235,
                0.69564344,
                0.74376438,
                0.76999397,
                0.7840045,
                0.79149927,
                0.79475601,
                0.79488539,
                0.7933805,
                0.79022314,
                0.78593621,
                0.78095263,
                0.77503085,
                0.76905952,
                0.76271836,
                0.75612639,
                0.74927534,
                0.74223533,
                0.73499466,
                0.72745938,
                0.71972347,
                0.71143357,
                0.70209462,
                0.69206076,
                0.67846608,
                0.6572271,
                0.64146696,
                0.64146696,
                0.64146696,
            ],
            [
                0.32471476,
                0.58635818,
                0.69039558,
                0.74192999,
                0.7712092,
                0.78798684,
                0.79738905,
                0.80228425,
                0.80432569,
                0.80415267,
                0.80281659,
                0.80005516,
                0.7963294,
                0.79217418,
                0.78707586,
                0.78177645,
                0.77622345,
                0.77040122,
                0.76451095,
                0.7582906,
                0.75198211,
                0.74523106,
                0.7382862,
                0.7309323,
                0.72253913,
                0.71336896,
                0.70163365,
                0.68447983,
                0.663251,
                0.663251,
            ],
            [
                0.32060259,
                0.57825861,
                0.68342211,
                0.73802561,
                0.76980819,
                0.78909635,
                0.80050975,
                0.80718087,
                0.81063591,
                0.81174742,
                0.81157086,
                0.81028252,
                0.80785159,
                0.80471234,
                0.80107565,
                0.79676752,
                0.79205362,
                0.78712525,
                0.78197697,
                0.77675869,
                0.77131429,
                0.76570256,
                0.75976232,
                0.75338828,
                0.74681263,
                0.73940135,
                0.73114866,
                0.72130738,
                0.70767664,
                0.68050973,
            ],
        ]
    )
    assert (
        np.sum(
            np.abs(
                thrust_SL
                - problem.get_val("data:aerodynamics:propeller:sea_level:thrust", units="N")
            )
        )
        < 1
    )
    assert (
        np.sum(
            np.abs(
                trust_SL_limit
                - problem.get_val("data:aerodynamics:propeller:sea_level:thrust_limit", units="N")
            )
        )
        < 1
    )
    assert (
        np.sum(
            np.abs(
                speed - problem.get_val("data:aerodynamics:propeller:sea_level:speed", units="m/s")
            )
        )
        < 1e-2
    )
    assert (
        np.sum(np.abs(efficiency_SL - problem["data:aerodynamics:propeller:sea_level:efficiency"]))
        < 1e-5
    )
    thrust_CL = np.array(
        [
            93.15819825,
            272.74322548,
            452.3282527,
            631.91327993,
            811.49830716,
            991.08333439,
            1170.66836162,
            1350.25338884,
            1529.83841607,
            1709.4234433,
            1889.00847053,
            2068.59349776,
            2248.17852498,
            2427.76355221,
            2607.34857944,
            2786.93360667,
            2966.5186339,
            3146.10366112,
            3325.68868835,
            3505.27371558,
            3684.85874281,
            3864.44377003,
            4044.02879726,
            4223.61382449,
            4403.19885172,
            4582.78387895,
            4762.36890617,
            4941.9539334,
            5121.53896063,
            5301.12398786,
        ]
    )
    trust_CL_limit = np.array(
        [
            3332.22612057,
            3568.5270316,
            3795.79754353,
            4010.15603254,
            4211.1313009,
            4403.77876683,
            4607.16086781,
            4825.00322804,
            5052.98188671,
            5301.12398786,
        ]
    )
    efficiency_CL = np.array(
        [
            [
                0.05832829,
                0.14079049,
                0.18250237,
                0.19889311,
                0.20185194,
                0.19864187,
                0.19232217,
                0.18489622,
                0.17713393,
                0.169383,
                0.16202158,
                0.15487315,
                0.14799255,
                0.14147145,
                0.1350509,
                0.12884379,
                0.12218142,
                0.1143114,
                0.098278,
                0.09629643,
                0.09629643,
                0.09629643,
                0.09629643,
                0.09629643,
                0.09629643,
                0.09629643,
                0.09629643,
                0.09629643,
                0.09629643,
                0.09629643,
            ],
            [
                0.16464549,
                0.34937694,
                0.4320869,
                0.46456366,
                0.47241919,
                0.46932065,
                0.45942554,
                0.44737059,
                0.43355694,
                0.41984293,
                0.40593461,
                0.39239829,
                0.37916171,
                0.36607983,
                0.35356898,
                0.34131591,
                0.32908306,
                0.31649795,
                0.30220592,
                0.28047326,
                0.26582644,
                0.26582644,
                0.26582644,
                0.26582644,
                0.26582644,
                0.26582644,
                0.26582644,
                0.26582644,
                0.26582644,
                0.26582644,
            ],
            [
                0.24807387,
                0.47626515,
                0.56688916,
                0.60334754,
                0.6150433,
                0.61437775,
                0.60704808,
                0.5963657,
                0.58367032,
                0.5701879,
                0.55623785,
                0.54233945,
                0.52843778,
                0.51468738,
                0.50095422,
                0.48741274,
                0.47419021,
                0.46079011,
                0.44699676,
                0.43163168,
                0.40985175,
                0.38410368,
                0.38410368,
                0.38410368,
                0.38410368,
                0.38410368,
                0.38410368,
                0.38410368,
                0.38410368,
                0.38410368,
            ],
            [
                0.30685727,
                0.54665378,
                0.63797837,
                0.67605774,
                0.69082869,
                0.69276381,
                0.68945124,
                0.68159086,
                0.67158595,
                0.66056794,
                0.64868949,
                0.63657521,
                0.62409671,
                0.61166259,
                0.59908618,
                0.58639865,
                0.57391603,
                0.56144183,
                0.54859472,
                0.53520903,
                0.52011367,
                0.50012129,
                0.46878648,
                0.46878648,
                0.46878648,
                0.46878648,
                0.46878648,
                0.46878648,
                0.46878648,
                0.46878648,
            ],
            [
                0.340129,
                0.58270009,
                0.67554724,
                0.7145794,
                0.7321444,
                0.7371614,
                0.73633602,
                0.73175949,
                0.72473221,
                0.71632368,
                0.70692595,
                0.69705051,
                0.68658939,
                0.67602641,
                0.66515926,
                0.65418825,
                0.64290923,
                0.63181933,
                0.62065987,
                0.60898019,
                0.59660921,
                0.58251464,
                0.56399548,
                0.53084885,
                0.53084885,
                0.53084885,
                0.53084885,
                0.53084885,
                0.53084885,
                0.53084885,
            ],
            [
                0.34184802,
                0.59950129,
                0.69104455,
                0.73395353,
                0.75405815,
                0.76278581,
                0.7648174,
                0.7628961,
                0.75819446,
                0.75201648,
                0.74490195,
                0.73705107,
                0.72867309,
                0.71993475,
                0.71084771,
                0.70150699,
                0.69196096,
                0.68227681,
                0.67252741,
                0.6625356,
                0.65207525,
                0.64079986,
                0.62818434,
                0.61097824,
                0.57760451,
                0.57729546,
                0.57729546,
                0.57729546,
                0.57729546,
                0.57729546,
            ],
            [
                0.3412724,
                0.60122211,
                0.69728334,
                0.74265334,
                0.76500517,
                0.77666731,
                0.78165439,
                0.78201356,
                0.78021451,
                0.77626877,
                0.7708136,
                0.76475612,
                0.75787127,
                0.75083939,
                0.74335507,
                0.73557937,
                0.72755363,
                0.71928625,
                0.71088322,
                0.70226482,
                0.69347577,
                0.68404099,
                0.67383429,
                0.66250047,
                0.64697365,
                0.62053704,
                0.61292161,
                0.61292161,
                0.61292161,
                0.61292161,
            ],
            [
                0.35036934,
                0.59905036,
                0.69698069,
                0.74466696,
                0.77044277,
                0.78410506,
                0.7912957,
                0.79426072,
                0.79414541,
                0.7924214,
                0.78904834,
                0.78457615,
                0.77938224,
                0.77333485,
                0.76721963,
                0.76076506,
                0.7540438,
                0.74707494,
                0.73994021,
                0.73259243,
                0.72498673,
                0.7171197,
                0.70878265,
                0.69934165,
                0.68926762,
                0.67541168,
                0.65337175,
                0.64065639,
                0.64065639,
                0.64065639,
            ],
            [
                0.32469377,
                0.5876176,
                0.69155958,
                0.74273127,
                0.7716199,
                0.78808525,
                0.79719517,
                0.8018334,
                0.80365341,
                0.80326908,
                0.80174261,
                0.79880096,
                0.79492425,
                0.7906093,
                0.78537555,
                0.77997202,
                0.77430827,
                0.76838767,
                0.76239203,
                0.75607923,
                0.74968677,
                0.7428594,
                0.73586644,
                0.72842889,
                0.72003172,
                0.71087646,
                0.6990703,
                0.68166696,
                0.66238912,
                0.66238912,
            ],
            [
                0.3145431,
                0.57867382,
                0.68424655,
                0.73867142,
                0.77014463,
                0.78915803,
                0.80029181,
                0.80674024,
                0.80998128,
                0.81092258,
                0.81055898,
                0.80911526,
                0.80654143,
                0.80325933,
                0.79950918,
                0.79507275,
                0.79024699,
                0.78523908,
                0.77999356,
                0.77469879,
                0.76916701,
                0.7634865,
                0.75749943,
                0.75109724,
                0.7444907,
                0.73707485,
                0.72888744,
                0.71917273,
                0.70566043,
                0.67963045,
            ],
        ]
    )
    assert (
        np.sum(
            np.abs(
                thrust_CL
                - problem.get_val("data:aerodynamics:propeller:cruise_level:thrust", units="N")
            )
        )
        < 1
    )
    assert (
        np.sum(
            np.abs(
                trust_CL_limit
                - problem.get_val(
                    "data:aerodynamics:propeller:cruise_level:thrust_limit", units="N"
                )
            )
        )
        < 1
    )
    assert (
        np.sum(
            np.abs(
                speed
                - problem.get_val("data:aerodynamics:propeller:cruise_level:speed", units="m/s")
            )
        )
        < 1e-2
    )
    assert (
        np.sum(
            np.abs(efficiency_CL - problem["data:aerodynamics:propeller:cruise_level:efficiency"])
        )
        < 1e-5
    )
