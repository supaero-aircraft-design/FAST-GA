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
import openmdao.api as om
import numpy as np
from platform import system
import tempfile
from pathlib import Path
from tempfile import TemporaryDirectory
import pytest
import time

from fastoad.io import VariableIO

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
from ..constants import SPAN_MESH_POINT
from ..components.compute_propeller_aero import ComputePropellerPerformance

from tests.testing_utilities import run_system, get_indep_var_comp, list_inputs
from tests.xfoil_exe.get_xfoil import get_xfoil_path

from .dummy_engines import ENGINE_WRAPPER_BE76 as ENGINE_WRAPPER

RESULTS_FOLDER = pth.join(pth.dirname(__file__), "results")
TMP_SAVE_FOLDER = "test_save"
xfoil_path = None if system() == "Windows" else get_xfoil_path()

XML_FILE = "beechcraft_76.xml"


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
    # Put saved polar results in a temporary folder to activate Xfoil run and have repeatable results  [need writting
    # permission]

    tmp_folder = _create_tmp_directory()

    files = glob.iglob(pth.join(resources.__path__[0], "*.csv"))

    for file in files:
        if os.path.isfile(file):
            shutil.copy(file, tmp_folder.name)
            try:
                os.remove(file)
            except:
                pass

    return tmp_folder


def polar_result_retrieve(tmp_folder):
    # Retrieve the polar results set aside during the test duration if there are some [need writting permission]

    files = glob.iglob(pth.join(tmp_folder.name, "*.csv"))

    for file in files:
        if os.path.isfile(file):
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
    system() != "Windows" and xfoil_path is None, reason="No XFOIL executable available"
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
    system() != "Windows" and xfoil_path is None, reason="No XFOIL executable available"
)
def test_airfoil_slope():
    """ Tests polar execution (XFOIL) @ high and low speed """

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


@pytest.mark.skipif(system() != "Windows", reason="OPENVSP is windows dependent platform")
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


@pytest.mark.skipif(system() != "Windows", reason="OPENVSP is windows dependent platform")
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
    # assert cl_max_clean_wing == pytest.approx(1.50, abs=1e-2)
    cl_min_clean_wing = problem["data:aerodynamics:wing:low_speed:CL_min_clean"]
    # assert cl_min_clean_wing == pytest.approx(-1.20, abs=1e-2)
    cl_max_takeoff_wing = problem["data:aerodynamics:aircraft:takeoff:CL_max"]
    # assert cl_max_takeoff_wing == pytest.approx(1.58, abs=1e-2)
    cl_max_landing_wing = problem["data:aerodynamics:aircraft:landing:CL_max"]
    # assert cl_max_landing_wing == pytest.approx(1.87, abs=1e-2)
    cl_max_clean_htp = problem["data:aerodynamics:horizontal_tail:low_speed:CL_max_clean"]
    # assert cl_max_clean_htp == pytest.approx(0.30, abs=1e-2)
    cl_min_clean_htp = problem["data:aerodynamics:horizontal_tail:low_speed:CL_min_clean"]
    # assert cl_min_clean_htp == pytest.approx(-0.30, abs=1e-2)
    alpha_max_clean_htp = problem[
        "data:aerodynamics:horizontal_tail:low_speed:clean:alpha_aircraft_max"
    ]
    # assert alpha_max_clean_htp == pytest.approx(30.39, abs=1e-2)
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


@pytest.mark.skipif(system() != "Windows", reason="OPENVSP is windows dependent platform")
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


@pytest.mark.skipif(system() != "Windows", reason="OPENVSP is windows dependent platform")
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


@pytest.mark.skipif(system() != "Windows", reason="OPENVSP is windows dependent platform")
def test_high_speed_connection():
    """ Tests high speed components connection """

    # load all inputs
    ivc_vlm = get_indep_var_comp(
        list_inputs(AerodynamicsHighSpeed(propulsion_id=ENGINE_WRAPPER, use_openvsp=False)),
        __file__,
        XML_FILE,
    )

    # Run problem with VLM
    # noinspection PyTypeChecker
    run_system(AerodynamicsHighSpeed(propulsion_id=ENGINE_WRAPPER, use_openvsp=False), ivc_vlm)

    # load all inputs
    ivc_openvsp = get_indep_var_comp(
        list_inputs(AerodynamicsHighSpeed(propulsion_id=ENGINE_WRAPPER, use_openvsp=True)),
        __file__,
        XML_FILE,
    )

    # Run problem with OPENVSP
    # noinspection PyTypeChecker
    run_system(AerodynamicsHighSpeed(propulsion_id=ENGINE_WRAPPER, use_openvsp=True), ivc_openvsp)


@pytest.mark.skipif(system() != "Windows", reason="OPENVSP is windows dependent platform")
def test_low_speed_connection():
    """ Tests low speed components connection """

    # load all inputs
    ivc_vlm = get_indep_var_comp(
        list_inputs(AerodynamicsLowSpeed(propulsion_id=ENGINE_WRAPPER, use_openvsp=False)),
        __file__,
        XML_FILE,
    )

    # Run problem with VLM
    # noinspection PyTypeChecker
    run_system(AerodynamicsLowSpeed(propulsion_id=ENGINE_WRAPPER, use_openvsp=False), ivc_vlm)

    # load all inputs
    ivc_openvsp = get_indep_var_comp(
        list_inputs(AerodynamicsLowSpeed(propulsion_id=ENGINE_WRAPPER, use_openvsp=True)),
        __file__,
        XML_FILE,
    )

    # Run problem with OPENVSP
    # noinspection PyTypeChecker
    run_system(AerodynamicsLowSpeed(propulsion_id=ENGINE_WRAPPER, use_openvsp=True), ivc_openvsp)


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


def test_propeller():
    # Transfer saved polar results to temporary folder
    tmp_folder = polar_result_transfer()

    # load all inputs and add missing ones
    ivc = get_indep_var_comp(
        list_inputs(ComputePropellerPerformance(vectors_length=7)), __file__, XML_FILE
    )
    twist_law = lambda x: 25.0 / (x + 0.75) ** 1.5 + 46.3917 - 15.0
    radius_ratio = [0.165, 0.3, 0.45, 0.655, 0.835, 0.975, 1.0]
    chord = list(
        np.array(
            [0.11568421, 0.16431579, 0.16844211, 0.21957895, 0.19231579, 0.11568421, 0.11568421]
        )
        * 0.965
    )
    sweep = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    # Run problem
    problem = run_system(ComputePropellerPerformance(vectors_length=7), ivc)

    # Retrieve polar results from temporary folder
    polar_result_retrieve(tmp_folder)

    # Check obtained value(s) is/(are) correct
    thrust_SL = np.array(
        [
            170.38807199,
            403.1482426,
            635.90841322,
            868.66858383,
            1101.42875445,
            1334.18892506,
            1566.94909568,
            1799.70926629,
            2032.46943691,
            2265.22960752,
            2497.98977814,
            2730.74994875,
            2963.51011937,
            3196.27028998,
            3429.0304606,
            3661.79063121,
            3894.55080183,
            4127.31097244,
            4360.07114306,
            4592.83131367,
            4825.59148429,
            5058.3516549,
            5291.11182551,
            5523.87199613,
            5756.63216674,
            5989.39233736,
            6222.15250797,
            6454.91267859,
            6687.6728492,
            6920.43301982,
        ]
    )
    trust_SL_limit = np.array(
        [
            4232.24869826,
            4675.80705472,
            5017.41035873,
            5307.96932824,
            5571.20112011,
            5821.98270314,
            6074.72689373,
            6337.37179199,
            6616.05323244,
            6920.43301982,
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
                0.10861576,
                0.18982678,
                0.22232571,
                0.22798754,
                0.22249053,
                0.21279113,
                0.20189521,
                0.19120108,
                0.18128458,
                0.17180358,
                0.16248651,
                0.15387892,
                0.14576032,
                0.13813155,
                0.13141977,
                0.12541624,
                0.11953427,
                0.11334617,
                0.10985857,
                0.10985857,
                0.10985857,
                0.10985857,
                0.10985857,
                0.10985857,
                0.10985857,
                0.10985857,
                0.10985857,
                0.10985857,
                0.10985857,
                0.10985857,
            ],
            [
                0.27474597,
                0.44722469,
                0.50622668,
                0.51877779,
                0.51187894,
                0.49691021,
                0.4790105,
                0.46067864,
                0.44311,
                0.42621449,
                0.40889781,
                0.39227945,
                0.37635993,
                0.36085742,
                0.34603096,
                0.33299526,
                0.32111602,
                0.30952116,
                0.29767025,
                0.28328091,
                0.27487907,
                0.27487907,
                0.27487907,
                0.27487907,
                0.27487907,
                0.27487907,
                0.27487907,
                0.27487907,
                0.27487907,
                0.27487907,
            ],
            [
                0.3928242,
                0.5844404,
                0.64560849,
                0.66085426,
                0.65725526,
                0.645226,
                0.62926425,
                0.61188304,
                0.59510835,
                0.57906898,
                0.56154161,
                0.5439542,
                0.52714661,
                0.51072498,
                0.49377803,
                0.47781239,
                0.46400537,
                0.45088986,
                0.43803787,
                0.42504527,
                0.40982043,
                0.38993429,
                0.38993429,
                0.38993429,
                0.38993429,
                0.38993429,
                0.38993429,
                0.38993429,
                0.38993429,
                0.38993429,
            ],
            [
                0.45684056,
                0.6511048,
                0.70995559,
                0.72889744,
                0.72995114,
                0.72266921,
                0.7110258,
                0.69731743,
                0.68309761,
                0.67022585,
                0.65540284,
                0.63956046,
                0.62415575,
                0.60899795,
                0.5940393,
                0.57783077,
                0.56340177,
                0.55054832,
                0.5380032,
                0.52562757,
                0.51315258,
                0.49876788,
                0.47815225,
                0.47191655,
                0.47191655,
                0.47191655,
                0.47191655,
                0.47191655,
                0.47191655,
                0.47191655,
            ],
            [
                0.47973531,
                0.67303155,
                0.73648523,
                0.75943227,
                0.76569156,
                0.76315126,
                0.75603738,
                0.74642172,
                0.73583803,
                0.72574917,
                0.71408422,
                0.70092031,
                0.68765819,
                0.67438753,
                0.66141168,
                0.64768801,
                0.6336876,
                0.62129841,
                0.60971625,
                0.59831839,
                0.58701379,
                0.57551116,
                0.56253463,
                0.54478667,
                0.53662495,
                0.53662495,
                0.53662495,
                0.53662495,
                0.53662495,
                0.53662495,
            ],
            [
                0.47820333,
                0.67268401,
                0.74186958,
                0.77077151,
                0.78181868,
                0.78384952,
                0.78095737,
                0.77516892,
                0.76820711,
                0.76067688,
                0.75160829,
                0.74118479,
                0.7302503,
                0.71912998,
                0.70804052,
                0.69652612,
                0.68437405,
                0.67285824,
                0.66211318,
                0.65180817,
                0.64164789,
                0.6314973,
                0.6211347,
                0.60943629,
                0.59406525,
                0.58553265,
                0.58553265,
                0.58553265,
                0.58553265,
                0.58553265,
            ],
            [
                0.4694931,
                0.66354609,
                0.73749253,
                0.77135476,
                0.78717555,
                0.79348754,
                0.79438574,
                0.79204691,
                0.78836287,
                0.78323396,
                0.77637652,
                0.76834431,
                0.75958264,
                0.75045045,
                0.7411742,
                0.73155028,
                0.72125957,
                0.71116024,
                0.70150259,
                0.69217348,
                0.68300382,
                0.67395216,
                0.66488952,
                0.65562193,
                0.64526058,
                0.6319373,
                0.62059589,
                0.62059589,
                0.62059589,
                0.62059589,
            ],
            [
                0.43766433,
                0.64556516,
                0.72706558,
                0.76647706,
                0.78675931,
                0.79690026,
                0.8011338,
                0.80197639,
                0.80079957,
                0.79790512,
                0.79310444,
                0.78710318,
                0.7802845,
                0.77298539,
                0.76535297,
                0.75735776,
                0.74878319,
                0.74006098,
                0.73157912,
                0.72327287,
                0.71513252,
                0.70702047,
                0.69892821,
                0.69080801,
                0.68253084,
                0.67336879,
                0.66204381,
                0.65035079,
                0.65035079,
                0.65035079,
            ],
            [
                0.42000531,
                0.63044836,
                0.71586093,
                0.75916812,
                0.78324179,
                0.79674113,
                0.80396604,
                0.80752501,
                0.80856957,
                0.80753462,
                0.80457486,
                0.80030257,
                0.79519898,
                0.78951211,
                0.78337369,
                0.77682643,
                0.76975103,
                0.7623188,
                0.75494583,
                0.74767725,
                0.74044563,
                0.73321147,
                0.7260436,
                0.71884023,
                0.71155355,
                0.70413663,
                0.69606953,
                0.68649674,
                0.67404597,
                0.67404597,
            ],
            [
                0.39460554,
                0.61021107,
                0.70297437,
                0.75079303,
                0.77806416,
                0.79453442,
                0.80445192,
                0.81032457,
                0.81324017,
                0.81377403,
                0.81249575,
                0.80979879,
                0.80615478,
                0.80185828,
                0.79705328,
                0.79177447,
                0.786021,
                0.77982679,
                0.77343537,
                0.76707201,
                0.76074805,
                0.7543704,
                0.74794582,
                0.74152222,
                0.73508202,
                0.72856648,
                0.72189957,
                0.71479531,
                0.70668236,
                0.69314684,
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
            133.83908203,
            319.93750135,
            506.03592067,
            692.13433999,
            878.23275931,
            1064.33117863,
            1250.42959795,
            1436.52801727,
            1622.62643659,
            1808.72485591,
            1994.82327523,
            2180.92169455,
            2367.02011386,
            2553.11853318,
            2739.2169525,
            2925.31537182,
            3111.41379114,
            3297.51221046,
            3483.61062978,
            3669.7090491,
            3855.80746842,
            4041.90588774,
            4228.00430706,
            4414.10272638,
            4600.2011457,
            4786.29956502,
            4972.39798433,
            5158.49640365,
            5344.59482297,
            5530.69324229,
        ]
    )
    trust_CL_limit = np.array(
        [
            3346.37707663,
            3699.93954793,
            3974.3527498,
            4209.07389163,
            4421.63404352,
            4626.50453938,
            4833.28495571,
            5048.37435786,
            5278.83393821,
            5530.69324229,
        ]
    )
    efficiency_CL = np.array(
        [
            [
                0.10845979,
                0.19057929,
                0.2228352,
                0.22782744,
                0.2218569,
                0.211824,
                0.20071564,
                0.18990256,
                0.17988359,
                0.1703122,
                0.16096555,
                0.15234971,
                0.14421902,
                0.13662396,
                0.12999441,
                0.12398338,
                0.1180524,
                0.11162023,
                0.10948711,
                0.10948711,
                0.10948711,
                0.10948711,
                0.10948711,
                0.10948711,
                0.10948711,
                0.10948711,
                0.10948711,
                0.10948711,
                0.10948711,
                0.10948711,
            ],
            [
                0.27448267,
                0.44863321,
                0.50703125,
                0.51864198,
                0.51092395,
                0.49529503,
                0.47697227,
                0.45833466,
                0.4405272,
                0.42339266,
                0.4059229,
                0.38921066,
                0.37320841,
                0.35761731,
                0.34289476,
                0.32996139,
                0.31804167,
                0.30635879,
                0.29422248,
                0.27856101,
                0.27405889,
                0.27405889,
                0.27405889,
                0.27405889,
                0.27405889,
                0.27405889,
                0.27405889,
                0.27405889,
                0.27405889,
                0.27405889,
            ],
            [
                0.39259819,
                0.58585127,
                0.6463158,
                0.66082347,
                0.65641767,
                0.64373887,
                0.62728197,
                0.60953686,
                0.59250451,
                0.57612281,
                0.55831308,
                0.5406009,
                0.52368781,
                0.50709622,
                0.48998461,
                0.47419881,
                0.46040757,
                0.44721415,
                0.43426826,
                0.42099341,
                0.40497542,
                0.386045,
                0.386045,
                0.386045,
                0.386045,
                0.386045,
                0.386045,
                0.386045,
                0.386045,
                0.386045,
            ],
            [
                0.45642271,
                0.65232364,
                0.71069149,
                0.72882355,
                0.72929557,
                0.72142576,
                0.70931451,
                0.69525031,
                0.68075982,
                0.66755703,
                0.65241251,
                0.63639683,
                0.62085029,
                0.60555933,
                0.59039151,
                0.57409584,
                0.55974679,
                0.54686726,
                0.53421208,
                0.52173552,
                0.5090597,
                0.49391862,
                0.47069252,
                0.47069252,
                0.47069252,
                0.47069252,
                0.47069252,
                0.47069252,
                0.47069252,
                0.47069252,
            ],
            [
                0.47918825,
                0.67403402,
                0.73706073,
                0.75945986,
                0.76520659,
                0.76214112,
                0.75464367,
                0.7446572,
                0.73374239,
                0.7234038,
                0.71144664,
                0.69807413,
                0.68465374,
                0.67124034,
                0.65811926,
                0.64417233,
                0.63015387,
                0.61776735,
                0.60610902,
                0.5946216,
                0.58318731,
                0.57150175,
                0.5579217,
                0.53770381,
                0.53521823,
                0.53521823,
                0.53521823,
                0.53521823,
                0.53521823,
                0.53521823,
            ],
            [
                0.47788522,
                0.67347403,
                0.74239202,
                0.77085303,
                0.78137805,
                0.78302764,
                0.77978951,
                0.77365341,
                0.76638261,
                0.75861084,
                0.74929559,
                0.7386755,
                0.72756861,
                0.71629176,
                0.70505362,
                0.69338048,
                0.68111518,
                0.66952145,
                0.6587254,
                0.64834676,
                0.63809031,
                0.62783621,
                0.61730042,
                0.6051866,
                0.5885729,
                0.58124924,
                0.58124924,
                0.58124924,
                0.58124924,
                0.58124924,
            ],
            [
                0.46976752,
                0.66426336,
                0.73790841,
                0.77137569,
                0.78682333,
                0.79278952,
                0.79337448,
                0.79070843,
                0.78671519,
                0.78135675,
                0.77433206,
                0.76611079,
                0.75718108,
                0.7478957,
                0.73847722,
                0.72871801,
                0.71830489,
                0.70809798,
                0.69836525,
                0.68894394,
                0.67968298,
                0.67054927,
                0.66139262,
                0.65198357,
                0.64132234,
                0.62731546,
                0.61896169,
                0.61896169,
                0.61896169,
                0.61896169,
            ],
            [
                0.43467397,
                0.64566803,
                0.72729423,
                0.76644958,
                0.78641957,
                0.79625643,
                0.80020835,
                0.8007301,
                0.79931546,
                0.79621631,
                0.79125658,
                0.78508131,
                0.77810308,
                0.77065719,
                0.76288588,
                0.75478453,
                0.74609589,
                0.73724764,
                0.72867094,
                0.72026303,
                0.71203178,
                0.70383833,
                0.69565882,
                0.68746224,
                0.67908459,
                0.66975356,
                0.65806016,
                0.64854424,
                0.64854424,
                0.64854424,
            ],
            [
                0.41971434,
                0.63087382,
                0.71571354,
                0.75899253,
                0.78284109,
                0.79608326,
                0.80303283,
                0.80629931,
                0.80710405,
                0.80592606,
                0.80282486,
                0.79843249,
                0.79317595,
                0.78734895,
                0.78107991,
                0.7744278,
                0.76726598,
                0.7597231,
                0.75223855,
                0.74486533,
                0.73753858,
                0.73022576,
                0.72296538,
                0.71567907,
                0.70832102,
                0.70084492,
                0.69267437,
                0.68296571,
                0.67205611,
                0.67205611,
            ],
            [
                0.39063402,
                0.61000734,
                0.7029633,
                0.75036584,
                0.77752962,
                0.7937979,
                0.80342211,
                0.809087,
                0.81176686,
                0.81222557,
                0.8108125,
                0.80800468,
                0.80422304,
                0.79979815,
                0.79486889,
                0.78949122,
                0.78365965,
                0.77737039,
                0.77088214,
                0.76441662,
                0.75799128,
                0.75152696,
                0.74501873,
                0.73852178,
                0.73201053,
                0.72541821,
                0.71870111,
                0.71156301,
                0.70342115,
                0.68860137,
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
