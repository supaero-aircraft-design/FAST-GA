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

from tests.testing_utilities import run_system, get_indep_var_comp, list_inputs
from tests.xfoil_exe.get_xfoil import get_xfoil_path

from .dummy_engines import ENGINE_WRAPPER_SR22 as ENGINE_WRAPPER

RESULTS_FOLDER = pth.join(pth.dirname(__file__), "results")
TMP_SAVE_FOLDER = "test_save"
xfoil_path = None if system() == "Windows" else get_xfoil_path()

XML_FILE = "cirrus_sr22.xml"


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
    assert mach == pytest.approx(0.1194, abs=1e-4)
    reynolds = problem.get_val("data:aerodynamics:low_speed:unit_reynolds", units="m**-1")
    assert reynolds == pytest.approx(2782216, abs=1)


def test_cd0_high_speed():
    """ Tests drag coefficient @ high speed """

    # Research independent input value in .xml file
    # noinspection PyTypeChecker
    ivc = get_indep_var_comp(list_inputs(Cd0(propulsion_id=ENGINE_WRAPPER)), __file__, XML_FILE)

    # noinspection PyTypeChecker
    problem = run_system(Cd0(propulsion_id=ENGINE_WRAPPER), ivc)
    cd0_wing = problem["data:aerodynamics:wing:cruise:CD0"]
    assert cd0_wing == pytest.approx(0.00525, abs=1e-5)
    cd0_fus = problem["data:aerodynamics:fuselage:cruise:CD0"]
    assert cd0_fus == pytest.approx(0.00579, abs=1e-5)
    cd0_ht = problem["data:aerodynamics:horizontal_tail:cruise:CD0"]
    assert cd0_ht == pytest.approx(0.00142, abs=1e-5)
    cd0_vt = problem["data:aerodynamics:vertical_tail:cruise:CD0"]
    assert cd0_vt == pytest.approx(0.00072, abs=1e-5)
    cd0_nac = problem["data:aerodynamics:nacelles:cruise:CD0"]
    assert cd0_nac == pytest.approx(0.00000, abs=1e-5)
    cd0_lg = problem["data:aerodynamics:landing_gear:cruise:CD0"]
    assert cd0_lg == pytest.approx(0.00293, abs=1e-5)
    cd0_other = problem["data:aerodynamics:other:cruise:CD0"]
    assert cd0_other == pytest.approx(0.00446, abs=1e-5)
    cd0_total = 1.25 * (cd0_wing + cd0_fus + cd0_ht + cd0_vt + cd0_nac + cd0_lg + cd0_other)
    assert cd0_total == pytest.approx(0.02573, abs=1e-5)


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
    # assert cd0_wing == pytest.approx(0.00570, abs=1e-5)
    cd0_fus = problem["data:aerodynamics:fuselage:low_speed:CD0"]
    # assert cd0_fus == pytest.approx(0.00639, abs=1e-5)
    cd0_ht = problem["data:aerodynamics:horizontal_tail:low_speed:CD0"]
    # assert cd0_ht == pytest.approx(0.00154, abs=1e-5)
    cd0_vt = problem["data:aerodynamics:vertical_tail:low_speed:CD0"]
    # assert cd0_vt == pytest.approx(0.00079, abs=1e-5)
    cd0_nac = problem["data:aerodynamics:nacelles:low_speed:CD0"]
    # assert cd0_nac == pytest.approx(0.00000, abs=1e-5)
    cd0_lg = problem["data:aerodynamics:landing_gear:low_speed:CD0"]
    # assert cd0_lg == pytest.approx(0.00293, abs=1e-5)
    cd0_other = problem["data:aerodynamics:other:low_speed:CD0"]
    # assert cd0_other == pytest.approx(0.00446, abs=1e-5)
    cd0_total = 1.25 * (cd0_wing + cd0_fus + cd0_ht + cd0_vt + cd0_nac + cd0_lg + cd0_other)
    assert cd0_total == pytest.approx(0.02729, abs=1e-5)


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

    # Run problem and check obtained value(s) is/(are) correct
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
    assert cdp_1 == pytest.approx(0.00464, abs=1e-4)

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
    assert cdp_1 == pytest.approx(0.00488, abs=1e-4)


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
                wing_airfoil_file="roncz.af",
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
            wing_airfoil_file="roncz.af",
            htp_airfoil_file="naca0012.af",
            vtp_airfoil_file="naca0012.af",
        ),
        ivc,
    )

    # Retrieve polar results from temporary folder
    polar_result_retrieve(tmp_folder)

    # Check obtained value(s) is/(are) correct
    cl_alpha_wing = problem.get_val("data:aerodynamics:wing:airfoil:CL_alpha", units="rad**-1")
    assert cl_alpha_wing == pytest.approx(6.5755, abs=1e-4)
    cl_alpha_htp = problem.get_val(
        "data:aerodynamics:horizontal_tail:airfoil:CL_alpha", units="rad**-1"
    )
    assert cl_alpha_htp == pytest.approx(6.2703, abs=1e-4)
    cl_alpha_vtp = problem.get_val(
        "data:aerodynamics:vertical_tail:airfoil:CL_alpha", units="rad**-1"
    )
    assert cl_alpha_vtp == pytest.approx(6.2703, abs=1e-4)


def test_vlm_comp_high_speed():
    """ Tests vlm f @ high speed """

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
        assert cl0_wing == pytest.approx(0.0969, abs=1e-4)
        cl_alpha_wing = problem.get_val("data:aerodynamics:wing:cruise:CL_alpha", units="rad**-1")
        assert cl_alpha_wing == pytest.approx(5.248, abs=1e-3)
        cm0 = problem["data:aerodynamics:wing:cruise:CM0_clean"]
        assert cm0 == pytest.approx(-0.0280, abs=1e-4)
        coef_k_wing = problem["data:aerodynamics:wing:cruise:induced_drag_coefficient"]
        assert coef_k_wing == pytest.approx(0.0412, abs=1e-4)
        if mach_interpolation:
            cl_alpha_vector = problem[
                "data:aerodynamics:aircraft:mach_interpolation:CL_alpha_vector"
            ]
            assert cl_alpha_vector == pytest.approx(
                [5.238, 5.238, 5.301, 5.384, 5.487, 5.609], abs=1e-2
            )
            mach_vector = problem["data:aerodynamics:aircraft:mach_interpolation:mach_vector"]
            assert mach_vector == pytest.approx([0.0, 0.15, 0.214, 0.274, 0.331, 0.385], abs=1e-2)
        cl0_htp = problem["data:aerodynamics:horizontal_tail:cruise:CL0"]
        assert cl0_htp == pytest.approx(-0.0053, abs=1e-4)
        cl_alpha_htp = problem.get_val(
            "data:aerodynamics:horizontal_tail:cruise:CL_alpha", units="rad**-1"
        )
        assert cl_alpha_htp == pytest.approx(0.5928, abs=1e-4)
        cl_alpha_htp_isolated = problem.get_val(
            "data:aerodynamics:horizontal_tail:cruise:CL_alpha_isolated", units="rad**-1"
        )
        assert cl_alpha_htp_isolated == pytest.approx(0.8839, abs=1e-4)
        coef_k_htp = problem["data:aerodynamics:horizontal_tail:cruise:induced_drag_coefficient"]
        assert coef_k_htp == pytest.approx(0.3267, abs=1e-4)
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
    assert cl0_wing == pytest.approx(0.0945, abs=1e-4)
    cl_alpha_wing = problem.get_val("data:aerodynamics:wing:low_speed:CL_alpha", units="rad**-1")
    assert cl_alpha_wing == pytest.approx(5.119, abs=1e-3)
    cm0 = problem["data:aerodynamics:wing:low_speed:CM0_clean"]
    assert cm0 == pytest.approx(-0.0273, abs=1e-4)
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
    assert coef_k_wing == pytest.approx(0.03950, abs=1e-4)
    cl0_htp = problem["data:aerodynamics:horizontal_tail:low_speed:CL0"]
    assert cl0_htp == pytest.approx(-0.0051, abs=1e-4)
    cl_ref_htp = problem["data:aerodynamics:horizontal_tail:low_speed:CL_ref"]
    assert cl_ref_htp == pytest.approx(0.0970, abs=1e-4)
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
    y, cl = reshape_curve(
        problem.get_val("data:aerodynamics:horizontal_tail:low_speed:Y_vector", "m"),
        problem["data:aerodynamics:horizontal_tail:low_speed:CL_vector"],
    )
    assert np.max(np.abs(y_vector_htp - y)) <= 1e-3
    assert np.max(np.abs(cl_vector_htp - cl)) <= 1e-3
    cl_alpha_htp = problem.get_val(
        "data:aerodynamics:horizontal_tail:low_speed:CL_alpha", units="rad**-1"
    )
    assert cl_alpha_htp == pytest.approx(0.5852, abs=1e-4)
    cl_alpha_htp_isolated = problem.get_val(
        "data:aerodynamics:horizontal_tail:low_speed:CL_alpha_isolated", units="rad**-1"
    )
    assert cl_alpha_htp_isolated == pytest.approx(0.8622, abs=1e-4)
    coef_k_htp = problem["data:aerodynamics:horizontal_tail:low_speed:induced_drag_coefficient"]
    assert coef_k_htp == pytest.approx(0.3321, abs=1e-4)
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
        assert cl0_wing == pytest.approx(0.1269, abs=1e-4)
        cl_alpha_wing = problem.get_val("data:aerodynamics:wing:cruise:CL_alpha", units="rad**-1")
        assert cl_alpha_wing == pytest.approx(5.079, abs=1e-3)
        cm0 = problem["data:aerodynamics:wing:cruise:CM0_clean"]
        assert cm0 == pytest.approx(-0.0271, abs=1e-4)
        coef_k_wing = problem["data:aerodynamics:wing:cruise:induced_drag_coefficient"]
        assert coef_k_wing == pytest.approx(0.0382, abs=1e-4)
        if mach_interpolation:
            cl_alpha_vector = problem[
                "data:aerodynamics:aircraft:mach_interpolation:CL_alpha_vector"
            ]
            assert cl_alpha_vector == pytest.approx([5.50, 5.50, 5.55, 5.61, 5.69, 5.79], abs=1e-2)
            mach_vector = problem["data:aerodynamics:aircraft:mach_interpolation:mach_vector"]
            assert mach_vector == pytest.approx([0.0, 0.15, 0.21, 0.27, 0.33, 0.38], abs=1e-2)
        cl0_htp = problem["data:aerodynamics:horizontal_tail:cruise:CL0"]
        assert cl0_htp == pytest.approx(-0.0074, abs=1e-4)
        cl_alpha_htp = problem.get_val(
            "data:aerodynamics:horizontal_tail:cruise:CL_alpha", units="rad**-1"
        )
        assert cl_alpha_htp == pytest.approx(0.5093, abs=1e-4)
        cl_alpha_htp_isolated = problem.get_val(
            "data:aerodynamics:horizontal_tail:cruise:CL_alpha_isolated", units="rad**-1"
        )
        assert cl_alpha_htp_isolated == pytest.approx(0.9395, abs=1e-4)
        coef_k_htp = problem["data:aerodynamics:horizontal_tail:cruise:induced_drag_coefficient"]
        assert coef_k_htp == pytest.approx(0.6778, abs=1e-4)
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
    assert cl0_wing == pytest.approx(0.1243, abs=1e-4)
    cl_alpha_wing = problem.get_val("data:aerodynamics:wing:low_speed:CL_alpha", units="rad**-1")
    assert cl_alpha_wing == pytest.approx(4.981, abs=1e-3)
    cm0 = problem["data:aerodynamics:wing:low_speed:CM0_clean"]
    assert cm0 == pytest.approx(-0.0265, abs=1e-4)
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
    y, cl = reshape_curve(
        problem.get_val("data:aerodynamics:wing:low_speed:Y_vector", "m"),
        problem["data:aerodynamics:wing:low_speed:CL_vector"],
    )
    assert np.max(np.abs(y_vector_wing - y)) <= 1e-2
    assert np.max(np.abs(cl_vector_wing - cl)) <= 1e-3
    coef_k_wing = problem["data:aerodynamics:wing:low_speed:induced_drag_coefficient"]
    assert coef_k_wing == pytest.approx(0.0381, abs=1e-4)
    cl0_htp = problem["data:aerodynamics:horizontal_tail:low_speed:CL0"]
    assert cl0_htp == pytest.approx(-0.0071, abs=1e-4)
    cl_ref_htp = problem["data:aerodynamics:horizontal_tail:low_speed:CL_ref"]
    assert cl_ref_htp == pytest.approx(0.081, abs=1e-4)
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
    y, cl = reshape_curve(
        problem.get_val("data:aerodynamics:horizontal_tail:low_speed:Y_vector", "m"),
        problem["data:aerodynamics:horizontal_tail:low_speed:CL_vector"],
    )
    assert np.max(np.abs(y_vector_htp - y)) <= 1e-2
    assert np.max(np.abs(cl_vector_htp - cl)) <= 1e-3
    cl_alpha_htp = problem.get_val(
        "data:aerodynamics:horizontal_tail:low_speed:CL_alpha", units="rad**-1"
    )
    assert cl_alpha_htp == pytest.approx(0.5052, abs=1e-4)
    cl_alpha_htp_isolated = problem.get_val(
        "data:aerodynamics:horizontal_tail:low_speed:CL_alpha_isolated", units="rad**-1"
    )
    assert cl_alpha_htp_isolated == pytest.approx(0.9224, abs=1e-4)
    coef_k_htp = problem["data:aerodynamics:horizontal_tail:low_speed:induced_drag_coefficient"]
    assert coef_k_htp == pytest.approx(0.6736, abs=1e-4)
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
    assert ch_alpha_2d == pytest.approx(-0.3548, abs=1e-4)
    ch_delta_2d = problem.get_val(
        "data:aerodynamics:horizontal_tail:cruise:hinge_moment:CH_delta_2D", units="rad**-1"
    )
    assert ch_delta_2d == pytest.approx(-0.5755, abs=1e-4)


def test_3d_hinge_moment():
    """ Tests tail hinge-moments """

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(Compute3DHingeMomentsTail()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(Compute3DHingeMomentsTail(), ivc)
    ch_alpha = problem.get_val(
        "data:aerodynamics:horizontal_tail:cruise:hinge_moment:CH_alpha", units="rad**-1"
    )
    assert ch_alpha == pytest.approx(-0.2594, abs=1e-4)
    ch_delta = problem.get_val(
        "data:aerodynamics:horizontal_tail:cruise:hinge_moment:CH_delta", units="rad**-1"
    )
    assert ch_delta == pytest.approx(-0.6216, abs=1e-4)


def test_high_lift():
    """ Tests high-lift contribution """

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(ComputeDeltaHighLift()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeDeltaHighLift(), ivc)
    delta_cl0_landing = problem["data:aerodynamics:flaps:landing:CL"]
    assert delta_cl0_landing == pytest.approx(0.6167, abs=1e-4)
    delta_clmax_landing = problem["data:aerodynamics:flaps:landing:CL_max"]
    assert delta_clmax_landing == pytest.approx(0.4194, abs=1e-4)
    delta_cm_landing = problem["data:aerodynamics:flaps:landing:CM"]
    assert delta_cm_landing == pytest.approx(-0.1049, abs=1e-4)
    delta_cd_landing = problem["data:aerodynamics:flaps:landing:CD"]
    assert delta_cd_landing == pytest.approx(0.0136, abs=1e-4)
    delta_cl0_takeoff = problem["data:aerodynamics:flaps:takeoff:CL"]
    assert delta_cl0_takeoff == pytest.approx(0.2300, abs=1e-4)
    delta_clmax_takeoff = problem["data:aerodynamics:flaps:takeoff:CL_max"]
    assert delta_clmax_takeoff == pytest.approx(0.0858, abs=1e-4)
    delta_cm_takeoff = problem["data:aerodynamics:flaps:takeoff:CM"]
    assert delta_cm_takeoff == pytest.approx(-0.0391, abs=1e-4)
    delta_cd_takeoff = problem["data:aerodynamics:flaps:takeoff:CD"]
    assert delta_cd_takeoff == pytest.approx(0.0011, abs=1e-4)
    cl_delta_elev = problem.get_val(
        "data:aerodynamics:elevator:low_speed:CL_delta", units="rad**-1"
    )
    assert cl_delta_elev == pytest.approx(0.4686, abs=1e-4)
    cd_delta_elev = problem.get_val(
        "data:aerodynamics:elevator:low_speed:CD_delta", units="rad**-2"
    )
    assert cd_delta_elev == pytest.approx(0.06226, abs=1e-4)


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
    assert cl_max_clean_wing == pytest.approx(1.56, abs=1e-2)
    cl_min_clean_wing = problem["data:aerodynamics:wing:low_speed:CL_min_clean"]
    assert cl_min_clean_wing == pytest.approx(-1.25, abs=1e-2)
    cl_max_takeoff_wing = problem["data:aerodynamics:aircraft:takeoff:CL_max"]
    assert cl_max_takeoff_wing == pytest.approx(1.65, abs=1e-2)
    cl_max_landing_wing = problem["data:aerodynamics:aircraft:landing:CL_max"]
    assert cl_max_landing_wing == pytest.approx(1.98, abs=1e-2)
    cl_max_clean_htp = problem["data:aerodynamics:horizontal_tail:low_speed:CL_max_clean"]
    assert cl_max_clean_htp == pytest.approx(0.27, abs=1e-2)
    cl_min_clean_htp = problem["data:aerodynamics:horizontal_tail:low_speed:CL_min_clean"]
    assert cl_min_clean_htp == pytest.approx(-0.27, abs=1e-2)
    alpha_max_clean_htp = problem[
        "data:aerodynamics:horizontal_tail:low_speed:clean:alpha_aircraft_max"
    ]
    assert alpha_max_clean_htp == pytest.approx(30.37, abs=1)
    alpha_min_clean_htp = problem[
        "data:aerodynamics:horizontal_tail:low_speed:clean:alpha_aircraft_min"
    ]
    assert alpha_min_clean_htp == pytest.approx(-30.35, abs=1)


def test_l_d_max():
    """ Tests best lift/drag component """

    # Define independent input value (openVSP)
    ivc = get_indep_var_comp(list_inputs(ComputeLDMax()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeLDMax(), ivc)
    l_d_max = problem["data:aerodynamics:aircraft:cruise:L_D_max"]
    assert l_d_max == pytest.approx(15.86, abs=1e-1)
    optimal_cl = problem["data:aerodynamics:aircraft:cruise:optimal_CL"]
    assert optimal_cl == pytest.approx(0.8166, abs=1e-4)
    optimal_cd = problem["data:aerodynamics:aircraft:cruise:optimal_CD"]
    assert optimal_cd == pytest.approx(0.0514, abs=1e-4)
    optimal_alpha = problem.get_val("data:aerodynamics:aircraft:cruise:optimal_alpha", units="deg")
    assert optimal_alpha == pytest.approx(7.74, abs=1e-2)


def test_cnbeta():
    """ Tests cn beta fuselage """

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(ComputeCnBetaFuselage()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeCnBetaFuselage(), ivc)
    cn_beta_fus = problem["data:aerodynamics:fuselage:cruise:CnBeta"]
    assert cn_beta_fus == pytest.approx(-0.0684, abs=1e-4)


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
    assert np.max(np.abs(y_vector_prop_on - y_result_prop_on)) <= 1e-2
    cl_vector_prop_on = problem.get_val(
        "data:aerodynamics:slipstream:wing:cruise:prop_on:CL_vector"
    )
    cl_result_prop_on = np.array(
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
    assert np.max(np.abs(cl_vector_prop_on - cl_result_prop_on)) <= 1e-2
    ct = problem.get_val("data:aerodynamics:slipstream:wing:cruise:prop_on:CT_ref")
    assert ct == pytest.approx(0.9231, abs=1e-4)
    delta_cl = problem.get_val(
        "data:aerodynamics:slipstream:wing:cruise:prop_on:CL"
    ) - problem.get_val("data:aerodynamics:slipstream:wing:cruise:prop_off:CL")
    assert delta_cl == pytest.approx(-0.0002, abs=1e-4)


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
    assert np.max(np.abs(y_vector_prop_on - y_result_prop_on)) <= 1e-2
    cl_vector_prop_on = problem.get_val(
        "data:aerodynamics:slipstream:wing:low_speed:prop_on:CL_vector"
    )
    cl_result_prop_on = np.array(
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
    assert np.max(np.abs(cl_vector_prop_on - cl_result_prop_on)) <= 1e-2
    ct = problem.get_val("data:aerodynamics:slipstream:wing:low_speed:prop_on:CT_ref")
    assert ct == pytest.approx(0.7256, abs=1e-4)
    delta_cl = problem.get_val(
        "data:aerodynamics:slipstream:wing:low_speed:prop_on:CL"
    ) - problem.get_val("data:aerodynamics:slipstream:wing:low_speed:prop_off:CL")
    assert delta_cl == pytest.approx(-0.0007, abs=1e-4)


def test_compute_mach_interpolation_roskam():
    """ Tests computation of the mach interpolation vector using Roskam's approach """

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(ComputeMachInterpolation()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeMachInterpolation(), ivc)
    cl_alpha_vector = problem["data:aerodynamics:aircraft:mach_interpolation:CL_alpha_vector"]
    cl_alpha_result = np.array([5.48, 5.51, 5.58, 5.72, 5.91, 6.18])
    assert np.max(np.abs(cl_alpha_vector - cl_alpha_result)) <= 1e-2
    mach_vector = problem["data:aerodynamics:aircraft:mach_interpolation:mach_vector"]
    mach_result = np.array([0.0, 0.07, 0.15, 0.23, 0.30, 0.38])
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
    assert cl_alpha_vt_ls == pytest.approx(3.0253, abs=1e-4)
    k_ar_effective = problem.get_val("data:aerodynamics:vertical_tail:k_ar_effective")
    assert k_ar_effective == pytest.approx(1.2612, abs=1e-4)

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(ComputeClAlphaVT()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeClAlphaVT(), ivc)
    cl_alpha_vt_cruise = problem.get_val(
        "data:aerodynamics:vertical_tail:cruise:CL_alpha", units="rad**-1"
    )
    assert cl_alpha_vt_cruise == pytest.approx(3.0889, abs=1e-4)


def test_cy_delta_r():
    """ Tests cy delta of the rudder """

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(ComputeCyDeltaRudder()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeCyDeltaRudder(), ivc)
    cy_delta_r = problem.get_val("data:aerodynamics:rudder:low_speed:Cy_delta_r", units="rad**-1")
    assert cy_delta_r == pytest.approx(1.8858, abs=1e-4)


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
    assert load_factor_ultimate == pytest.approx(5.9, abs=1e-1)
    assert vh == pytest.approx(88.71, abs=1e-2)
