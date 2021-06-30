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
import glob
import shutil
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
    ComputeAirfoilLiftCurveSlope,
    ComputeClAlphaVT,
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
    assert mach == pytest.approx(0.255041, abs=1e-4)
    reynolds = problem.get_val("data:aerodynamics:cruise:unit_reynolds", units="m**-1")
    assert reynolds == pytest.approx(4745380, abs=1)

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
    ivc.add_output("data:aerodynamics:cruise:mach", 0.2457)
    ivc.add_output("data:aerodynamics:cruise:unit_reynolds", 4571770, units="m**-1")

    # noinspection PyTypeChecker
    problem = run_system(Cd0(propulsion_id=ENGINE_WRAPPER), ivc)
    cd0_wing = problem["data:aerodynamics:wing:cruise:CD0"]
    assert cd0_wing == pytest.approx(0.00503, abs=1e-5)
    cd0_fus = problem["data:aerodynamics:fuselage:cruise:CD0"]
    assert cd0_fus == pytest.approx(0.00490, abs=1e-5)
    cd0_ht = problem["data:aerodynamics:horizontal_tail:cruise:CD0"]
    assert cd0_ht == pytest.approx(0.00123, abs=1e-5)
    cd0_vt = problem["data:aerodynamics:vertical_tail:cruise:CD0"]
    assert cd0_vt == pytest.approx(0.00077, abs=1e-5)
    cd0_nac = problem["data:aerodynamics:nacelles:cruise:CD0"]
    assert cd0_nac == pytest.approx(0.00185, abs=1e-5)
    cd0_lg = problem["data:aerodynamics:landing_gear:cruise:CD0"]
    assert cd0_lg == pytest.approx(0.0, abs=1e-5)
    cd0_other = problem["data:aerodynamics:other:cruise:CD0"]
    assert cd0_other == pytest.approx(0.00187, abs=1e-5)
    cd0_total = 1.25 * (cd0_wing + cd0_fus + cd0_ht + cd0_vt + cd0_nac + cd0_lg + cd0_other)
    assert cd0_total == pytest.approx(0.01958, abs=1e-5)


def test_cd0_low_speed():
    """ Tests drag coefficient @ low speed """

    # Research independent input value in .xml file
    # noinspection PyTypeChecker
    ivc = get_indep_var_comp(
        list_inputs(Cd0(propulsion_id=ENGINE_WRAPPER, low_speed_aero=True)), __file__, XML_FILE
    )
    ivc.add_output(
        "data:aerodynamics:low_speed:mach", 0.1149
    )  # correction to compensate old version conversion error
    ivc.add_output(
        "data:aerodynamics:low_speed:unit_reynolds", 2613822, units="m**-1"
    )  # correction to ...

    # noinspection PyTypeChecker
    problem = run_system(Cd0(propulsion_id=ENGINE_WRAPPER, low_speed_aero=True), ivc)
    cd0_wing = problem["data:aerodynamics:wing:low_speed:CD0"]
    assert cd0_wing == pytest.approx(0.00552, abs=1e-5)
    cd0_fus = problem["data:aerodynamics:fuselage:low_speed:CD0"]
    assert cd0_fus == pytest.approx(0.00547, abs=1e-5)
    cd0_ht = problem["data:aerodynamics:horizontal_tail:low_speed:CD0"]
    assert cd0_ht == pytest.approx(0.00135, abs=1e-5)
    cd0_vt = problem["data:aerodynamics:vertical_tail:low_speed:CD0"]
    assert cd0_vt == pytest.approx(0.00086, abs=1e-5)
    cd0_nac = problem["data:aerodynamics:nacelles:low_speed:CD0"]
    assert cd0_nac == pytest.approx(0.00202, abs=1e-5)
    cd0_lg = problem["data:aerodynamics:landing_gear:low_speed:CD0"]
    assert cd0_lg == pytest.approx(0.01900, abs=1e-5)
    cd0_other = problem["data:aerodynamics:other:low_speed:CD0"]
    assert cd0_other == pytest.approx(0.00187, abs=1e-5)
    cd0_total = 1.25 * (cd0_wing + cd0_fus + cd0_ht + cd0_vt + cd0_nac + cd0_lg + cd0_other)
    assert cd0_total == pytest.approx(0.04513, abs=1e-5)


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
    ivc.add_output("data:aerodynamics:low_speed:unit_reynolds", val=2782216)
    ivc.add_output("data:aerodynamics:low_speed:mach", val=0.1194)

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
    assert cl_alpha_wing == pytest.approx(6.4810, abs=1e-4)
    cl_alpha_htp = problem.get_val(
        "data:aerodynamics:horizontal_tail:airfoil:CL_alpha", units="rad**-1"
    )
    assert cl_alpha_htp == pytest.approx(6.3081, abs=1e-4)
    cl_alpha_vtp = problem.get_val(
        "data:aerodynamics:vertical_tail:airfoil:CL_alpha", units="rad**-1"
    )
    assert cl_alpha_vtp == pytest.approx(6.3081, abs=1e-4)


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
        assert cl0_wing == pytest.approx(0.0896, abs=1e-4)
        cl_alpha_wing = problem.get_val("data:aerodynamics:wing:cruise:CL_alpha", units="rad**-1")
        assert cl_alpha_wing == pytest.approx(4.832, abs=1e-3)
        cm0 = problem["data:aerodynamics:wing:cruise:CM0_clean"]
        assert cm0 == pytest.approx(-0.0263, abs=1e-4)
        coef_k_wing = problem["data:aerodynamics:wing:cruise:induced_drag_coefficient"]
        assert coef_k_wing == pytest.approx(0.05194, abs=1e-4)
        cl0_htp = problem["data:aerodynamics:horizontal_tail:cruise:CL0"]
        assert cl0_htp == pytest.approx(-0.0072, abs=1e-4)
        cl_alpha_htp = problem.get_val(
            "data:aerodynamics:horizontal_tail:cruise:CL_alpha", units="rad**-1"
        )
        assert cl_alpha_htp == pytest.approx(0.6266, abs=1e-4)
        cl_alpha_htp_isolated = problem.get_val(
            "data:aerodynamics:horizontal_tail:cruise:CL_alpha_isolated", units="rad**-1"
        )
        assert cl_alpha_htp_isolated == pytest.approx(1.0182, abs=1e-4)
        coef_k_htp = problem["data:aerodynamics:horizontal_tail:cruise:induced_drag_coefficient"]
        assert coef_k_htp == pytest.approx(0.2742, abs=1e-4)
        if mach_interpolation:
            cl_alpha_vector = problem[
                "data:aerodynamics:aircraft:mach_interpolation:CL_alpha_vector"
            ]
            assert cl_alpha_vector == pytest.approx(
                [4.827, 4.827, 4.888, 4.969, 5.069, 5.189], abs=1e-2
            )
            mach_vector = problem["data:aerodynamics:aircraft:mach_interpolation:mach_vector"]
            assert mach_vector == pytest.approx([0.0, 0.15, 0.217, 0.28, 0.339, 0.395], abs=1e-2)

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
    assert cl0_wing == pytest.approx(0.0873, abs=1e-4)
    cl_alpha_wing = problem.get_val("data:aerodynamics:wing:low_speed:CL_alpha", units="rad**-1")
    assert cl_alpha_wing == pytest.approx(4.705, abs=1e-3)
    cm0 = problem["data:aerodynamics:wing:low_speed:CM0_clean"]
    assert cm0 == pytest.approx(-0.0256, abs=1e-4)
    y_vector_wing = np.array(
        [0.09983333, 0.2995, 0.49916667, 0.918, 1.556,
         2.194, 2.832, 3.47, 4.108, 4.746,
         5.14475, 5.30425, 5.46375, 5.62325, 5.78275,
         5.94225, 6.10175]
    )
    cl_vector_wing = np.array(
        [0.09919948, 0.09915598, 0.09906166, 0.09885543, 0.09813688,
         0.09692097, 0.09506925, 0.09232772, 0.08824689, 0.08206201,
         0.07619768, 0.07167966, 0.06679433, 0.06105195, 0.05393181,
         0.04454367, 0.03069645]
    )
    chord_vector_wing = np.array(
        [1.549, 1.549, 1.549, 1.549, 1.549, 1.549, 1.549, 1.549, 1.549,
         1.549, 1.549, 1.549, 1.549, 1.549, 1.549, 1.549, 1.549, 0.,
         0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 0., 0., 0.]
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
    assert coef_k_wing == pytest.approx(0.04963, abs=1e-4)
    cl0_htp = problem["data:aerodynamics:horizontal_tail:low_speed:CL0"]
    assert cl0_htp == pytest.approx(-0.0068, abs=1e-4)
    cl_ref_htp = problem["data:aerodynamics:horizontal_tail:low_speed:CL_ref"]
    assert cl_ref_htp == pytest.approx(0.1013, abs=1e-4)
    y_vector_htp = np.array(
        [0.07492647, 0.22477941, 0.37463235, 0.52448529, 0.67433824,
         0.82419118, 0.97404412, 1.12389706, 1.27375, 1.42360294,
         1.57345588, 1.72330882, 1.87316176, 2.02301471, 2.17286765,
         2.32272059, 2.47257353]
    )
    cl_vector_htp = np.array(
        [0.11854094, 0.11834224, 0.11793907, 0.11731947, 0.1164645,
         0.11534688, 0.11392887, 0.1121591, 0.10996788, 0.10725986,
         0.10390263, 0.09970786, 0.09439854, 0.08754658, 0.07843773,
         0.06571008, 0.04595349]
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
    assert cl_alpha_htp == pytest.approx(0.62016, abs=1e-4)
    cl_alpha_htp_isolated = problem.get_val(
        "data:aerodynamics:horizontal_tail:low_speed:CL_alpha_isolated", units="rad**-1"
    )
    assert cl_alpha_htp_isolated == pytest.approx(0.99148, abs=1e-4)
    coef_k_htp = problem["data:aerodynamics:horizontal_tail:low_speed:induced_drag_coefficient"]
    assert coef_k_htp == pytest.approx(0.2782, abs=1e-4)
    assert (duration_2nd_run / duration_1st_run) <= 0.1

    # Remove existing result files
    results_folder.cleanup()


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
        assert cl0_wing == pytest.approx(0.1171, abs=1e-4)
        cl_alpha_wing = problem.get_val("data:aerodynamics:wing:cruise:CL_alpha", units="rad**-1")
        assert cl_alpha_wing == pytest.approx(4.595, abs=1e-3)
        cm0 = problem["data:aerodynamics:wing:cruise:CM0_clean"]
        assert cm0 == pytest.approx(-0.0265, abs=1e-4)
        coef_k_wing = problem["data:aerodynamics:wing:cruise:induced_drag_coefficient"]
        assert coef_k_wing == pytest.approx(0.0482, abs=1e-4)
        if mach_interpolation:
            cl_alpha_vector = problem[
                "data:aerodynamics:aircraft:mach_interpolation:CL_alpha_vector"
            ]
            assert cl_alpha_vector == pytest.approx([5.20, 5.20, 5.24, 5.30, 5.37, 5.45], abs=1e-2)
            mach_vector = problem["data:aerodynamics:aircraft:mach_interpolation:mach_vector"]
            assert mach_vector == pytest.approx([0.0, 0.15, 0.21, 0.28, 0.34, 0.39], abs=1e-2)
        cl0_htp = problem["data:aerodynamics:horizontal_tail:cruise:CL0"]
        assert cl0_htp == pytest.approx(-0.0058, abs=1e-4)
        cl_alpha_htp = problem.get_val(
            "data:aerodynamics:horizontal_tail:cruise:CL_alpha", units="rad**-1"
        )
        assert cl_alpha_htp == pytest.approx(0.6826, abs=1e-4)
        cl_alpha_htp_isolated = problem.get_val(
            "data:aerodynamics:horizontal_tail:cruise:CL_alpha_isolated", units="rad**-1"
        )
        assert cl_alpha_htp_isolated == pytest.approx(1.0387, abs=1e-4)
        coef_k_htp = problem["data:aerodynamics:horizontal_tail:cruise:induced_drag_coefficient"]
        assert coef_k_htp == pytest.approx(0.4605, abs=1e-4)
        assert (duration_2nd_run / duration_1st_run) <= 0.01

        # Remove existing result files
        results_folder.cleanup()


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
    assert cl_alpha_wing == pytest.approx(4.509, abs=1e-3)
    cm0 = problem["data:aerodynamics:wing:low_speed:CM0_clean"]
    assert cm0 == pytest.approx(-0.0258, abs=1e-4)
    y_vector_wing = np.array(
        [
            0.04279,
            0.12836,
            0.21393,
            0.2995,
            0.38507,
            0.47064,
            0.55621,
            0.68649,
            0.862,
            1.0385,
            1.21588,
            1.39404,
            1.57287,
            1.75226,
            1.93212,
            2.11231,
            2.29274,
            2.47328,
            2.65384,
            2.83429,
            3.01452,
            3.19443,
            3.37389,
            3.5528,
            3.73106,
            3.90855,
            4.08517,
            4.26082,
            4.43539,
            4.60879,
            4.78093,
            4.9517,
            5.12103,
            5.28882,
            5.45499,
            5.61947,
            5.78218,
            5.94305,
            6.10201,
        ]
    )
    cl_vector_wing = np.array(
        [
            0.12714757,
            0.1275284,
            0.12742818,
            0.12739811,
            0.12730792,
            0.12730792,
            0.1272077,
            0.12667654,
            0.12628568,
            0.12660638,
            0.126887,
            0.12680682,
            0.12623558,
            0.12584472,
            0.12580463,
            0.12553404,
            0.12503295,
            0.12456192,
            0.12390048,
            0.12320897,
            0.12222682,
            0.12144512,
            0.12058324,
            0.11971133,
            0.11849869,
            0.11724595,
            0.11562241,
            0.11403895,
            0.11208468,
            0.11031081,
            0.10819619,
            0.10589116,
            0.10301488,
            0.10000832,
            0.09569891,
            0.09022697,
            0.08247003,
            0.07037363,
            0.05267499,
        ]
    )
    y, cl = reshape_curve(
        problem.get_val("data:aerodynamics:wing:low_speed:Y_vector", "m"),
        problem["data:aerodynamics:wing:low_speed:CL_vector"],
    )
    assert np.max(np.abs(y_vector_wing - y)) <= 1e-3
    assert np.max(np.abs(cl_vector_wing - cl)) <= 1e-3
    coef_k_wing = problem["data:aerodynamics:wing:low_speed:induced_drag_coefficient"]
    assert coef_k_wing == pytest.approx(0.0482, abs=1e-4)
    cl0_htp = problem["data:aerodynamics:horizontal_tail:low_speed:CL0"]
    assert cl0_htp == pytest.approx(-0.0055, abs=1e-4)
    cl_ref_htp = problem["data:aerodynamics:horizontal_tail:low_speed:CL_ref"]
    assert cl_ref_htp == pytest.approx(0.1124, abs=1e-4)
    y_vector_htp = np.array(
        [
            0.05307,
            0.15922,
            0.26536,
            0.37151,
            0.47766,
            0.5838,
            0.68995,
            0.79609,
            0.90224,
            1.00839,
            1.11453,
            1.22068,
            1.32682,
            1.43297,
            1.53911,
            1.64526,
            1.75141,
            1.85755,
            1.9637,
            2.06984,
            2.17599,
            2.28214,
            2.38828,
            2.49443,
        ]
    )
    cl_vector_htp = np.array(
        [
            0.12706265,
            0.12950803,
            0.12983592,
            0.12961425,
            0.12981745,
            0.12923324,
            0.12780388,
            0.12690332,
            0.12606048,
            0.12506524,
            0.12356892,
            0.12197331,
            0.11966648,
            0.11778684,
            0.11551696,
            0.11321936,
            0.11022672,
            0.10689925,
            0.1019831,
            0.0968037,
            0.08927591,
            0.08074826,
            0.06745917,
            0.05250057,
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
    assert cl_alpha_htp == pytest.approx(0.6760, abs=1e-4)
    cl_alpha_htp_isolated = problem.get_val(
        "data:aerodynamics:horizontal_tail:low_speed:CL_alpha_isolated", units="rad**-1"
    )
    assert cl_alpha_htp_isolated == pytest.approx(1.0198, abs=1e-4)
    coef_k_htp = problem["data:aerodynamics:horizontal_tail:low_speed:induced_drag_coefficient"]
    assert coef_k_htp == pytest.approx(0.4587, abs=1e-4)
    assert (duration_2nd_run / duration_1st_run) <= 0.1

    # Remove existing result files
    results_folder.cleanup()


def test_2d_hinge_moment():
    """ Tests tail hinge-moments """

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(Compute2DHingeMomentsTail()), __file__, XML_FILE)
    ivc.add_output("data:aerodynamics:horizontal_tail:cruise:CL_alpha", 0.6826, units="rad**-1")
    ivc.add_output("data:aerodynamics:horizontal_tail:airfoil:CL_alpha", 6.3090, units="rad**-1")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(Compute2DHingeMomentsTail(), ivc)
    ch_alpha_2d = problem.get_val(
        "data:aerodynamics:horizontal_tail:cruise:hinge_moment:CH_alpha_2D", units="rad**-1"
    )
    assert ch_alpha_2d == pytest.approx(-0.3557, abs=1e-4)
    ch_delta_2d = problem.get_val(
        "data:aerodynamics:horizontal_tail:cruise:hinge_moment:CH_delta_2D", units="rad**-1"
    )
    assert ch_delta_2d == pytest.approx(-0.5752, abs=1e-4)


def test_3d_hinge_moment():
    """ Tests tail hinge-moments """

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(Compute3DHingeMomentsTail()), __file__, XML_FILE)
    ivc.add_output(
        "data:aerodynamics:horizontal_tail:cruise:hinge_moment:CH_alpha_2D",
        -0.3339,
        units="rad**-1",
    )
    ivc.add_output(
        "data:aerodynamics:horizontal_tail:cruise:hinge_moment:CH_delta_2D",
        -0.6358,
        units="rad**-1",
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(Compute3DHingeMomentsTail(), ivc)
    ch_alpha = problem.get_val(
        "data:aerodynamics:horizontal_tail:cruise:hinge_moment:CH_alpha", units="rad**-1"
    )
    assert ch_alpha == pytest.approx(-0.2486, abs=1e-4)
    ch_delta = problem.get_val(
        "data:aerodynamics:horizontal_tail:cruise:hinge_moment:CH_delta", units="rad**-1"
    )
    assert ch_delta == pytest.approx(-0.6765, abs=1e-4)


def test_high_lift():
    """ Tests high-lift contribution """

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(ComputeDeltaHighLift()), __file__, XML_FILE)
    ivc.add_output("data:aerodynamics:low_speed:mach", 0.1149)
    ivc.add_output("data:aerodynamics:wing:low_speed:CL_alpha", 4.569, units="rad**-1")
    ivc.add_output("data:aerodynamics:horizontal_tail:airfoil:CL_alpha", 6.3090, units="rad**-1")
    ivc.add_output("data:aerodynamics:wing:airfoil:CL_alpha", 6.4810, units="rad**-1")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeDeltaHighLift(), ivc)
    delta_cl0_landing = problem["data:aerodynamics:flaps:landing:CL"]
    assert delta_cl0_landing == pytest.approx(0.7145, abs=1e-4)
    delta_clmax_landing = problem["data:aerodynamics:flaps:landing:CL_max"]
    assert delta_clmax_landing == pytest.approx(0.5258, abs=1e-4)
    delta_cm_landing = problem["data:aerodynamics:flaps:landing:CM"]
    assert delta_cm_landing == pytest.approx(-0.0964, abs=1e-4)
    delta_cd_landing = problem["data:aerodynamics:flaps:landing:CD"]
    assert delta_cd_landing == pytest.approx(0.01383, abs=1e-4)
    delta_cl0_takeoff = problem["data:aerodynamics:flaps:takeoff:CL"]
    assert delta_cl0_takeoff == pytest.approx(0.2735, abs=1e-4)
    delta_clmax_takeoff = problem["data:aerodynamics:flaps:takeoff:CL_max"]
    assert delta_clmax_takeoff == pytest.approx(0.1076, abs=1e-4)
    delta_cm_takeoff = problem["data:aerodynamics:flaps:takeoff:CM"]
    assert delta_cm_takeoff == pytest.approx(-0.0369, abs=1e-4)
    delta_cd_takeoff = problem["data:aerodynamics:flaps:takeoff:CD"]
    assert delta_cd_takeoff == pytest.approx(0.00111811, abs=1e-4)
    cl_delta_elev = problem.get_val(
        "data:aerodynamics:elevator:low_speed:CL_delta", units="rad**-1"
    )
    assert cl_delta_elev == pytest.approx(0.5424, abs=1e-4)
    cd_delta_elev = problem.get_val(
        "data:aerodynamics:elevator:low_speed:CD_delta", units="rad**-1"
    )
    assert cd_delta_elev == pytest.approx(0.0312, abs=1e-4)


def test_extreme_cl():
    """ Tests maximum/minimum cl component with default result cl=f(y) curve"""

    # Transfer saved polar results to temporary folder
    tmp_folder = polar_result_transfer()

    # Research independent input value in .xml file for Openvsp test
    ivc = get_indep_var_comp(list_inputs(ComputeExtremeCL()), __file__, XML_FILE)
    y_vector_wing = np.zeros(SPAN_MESH_POINT)
    cl_vector_wing = np.zeros(SPAN_MESH_POINT)
    y_vector_htp = np.zeros(SPAN_MESH_POINT)
    cl_vector_htp = np.zeros(SPAN_MESH_POINT)
    y_vector_wing[0:39] = [
        0.04279,
        0.12836,
        0.21393,
        0.2995,
        0.38507,
        0.47064,
        0.55621,
        0.68649,
        0.862,
        1.0385,
        1.21588,
        1.39404,
        1.57287,
        1.75226,
        1.93212,
        2.11231,
        2.29274,
        2.47328,
        2.65384,
        2.83429,
        3.01452,
        3.19443,
        3.37389,
        3.5528,
        3.73106,
        3.90855,
        4.08517,
        4.26082,
        4.43539,
        4.60879,
        4.78093,
        4.9517,
        5.12103,
        5.28882,
        5.45499,
        5.61947,
        5.78218,
        5.94305,
        6.10201,
    ]
    cl_vector_wing[0:39] = [
        0.0989,
        0.09908,
        0.09901,
        0.09898,
        0.09892,
        0.09888,
        0.09887,
        0.09871,
        0.09823,
        0.09859,
        0.09894,
        0.09888,
        0.09837,
        0.098,
        0.0979,
        0.09763,
        0.09716,
        0.09671,
        0.0961,
        0.09545,
        0.09454,
        0.09377,
        0.09295,
        0.09209,
        0.09087,
        0.08965,
        0.08812,
        0.0866,
        0.08465,
        0.08284,
        0.08059,
        0.07817,
        0.07494,
        0.07178,
        0.06773,
        0.06279,
        0.05602,
        0.04639,
        0.03265,
    ]
    y_vector_htp[0:24] = [
        0.05307,
        0.15922,
        0.26536,
        0.37151,
        0.47766,
        0.5838,
        0.68995,
        0.79609,
        0.90224,
        1.00839,
        1.11453,
        1.22068,
        1.32682,
        1.43297,
        1.53911,
        1.64526,
        1.75141,
        1.85755,
        1.9637,
        2.06984,
        2.17599,
        2.28214,
        2.38828,
        2.49443,
    ]
    cl_vector_htp[0:24] = [
        0.12706265,
        0.12950803,
        0.12983592,
        0.12961425,
        0.12981745,
        0.12923324,
        0.12780388,
        0.12690332,
        0.12606048,
        0.12506524,
        0.12356892,
        0.12197331,
        0.11966648,
        0.11778684,
        0.11551696,
        0.11321936,
        0.11022672,
        0.10689925,
        0.1019831,
        0.0968037,
        0.08927591,
        0.08074826,
        0.06745917,
        0.05250057,
    ]
    ivc.add_output("data:aerodynamics:flaps:landing:CL_max", 0.5788)
    ivc.add_output("data:aerodynamics:flaps:takeoff:CL_max", 0.1218)
    ivc.add_output("data:aerodynamics:wing:low_speed:Y_vector", y_vector_wing, units="m")
    ivc.add_output("data:aerodynamics:wing:low_speed:CL_vector", cl_vector_wing)
    ivc.add_output("data:aerodynamics:wing:low_speed:CL0_clean", 0.0877)
    ivc.add_output("data:aerodynamics:horizontal_tail:low_speed:Y_vector", y_vector_htp, units="m")
    ivc.add_output("data:aerodynamics:horizontal_tail:low_speed:CL_vector", cl_vector_htp)
    ivc.add_output("data:aerodynamics:horizontal_tail:low_speed:CL_ref", 0.11245)
    ivc.add_output("data:aerodynamics:horizontal_tail:low_speed:CL_alpha", 0.6760, units="rad**-1")
    ivc.add_output("data:aerodynamics:low_speed:mach", 0.1149)
    ivc.add_output("data:aerodynamics:low_speed:unit_reynolds", 2613822, units="m**-1")

    # Run problem
    problem = run_system(ComputeExtremeCL(), ivc)

    # Retrieve polar results from temporary folder
    polar_result_retrieve(tmp_folder)

    # Check obtained value(s) is/(are) correct
    cl_max_clean_wing = problem["data:aerodynamics:wing:low_speed:CL_max_clean"]
    assert cl_max_clean_wing == pytest.approx(1.49, abs=1e-2)
    cl_min_clean_wing = problem["data:aerodynamics:wing:low_speed:CL_min_clean"]
    assert cl_min_clean_wing == pytest.approx(-1.19, abs=1e-2)
    cl_max_takeoff_wing = problem["data:aerodynamics:aircraft:takeoff:CL_max"]
    assert cl_max_takeoff_wing == pytest.approx(1.618, abs=1e-2)
    cl_max_landing_wing = problem["data:aerodynamics:aircraft:landing:CL_max"]
    assert cl_max_landing_wing == pytest.approx(2.07, abs=1e-2)
    cl_max_clean_htp = problem["data:aerodynamics:horizontal_tail:low_speed:CL_max_clean"]
    assert cl_max_clean_htp == pytest.approx(0.314, abs=1e-2)
    cl_min_clean_htp = problem["data:aerodynamics:horizontal_tail:low_speed:CL_min_clean"]
    assert cl_min_clean_htp == pytest.approx(-0.314, abs=1e-2)
    alpha_max_clean_htp = problem[
        "data:aerodynamics:horizontal_tail:low_speed:clean:alpha_aircraft_max"
    ]
    assert alpha_max_clean_htp == pytest.approx(26.64, abs=1e-2)
    alpha_min_clean_htp = problem[
        "data:aerodynamics:horizontal_tail:low_speed:clean:alpha_aircraft_min"
    ]
    assert alpha_min_clean_htp == pytest.approx(-26.62, abs=1e-2)


def test_l_d_max():
    """ Tests best lift/drag component """

    # Define independent input value (openVSP)
    ivc = om.IndepVarComp()
    ivc.add_output("data:aerodynamics:wing:cruise:CL0_clean", 0.0906)
    ivc.add_output("data:aerodynamics:wing:cruise:CL_alpha", 4.650, units="rad**-1")
    ivc.add_output("data:aerodynamics:aircraft:cruise:CD0", 0.01603)
    ivc.add_output("data:aerodynamics:wing:cruise:induced_drag_coefficient", 0.0480)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeLDMax(), ivc)
    l_d_max = problem["data:aerodynamics:aircraft:cruise:L_D_max"]
    assert l_d_max == pytest.approx(18.0, abs=1e-1)
    optimal_cl = problem["data:aerodynamics:aircraft:cruise:optimal_CL"]
    assert optimal_cl == pytest.approx(0.5778, abs=1e-4)
    optimal_cd = problem["data:aerodynamics:aircraft:cruise:optimal_CD"]
    assert optimal_cd == pytest.approx(0.0320, abs=1e-4)
    optimal_alpha = problem.get_val("data:aerodynamics:aircraft:cruise:optimal_alpha", units="deg")
    assert optimal_alpha == pytest.approx(6.00, abs=1e-2)


def test_cnbeta():
    """ Tests cn beta fuselage """

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(ComputeCnBetaFuselage()), __file__, XML_FILE)
    ivc.add_output("data:aerodynamics:cruise:mach", 0.245)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeCnBetaFuselage(), ivc)
    cn_beta_fus = problem["data:aerodynamics:fuselage:cruise:CnBeta"]
    assert cn_beta_fus == pytest.approx(-0.0459, abs=1e-4)


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
    ivc.add_output("data:aerodynamics:wing:cruise:CL0_clean", val=0.1173)
    ivc.add_output("data:aerodynamics:wing:cruise:CL_alpha", val=4.5996, units="rad**-1")
    ivc.add_output("data:aerodynamics:wing:low_speed:CL_max_clean", val=1.4465)
    # Run problem and check obtained value(s) is/(are) correct
    # start = time.time()
    # noinspection PyTypeChecker
    problem = run_system(
        ComputeSlipstreamOpenvsp(
            propulsion_id=ENGINE_WRAPPER,
            result_folder_path=results_folder.name,
            low_speed_aero=False,
        ),
        ivc,
    )
    # stop = time.time()
    # duration_1st_run = stop - start
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
            0.69,
            0.86,
            1.04,
            1.22,
            1.39,
            1.57,
            1.75,
            1.93,
            2.11,
            2.29,
            2.47,
            2.65,
            2.83,
            3.01,
            3.19,
            3.37,
            3.55,
            3.73,
            3.91,
            4.09,
            4.26,
            4.44,
            4.61,
            4.78,
            4.95,
            5.12,
            5.29,
            5.45,
            5.62,
            5.78,
            5.94,
            6.1,
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
            1.657,
            1.655,
            1.655,
            1.655,
            1.654,
            1.653,
            1.658,
            1.649,
            1.645,
            1.65,
            1.659,
            1.714,
            1.737,
            1.752,
            1.737,
            1.658,
            1.627,
            1.606,
            1.594,
            1.571,
            1.535,
            1.529,
            1.518,
            1.506,
            1.484,
            1.465,
            1.439,
            1.414,
            1.378,
            1.348,
            1.309,
            1.271,
            1.22,
            1.171,
            1.1,
            1.024,
            0.923,
            0.793,
            0.758,
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
    assert ct == pytest.approx(0.0483, abs=1e-4)
    delta_cl = problem.get_val(
        "data:aerodynamics:slipstream:wing:cruise:prop_on:CL"
    ) - problem.get_val("data:aerodynamics:slipstream:wing:cruise:prop_off:CL")
    assert delta_cl == pytest.approx(0.00565, abs=1e-4)


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
    ivc.add_output("data:aerodynamics:wing:low_speed:CL0_clean", val=0.1147)
    ivc.add_output("data:aerodynamics:wing:low_speed:CL_alpha", val=4.509, units="rad**-1")
    ivc.add_output("data:aerodynamics:wing:low_speed:CL_max_clean", val=1.4465)
    # Run problem and check obtained value(s) is/(are) correct
    # start = time.time()
    # noinspection PyTypeChecker
    problem = run_system(
        ComputeSlipstreamOpenvsp(
            propulsion_id=ENGINE_WRAPPER,
            result_folder_path=results_folder.name,
            low_speed_aero=True,
        ),
        ivc,
    )
    # stop = time.time()
    # duration_1st_run = stop - start
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
            0.69,
            0.86,
            1.04,
            1.22,
            1.39,
            1.57,
            1.75,
            1.93,
            2.11,
            2.29,
            2.47,
            2.65,
            2.83,
            3.01,
            3.19,
            3.37,
            3.55,
            3.73,
            3.91,
            4.09,
            4.26,
            4.44,
            4.61,
            4.78,
            4.95,
            5.12,
            5.29,
            5.45,
            5.62,
            5.78,
            5.94,
            6.1,
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
            1.691,
            1.653,
            1.653,
            1.653,
            1.653,
            1.652,
            1.658,
            1.648,
            1.645,
            1.652,
            1.663,
            1.83,
            1.926,
            1.96,
            1.939,
            1.798,
            1.751,
            1.713,
            1.668,
            1.617,
            1.513,
            1.513,
            1.505,
            1.496,
            1.477,
            1.459,
            1.435,
            1.411,
            1.375,
            1.346,
            1.309,
            1.272,
            1.221,
            1.173,
            1.103,
            1.028,
            0.928,
            0.799,
            0.767,
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
    assert ct == pytest.approx(0.03796, abs=1e-4)
    delta_cl = problem.get_val(
        "data:aerodynamics:slipstream:wing:low_speed:prop_on:CL"
    ) - problem.get_val("data:aerodynamics:slipstream:wing:low_speed:prop_off:CL")
    assert delta_cl == pytest.approx(0.02196, abs=1e-4)


def test_compute_mach_interpolation_roskam():
    """ Tests computation of the mach interpolation vector using Roskam's approach """

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(ComputeMachInterpolation()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeMachInterpolation(), ivc)
    cl_alpha_vector = problem["data:aerodynamics:aircraft:mach_interpolation:CL_alpha_vector"]
    cl_alpha_result = np.array([5.45, 5.48, 5.55, 5.68, 5.87, 6.14])
    assert np.max(np.abs(cl_alpha_vector - cl_alpha_result)) <= 1e-2
    mach_vector = problem["data:aerodynamics:aircraft:mach_interpolation:mach_vector"]
    mach_result = np.array([0.0, 0.08, 0.16, 0.24, 0.32, 0.40])
    assert np.max(np.abs(mach_vector - mach_result)) <= 1e-2


def test_cl_alpha_vt():
    """ Tests Cl alpha vt """

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(ComputeClAlphaVT(low_speed_aero=True)), __file__, XML_FILE)
    ivc.add_output("data:aerodynamics:low_speed:mach", 0.119)
    ivc.add_output("data:aerodynamics:vertical_tail:airfoil:CL_alpha", 6.4038, units="rad**-1")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeClAlphaVT(low_speed_aero=True), ivc)
    cl_alpha_vt_ls = problem.get_val(
        "data:aerodynamics:vertical_tail:low_speed:CL_alpha", units="rad**-1"
    )
    assert cl_alpha_vt_ls == pytest.approx(2.8398, abs=1e-4)
    k_ar_effective = problem.get_val("data:aerodynamics:vertical_tail:k_ar_effective")
    assert k_ar_effective == pytest.approx(1.8832, abs=1e-4)

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(ComputeClAlphaVT()), __file__, XML_FILE)
    ivc.add_output("data:aerodynamics:cruise:mach", 0.248)
    ivc.add_output("data:aerodynamics:vertical_tail:airfoil:CL_alpha", 6.4038, units="rad**-1")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeClAlphaVT(), ivc)
    cl_alpha_vt_cruise = problem.get_val(
        "data:aerodynamics:vertical_tail:cruise:CL_alpha", units="rad**-1"
    )
    assert cl_alpha_vt_cruise == pytest.approx(2.8912, abs=1e-4)


def test_cy_delta_r():
    """ Tests cy delta of the rudder """

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(ComputeCyDeltaRudder()), __file__, XML_FILE)
    ivc.add_output(
        "data:aerodynamics:vertical_tail:low_speed:CL_alpha", val=2.8398, units="rad**-1"
    )
    ivc.add_output("data:aerodynamics:vertical_tail:airfoil:CL_alpha", val=6.4038, units="rad**-1")
    ivc.add_output("data:aerodynamics:vertical_tail:k_ar_effective", val=1.8832)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeCyDeltaRudder(), ivc)
    cy_delta_r = problem.get_val("data:aerodynamics:rudder:low_speed:Cy_delta_r", units="rad**-1")
    assert cy_delta_r == pytest.approx(1.6000, abs=1e-4)


def test_high_speed_connection():
    """ Tests high speed components connection """

    # load all inputs
    reader = VariableIO(pth.join(pth.dirname(__file__), "data", XML_FILE))
    reader.path_separator = ":"
    input_vars = reader.read().to_ivc()
    input_vars.add_output(
        "data:aerodynamics:vertical_tail:airfoil:CL_alpha", 6.3090, units="rad**-1"
    )
    input_vars.add_output(
        "data:aerodynamics:horizontal_tail:airfoil:CL_alpha", 6.3090, units="rad**-1"
    )

    # Run problem with VLM
    # noinspection PyTypeChecker
    run_system(AerodynamicsHighSpeed(propulsion_id=ENGINE_WRAPPER, use_openvsp=False), input_vars)

    # Run problem with OPENVSP
    # noinspection PyTypeChecker
    run_system(AerodynamicsHighSpeed(propulsion_id=ENGINE_WRAPPER, use_openvsp=True), input_vars)


def test_low_speed_connection():
    """ Tests low speed components connection """

    # load all inputs
    reader = VariableIO(pth.join(pth.dirname(__file__), "data", XML_FILE))
    reader.path_separator = ":"
    input_vars = reader.read().to_ivc()

    # Run problem with VLM
    # noinspection PyTypeChecker
    run_system(AerodynamicsLowSpeed(propulsion_id=ENGINE_WRAPPER, use_openvsp=False), input_vars)

    # Run problem with OPENVSP
    # noinspection PyTypeChecker
    run_system(AerodynamicsLowSpeed(propulsion_id=ENGINE_WRAPPER, use_openvsp=True), input_vars)


def test_v_n_diagram():
    # load all inputs
    reader = VariableIO(pth.join(pth.dirname(__file__), "data", XML_FILE))
    reader.path_separator = ":"
    input_vars = reader.read().to_ivc()
    input_vars.add_output("data:aerodynamics:aircraft:landing:CL_max", 1.9)
    input_vars.add_output("data:aerodynamics:wing:low_speed:CL_max_clean", 1.5)
    input_vars.add_output("data:aerodynamics:wing:low_speed:CL_min_clean", -1.5)
    input_vars.add_output(
        "data:aerodynamics:aircraft:mach_interpolation:CL_alpha_vector",
        [5.20, 5.20, 5.24, 5.30, 5.37, 5.45],
    )
    input_vars.add_output(
        "data:aerodynamics:aircraft:mach_interpolation:mach_vector",
        [0.0, 0.15, 0.21, 0.28, 0.34, 0.39],
    )
    input_vars.add_output("data:weight:aircraft:MTOW", 1700.0, units="kg")
    input_vars.add_output("data:aerodynamics:cruise:mach", 0.21)
    input_vars.add_output("data:aerodynamics:wing:cruise:induced_drag_coefficient", 0.048)
    input_vars.add_output("data:aerodynamics:aircraft:cruise:CD0", 0.016)

    # Run problem with VLM and check obtained value(s) is/(are) correct
    # noinspection PyTypeChecker
    problem = run_system(
        ComputeVNAndVH(propulsion_id=ENGINE_WRAPPER), input_vars
    )
    velocity_vect = np.array(
        [
            30.782,
            30.782,
            60.006,
            37.951,
            0.0,
            0.0,
            74.799,
            74.799,
            74.799,
            101.345,
            101.345,
            101.345,
            101.345,
            91.210,
            74.799,
            0.0,
            27.350,
            38.680,
            49.231,
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
            4.004,
            -2.004,
            3.8,
            0.0,
            3.059,
            -1.059,
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
    reader = VariableIO(pth.join(pth.dirname(__file__), "data", XML_FILE))
    reader.path_separator = ":"
    input_vars = reader.read().to_ivc()
    input_vars.add_output("data:aerodynamics:aircraft:landing:CL_max", 1.9)
    input_vars.add_output("data:aerodynamics:wing:low_speed:CL_max_clean", 1.5)
    input_vars.add_output("data:aerodynamics:wing:low_speed:CL_min_clean", -1.5)
    input_vars.add_output(
        "data:aerodynamics:aircraft:mach_interpolation:CL_alpha_vector",
        [5.20, 5.20, 5.24, 5.30, 5.37, 5.45],
    )
    input_vars.add_output(
        "data:aerodynamics:aircraft:mach_interpolation:mach_vector",
        [0.0, 0.15, 0.21, 0.28, 0.34, 0.39],
    )
    input_vars.add_output("data:weight:aircraft:MTOW", 1700.0, units="kg")
    input_vars.add_output("data:aerodynamics:cruise:mach", 0.21)
    input_vars.add_output("data:aerodynamics:wing:cruise:induced_drag_coefficient", 0.048)
    input_vars.add_output("data:aerodynamics:aircraft:cruise:CD0", 0.016)

    problem = run_system(
        LoadFactor(propulsion_id=ENGINE_WRAPPER), input_vars
    )

    load_factor_ultimate = problem.get_val("data:mission:sizing:cs23:sizing_factor_ultimate")
    vh = problem.get_val("data:TLAR:v_max_sl", units="m/s")
    assert load_factor_ultimate == pytest.approx(6.0, abs=1e-1)
    assert vh == pytest.approx(110.88, abs=1e-2)


def test_propeller():
    # Transfer saved polar results to temporary folder
    tmp_folder = polar_result_transfer()

    # load all inputs and add missing ones
    reader = VariableIO(pth.join(pth.dirname(__file__), "data", XML_FILE))
    reader.path_separator = ":"
    ivc = reader.read().to_ivc()
    ivc.add_output("data:geometry:propeller:diameter", 2.0 * 0.965, units="m")
    ivc.add_output("data:geometry:propeller:hub_diameter", 2.0 * 0.965 * 0.18, units="m")
    ivc.add_output("data:geometry:propeller:blades_number", 2)
    twist_law = lambda x: 25.0 / (x + 0.75) ** 1.5 + 46.3917 - 15.0
    radius_ratio = [0.165, 0.3, 0.45, 0.655, 0.835, 0.975, 1.0]
    chord = list(
        np.array(
            [0.11568421, 0.16431579, 0.16844211, 0.21957895, 0.19231579, 0.11568421, 0.11568421]
        )
        * 0.965
    )
    sweep = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    ivc.add_output("data:geometry:propeller:radius_ratio_vect", radius_ratio)
    ivc.add_output("data:geometry:propeller:chord_vect", chord, units="m")
    ivc.add_output(
        "data:geometry:propeller:twist_vect", twist_law(np.array(radius_ratio)), units="deg"
    )
    ivc.add_output("data:geometry:propeller:sweep_vect", sweep, units="deg")

    # Run problem
    problem = run_system(ComputePropellerPerformance(), ivc)

    # Retrieve polar results from temporary folder
    polar_result_retrieve(tmp_folder)

    # Check obtained value(s) is/(are) correct
    thrust_SL = np.array(
        [
            171.84274501,
            407.28396054,
            642.72517608,
            878.16639161,
            1113.60760715,
            1349.04882268,
            1584.49003822,
            1819.93125375,
            2055.37246929,
            2290.81368482,
            2526.25490036,
            2761.69611589,
            2997.13733143,
            3232.57854696,
            3468.0197625,
            3703.46097803,
            3938.90219357,
            4174.3434091,
            4409.78462464,
            4645.22584017,
            4880.66705571,
            5116.10827124,
            5351.54948678,
            5586.99070231,
            5822.43191785,
            6057.87313339,
            6293.31434892,
            6528.75556446,
            6764.19677999,
            6999.63799553,
        ]
    )
    trust_SL_limit = np.array(
        [
            4261.11298032,
            4693.26428823,
            5041.84304526,
            5337.32979481,
            5605.4559526,
            5862.06604269,
            6118.57481152,
            6389.01865338,
            6680.78504516,
            6999.63799553,
        ]
    )
    speed = np.array(
        [
            5.0,
            15.69362963,
            26.38725926,
            37.08088889,
            47.77451852,
            58.46814815,
            69.16177778,
            79.85540741,
            90.54903704,
            101.24266667,
        ]
    )
    efficiency_SL = np.array(
        [
            [
                0.10939049,
                0.1907898,
                0.22301954,
                0.22816292,
                0.22232582,
                0.21238634,
                0.20133362,
                0.19056759,
                0.18059171,
                0.17100852,
                0.16166078,
                0.15301322,
                0.14482924,
                0.13720703,
                0.13057262,
                0.12455244,
                0.11861937,
                0.1121606,
                0.10902989,
                0.10902989,
                0.10902989,
                0.10902989,
                0.10902989,
                0.10902989,
                0.10902989,
                0.10902989,
                0.10902989,
                0.10902989,
                0.10902989,
                0.10902989,
            ],
            [
                0.28069643,
                0.45368127,
                0.51230501,
                0.52424245,
                0.5168359,
                0.50144289,
                0.48324799,
                0.46472333,
                0.44702939,
                0.42988949,
                0.41233482,
                0.39556908,
                0.37948525,
                0.36373725,
                0.34882886,
                0.33580203,
                0.3237728,
                0.3120188,
                0.29984686,
                0.28427683,
                0.27893219,
                0.27893219,
                0.27893219,
                0.27893219,
                0.27893219,
                0.27893219,
                0.27893219,
                0.27893219,
                0.27893219,
                0.27893219,
            ],
            [
                0.39835103,
                0.59011651,
                0.65052804,
                0.66556023,
                0.66177375,
                0.64959625,
                0.63358332,
                0.61620613,
                0.59955548,
                0.58347061,
                0.56579142,
                0.54821194,
                0.53133393,
                0.51485367,
                0.49774971,
                0.48193154,
                0.46813046,
                0.45488897,
                0.44194993,
                0.42874159,
                0.41285839,
                0.39548822,
                0.39548822,
                0.39548822,
                0.39548822,
                0.39548822,
                0.39548822,
                0.39548822,
                0.39548822,
                0.39548822,
            ],
            [
                0.45987637,
                0.65270058,
                0.71240788,
                0.73113093,
                0.73255675,
                0.72539871,
                0.71397092,
                0.70050835,
                0.68665372,
                0.67390195,
                0.65902065,
                0.64331188,
                0.62799453,
                0.61295837,
                0.59796964,
                0.5817973,
                0.56766654,
                0.55488126,
                0.54231053,
                0.52988774,
                0.517322,
                0.50242233,
                0.4778603,
                0.4778603,
                0.4778603,
                0.4778603,
                0.4778603,
                0.4778603,
                0.4778603,
                0.4778603,
            ],
            [
                0.47756191,
                0.67330178,
                0.73539581,
                0.75934746,
                0.76605044,
                0.76403632,
                0.75742891,
                0.74818129,
                0.73818185,
                0.7283108,
                0.71669395,
                0.70378455,
                0.69066811,
                0.67764203,
                0.66477733,
                0.65105245,
                0.63738778,
                0.6251831,
                0.61378093,
                0.60247409,
                0.59118801,
                0.57963577,
                0.56627785,
                0.54718361,
                0.54244216,
                0.54244216,
                0.54244216,
                0.54244216,
                0.54244216,
                0.54244216,
            ],
            [
                0.46855797,
                0.66808683,
                0.73889969,
                0.76832514,
                0.78020875,
                0.78307783,
                0.78088372,
                0.77560106,
                0.76934349,
                0.76214968,
                0.75322206,
                0.7431267,
                0.732422,
                0.72156979,
                0.71068581,
                0.69926271,
                0.6873694,
                0.67618342,
                0.66565784,
                0.65546074,
                0.64546681,
                0.63542817,
                0.62507203,
                0.61321511,
                0.59705147,
                0.59097708,
                0.59097708,
                0.59097708,
                0.59097708,
                0.59097708,
            ],
            [
                0.45140381,
                0.654121,
                0.73122723,
                0.7669293,
                0.78399635,
                0.79131313,
                0.7930515,
                0.7914619,
                0.78835863,
                0.78366518,
                0.77709723,
                0.76937795,
                0.76095653,
                0.75216033,
                0.74317605,
                0.73369005,
                0.72363062,
                0.71388497,
                0.70445029,
                0.6953509,
                0.68635258,
                0.67745063,
                0.6685214,
                0.65936403,
                0.64891511,
                0.63528666,
                0.62823882,
                0.62823882,
                0.62823882,
                0.62823882,
            ],
            [
                0.43972308,
                0.63827974,
                0.7202443,
                0.76093124,
                0.78241596,
                0.79360912,
                0.79873362,
                0.80045207,
                0.79995125,
                0.79749807,
                0.79305204,
                0.7873716,
                0.78096492,
                0.77402273,
                0.76671151,
                0.75897546,
                0.75063723,
                0.74221685,
                0.73399596,
                0.72596405,
                0.71797537,
                0.71006195,
                0.70215295,
                0.69419953,
                0.68604987,
                0.67696452,
                0.66556421,
                0.65482192,
                0.65482192,
                0.65482192,
            ],
            [
                0.402497,
                0.61725279,
                0.70687777,
                0.75275184,
                0.77802324,
                0.79255757,
                0.80073543,
                0.80515805,
                0.8069356,
                0.8063223,
                0.80382137,
                0.79998908,
                0.79530264,
                0.78998716,
                0.78420113,
                0.77793393,
                0.77111327,
                0.76401911,
                0.75691526,
                0.74987942,
                0.74287332,
                0.73588059,
                0.72885322,
                0.72182983,
                0.71472761,
                0.70748583,
                0.69954923,
                0.69011428,
                0.67805898,
                0.67805898,
            ],
            [
                0.39716978,
                0.59877547,
                0.69265353,
                0.74311866,
                0.77216736,
                0.78956663,
                0.80054612,
                0.80726755,
                0.81099899,
                0.81205143,
                0.81122227,
                0.8089605,
                0.80575667,
                0.80186509,
                0.79741921,
                0.79246379,
                0.78697748,
                0.7810629,
                0.77498992,
                0.7688943,
                0.76279279,
                0.75662314,
                0.75040965,
                0.74416809,
                0.73787951,
                0.73156531,
                0.72508226,
                0.71815051,
                0.71029578,
                0.69669875,
            ],
        ]
    )
    assert np.sum(
        np.abs(
            thrust_SL - problem.get_val("data:aerodynamics:propeller:sea_level:thrust", units="N")
        )
        < 1
    ) == np.size(problem["data:aerodynamics:propeller:sea_level:thrust"])
    assert np.sum(
        np.abs(
            trust_SL_limit
            - problem.get_val("data:aerodynamics:propeller:sea_level:thrust_limit", units="N")
        )
        < 1
    ) == np.size(problem["data:aerodynamics:propeller:sea_level:thrust_limit"])
    assert np.sum(
        np.abs(speed - problem.get_val("data:aerodynamics:propeller:sea_level:speed", units="m/s"))
        < 1e-2
    ) == np.size(problem["data:aerodynamics:propeller:sea_level:speed"])
    assert np.sum(
        np.abs(efficiency_SL - problem["data:aerodynamics:propeller:sea_level:efficiency"]) < 1e-5
    ) == np.size(problem["data:aerodynamics:propeller:sea_level:efficiency"])
    thrust_CL = np.array(
        [
            134.97348152,
            323.30379616,
            511.6341108,
            699.96442543,
            888.29474007,
            1076.62505471,
            1264.95536935,
            1453.28568399,
            1641.61599862,
            1829.94631326,
            2018.2766279,
            2206.60694254,
            2394.93725717,
            2583.26757181,
            2771.59788645,
            2959.92820109,
            3148.25851573,
            3336.58883036,
            3524.919145,
            3713.24945964,
            3901.57977428,
            4089.91008891,
            4278.24040355,
            4466.57071819,
            4654.90103283,
            4843.23134747,
            5031.5616621,
            5219.89197674,
            5408.22229138,
            5596.55260602,
        ]
    )
    trust_CL_limit = np.array(
        [
            3369.52146682,
            3713.84681718,
            3994.44371797,
            4232.80434314,
            4449.048829,
            4657.59176872,
            4869.40407567,
            5092.6160618,
            5333.42818573,
            5596.55260602,
        ]
    )
    efficiency_CL = np.array(
        [
            [
                0.10922609,
                0.19156517,
                0.22346688,
                0.227992,
                0.22163671,
                0.21138177,
                0.20011264,
                0.18921923,
                0.17914821,
                0.16947165,
                0.16009817,
                0.15143926,
                0.14324954,
                0.13568378,
                0.12911094,
                0.12307313,
                0.11707481,
                0.11025894,
                0.10866362,
                0.10866362,
                0.10866362,
                0.10866362,
                0.10866362,
                0.10866362,
                0.10866362,
                0.10866362,
                0.10866362,
                0.10866362,
                0.10866362,
                0.10866362,
            ],
            [
                0.28040602,
                0.45515005,
                0.51314056,
                0.52409961,
                0.51576474,
                0.49974938,
                0.48113023,
                0.46228684,
                0.44434354,
                0.42695582,
                0.40924073,
                0.3923912,
                0.3762071,
                0.36039601,
                0.34558428,
                0.33264671,
                0.32057296,
                0.30871536,
                0.29618005,
                0.2782013,
                0.27810362,
                0.27810362,
                0.27810362,
                0.27810362,
                0.27810362,
                0.27810362,
                0.27810362,
                0.27810362,
                0.27810362,
                0.27810362,
            ],
            [
                0.39803704,
                0.59157321,
                0.65125996,
                0.66537505,
                0.66087619,
                0.64807268,
                0.63154642,
                0.61378854,
                0.59686822,
                0.58045116,
                0.5624672,
                0.54476017,
                0.52777909,
                0.51109489,
                0.49385233,
                0.47821461,
                0.46441105,
                0.45108669,
                0.43804161,
                0.42448571,
                0.40766183,
                0.39160356,
                0.39160356,
                0.39160356,
                0.39160356,
                0.39160356,
                0.39160356,
                0.39160356,
                0.39160356,
                0.39160356,
            ],
            [
                0.45916243,
                0.65395271,
                0.71318185,
                0.73107917,
                0.73184382,
                0.72414932,
                0.71223764,
                0.69839297,
                0.68426603,
                0.67119117,
                0.65594941,
                0.64008494,
                0.62461096,
                0.60946013,
                0.59416957,
                0.57796795,
                0.56393477,
                0.55110383,
                0.53840117,
                0.52585945,
                0.51304413,
                0.49733099,
                0.47662636,
                0.47662636,
                0.47662636,
                0.47662636,
                0.47662636,
                0.47662636,
                0.47662636,
                0.47662636,
            ],
            [
                0.47639849,
                0.67426248,
                0.73601062,
                0.75941227,
                0.7655009,
                0.76304655,
                0.7559946,
                0.74641048,
                0.73609125,
                0.7259463,
                0.71402375,
                0.70089677,
                0.68761412,
                0.67444175,
                0.66140783,
                0.64746971,
                0.63380236,
                0.6215912,
                0.61009202,
                0.59868587,
                0.58726593,
                0.57549363,
                0.56151121,
                0.54101776,
                0.54101776,
                0.54101776,
                0.54101776,
                0.54101776,
                0.54101776,
                0.54101776,
            ],
            [
                0.46736523,
                0.66878346,
                0.73945056,
                0.76836891,
                0.77981332,
                0.78228982,
                0.77971883,
                0.77409042,
                0.76752315,
                0.76004796,
                0.7509044,
                0.74058332,
                0.72971723,
                0.71870047,
                0.70767006,
                0.6960663,
                0.68407106,
                0.67281598,
                0.66222176,
                0.65193165,
                0.64184924,
                0.63168337,
                0.62114258,
                0.60880558,
                0.58982636,
                0.58667987,
                0.58667987,
                0.58667987,
                0.58667987,
                0.58667987,
            ],
            [
                0.45082571,
                0.65473157,
                0.73166471,
                0.76699835,
                0.78369255,
                0.79065995,
                0.79207601,
                0.79011598,
                0.7867329,
                0.78181855,
                0.77505325,
                0.76714458,
                0.75854755,
                0.74958668,
                0.74044332,
                0.7308418,
                0.72064983,
                0.71079578,
                0.70125418,
                0.69207,
                0.68298123,
                0.67398931,
                0.66496852,
                0.6556463,
                0.64489108,
                0.63017004,
                0.6239253,
                0.6239253,
                0.6239253,
                0.6239253,
            ],
            [
                0.43563234,
                0.63793596,
                0.72041389,
                0.7609256,
                0.78211649,
                0.79300781,
                0.79784393,
                0.79922658,
                0.79845732,
                0.79580929,
                0.7911916,
                0.78536283,
                0.77878862,
                0.77168393,
                0.76424194,
                0.75636858,
                0.74792215,
                0.73938067,
                0.7310553,
                0.72292293,
                0.71483768,
                0.70683389,
                0.69883248,
                0.69079867,
                0.68254361,
                0.67326074,
                0.66142208,
                0.65301615,
                0.65301615,
                0.65301615,
            ],
            [
                0.40023561,
                0.61739133,
                0.70706087,
                0.75268564,
                0.77764113,
                0.79192691,
                0.79981091,
                0.80395956,
                0.80548905,
                0.8047377,
                0.80209971,
                0.79812429,
                0.79327771,
                0.78781874,
                0.78190091,
                0.7755328,
                0.76860916,
                0.76138466,
                0.75417472,
                0.74703582,
                0.73993033,
                0.73284046,
                0.7257307,
                0.71862717,
                0.71144639,
                0.70412809,
                0.69611385,
                0.68649552,
                0.67605807,
                0.67605807,
            ],
            [
                0.39068096,
                0.59789789,
                0.69251978,
                0.74288986,
                0.77160993,
                0.7888376,
                0.79952666,
                0.80602495,
                0.8095345,
                0.81051385,
                0.80956383,
                0.80716824,
                0.80382279,
                0.79979626,
                0.795224,
                0.79016688,
                0.78459508,
                0.77858452,
                0.77240574,
                0.76620661,
                0.76000528,
                0.75374342,
                0.74744348,
                0.7411217,
                0.73475828,
                0.7283674,
                0.72183774,
                0.71485892,
                0.70698449,
                0.69214566,
            ],
        ]
    )
    assert np.sum(
        np.abs(
            thrust_CL
            - problem.get_val("data:aerodynamics:propeller:cruise_level:thrust", units="N")
        )
        < 1
    ) == np.size(problem["data:aerodynamics:propeller:cruise_level:thrust"])
    assert np.sum(
        np.abs(
            trust_CL_limit
            - problem.get_val("data:aerodynamics:propeller:cruise_level:thrust_limit", units="N")
        )
        < 1
    ) == np.size(problem["data:aerodynamics:propeller:cruise_level:thrust_limit"])
    assert np.sum(
        np.abs(
            speed - problem.get_val("data:aerodynamics:propeller:cruise_level:speed", units="m/s")
        )
        < 1e-2
    ) == np.size(problem["data:aerodynamics:propeller:cruise_level:speed"])
    assert np.sum(
        np.abs(efficiency_CL - problem["data:aerodynamics:propeller:cruise_level:efficiency"])
        < 1e-5
    ) == np.size(problem["data:aerodynamics:propeller:cruise_level:efficiency"])
