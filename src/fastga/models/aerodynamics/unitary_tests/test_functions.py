"""Test module for aerodynamics groups."""
#  This file is part of FAST-OAD_CS23 : A framework for rapid Overall Aircraft Design
#  Copyright (C) 2026  ONERA & ISAE-SUPAERO
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

import logging
import pathlib
import shutil
import time
from platform import system

import numpy as np
import openmdao.api as om
import pytest

from fastga.command.api import _create_tmp_directory
from fastga.models.aerodynamics.aerodynamics_high_speed import AerodynamicsHighSpeed
from fastga.models.aerodynamics.aerodynamics_low_speed import AerodynamicsLowSpeed
from fastga.models.aerodynamics.components import (
    Compute2DHingeMomentsTail,
    Compute3DHingeMomentsTail,
    ComputeAircraftMaxCl,
    ComputeAirfoilLiftCurveSlope,
    ComputeCLAlphaDotAircraft,
    ComputeCLPitchVelocityAircraft,
    ComputeCMAlphaDotAircraft,
    ComputeCMPitchVelocityAircraft,
    ComputeCYBetaAircraft,
    ComputeClBetaAircraft,
    ComputeClDeltaAileron,
    ComputeClDeltaRudder,
    ComputeClRollRateAircraft,
    ComputeClYawRateAircraft,
    ComputeCnBetaAircraft,
    ComputeCnDeltaAileron,
    ComputeCnDeltaRudder,
    ComputeCnRollRateAircraft,
    ComputeCyDeltaRudder,
    ComputeCyRollRateAircraft,
    ComputeCyYawRateAircraft,
    ComputeDeltaElevator,
    ComputeDeltaHighLift,
    ComputeEffectiveEfficiencyPropeller,
    ComputeEquilibratedPolar,
    ComputeExtremeCLHtp,
    ComputeExtremeCLWing,
    ComputeHingeMomentsTail,
    ComputeLDMax,
    ComputeMachInterpolation,
    ComputeNonEquilibratedPolar,
    ComputeUnitReynolds,
    ComputeVNAndVH,
)
from fastga.models.aerodynamics.components.cd0 import Cd0
from fastga.models.aerodynamics.components.compute_cn_yaw_rate import ComputeCnYawRateAircraft
from fastga.models.aerodynamics.components.compute_equilibrated_polar import FIRST_INVALID_COEFF
from fastga.models.aerodynamics.components.fuselage import (
    ComputeCmAlphaFuselage,
    ComputeCnBetaFuselage,
    ComputeCyBetaFuselage,
)
from fastga.models.aerodynamics.components.ht import (
    ComputeCLPitchVelocityHorizontalTail,
    ComputeCMPitchVelocityHorizontalTail,
    ComputeClBetaHorizontalTail,
    ComputeClRollRateHorizontalTail,
    DownWashGradientComputation,
)
from fastga.models.aerodynamics.components.vt import (
    ComputeClAlphaVerticalTail,
    ComputeClBetaVerticalTail,
    ComputeClRollRateVerticalTail,
    ComputeClYawRateVerticalTail,
    ComputeCnBetaVerticalTail,
    ComputeCnRollRateVerticalTail,
    ComputeCnYawRateVerticalTail,
    ComputeCyBetaVerticalTail,
)
from fastga.models.aerodynamics.components.wing import (
    ComputeCLPitchVelocityWing,
    ComputeCMPitchVelocityWing,
    ComputeClBetaWing,
    ComputeClRollRateWing,
    ComputeClYawRateWing,
    ComputeCnRollRateWing,
    ComputeCnYawRateWing,
    ComputeCyBetaWing,
)
from fastga.models.aerodynamics.external.neuralfoil.neuralfoil_polar import NeuralfoilPolar
from fastga.models.aerodynamics.external.openvsp import ComputeAeroOpenVSP, OpenVSPSimpleGeometry
from fastga.models.aerodynamics.external.openvsp.compute_aero_slipstream import (
    ComputeSlipstreamOpenvsp,
)
from fastga.models.aerodynamics.external.propeller_code.compute_propeller_aero import (
    ComputePropellerPerformance,
)
from fastga.models.aerodynamics.external.vlm import ComputeAeroVLM, VLMSimpleGeometry
from fastga.models.aerodynamics.external.xfoil import resources
from fastga.models.aerodynamics.external.xfoil.xfoil_polar import XfoilPolar
from fastga.models.aerodynamics.load_factor import LoadFactor
from tests.testing_utilities import (
    get_indep_var_comp,
    list_inputs,
    run_system,
    setup_and_run_system,
)
from tests.xfoil_exe.get_xfoil import get_xfoil_path

RESULTS_FOLDER = pathlib.Path(__file__).parent / "results"
DATA_FOLDER = pathlib.Path(__file__).parent / "data"
TMP_SAVE_FOLDER = "test_save"
xfoil_path = None if system() == "Windows" else get_xfoil_path()

_LOGGER = logging.getLogger(__name__)


def reshape_curve(y, cl):
    """Reshape data from openvsp/vlm lift curve!"""
    for idx in range(len(y)):
        if np.sum(y[idx : len(y)] == 0) == (len(y) - idx):
            y = y[0:idx]
            cl = cl[0:idx]
            break

    return y, cl


def reshape_polar(cl, cdp):
    """Reshape data from xfoil polar vectors!"""
    for idx in range(len(cl)):
        if np.sum(cl[idx : len(cl)] == 0) == (len(cl) - idx):
            cl = cl[0:idx]
            cdp = cdp[0:idx]
            break
    return cl, cdp


def polar_result_transfer():
    # Put saved polar results in a temporary folder to activate Xfoil run and have repeatable
    # results [need writing permission]

    tmp_folder = _create_tmp_directory()

    files = pathlib.Path(resources.__path__[0]).glob("*.csv")

    for file in files:
        if file.is_file():
            shutil.copy(file, tmp_folder.name)
            # noinspection PyBroadException
            try:
                file.unlink()
            except OSError:
                _LOGGER.info(f"Cannot remove {file.as_posix()} file!")

    return tmp_folder


def polar_result_retrieve(tmp_folder):
    # Retrieve the polar results set aside during the test duration if there are some [need
    # writing permission]

    files = pathlib.Path(tmp_folder.name).glob("*.csv")

    for file in files:
        if file.is_file():
            # noinspection PyBroadException
            try:
                shutil.copy(file, resources.__path__[0])
            except (OSError, shutil.SameFileError) as e:
                if isinstance(e, OSError):
                    _LOGGER.info(
                        f"Cannot copy {file.as_posix()} file to {tmp_folder.name}! "
                        f"Likely due to permission error"
                    )
                else:
                    _LOGGER.info(
                        f"Cannot copy {file.as_posix()} file to {tmp_folder.name}! Likely because "
                        f"the file already exists in the target directory"
                    )

    tmp_folder.cleanup()


def compute_reynolds(
    xml_file_name: str,
    mach_low_speed: float,
    reynolds_low_speed: float,
    mach_high_speed: float,
    reynolds_high_speed: float,
):
    """Tests high and low speed reynolds calculation!"""
    # Research independent input value in .xml file
    problem = setup_and_run_system(
        ComputeUnitReynolds(low_speed_aero=True), __file__, xml_file_name
    )
    assert problem["data:aerodynamics:low_speed:mach"] == pytest.approx(mach_low_speed, abs=1e-4)
    assert problem.get_val(
        "data:aerodynamics:low_speed:unit_reynolds", units="m**-1"
    ) == pytest.approx(reynolds_low_speed, abs=1)

    problem.check_partials(compact_print=True)

    # Research independent input value in .xml file
    problem = setup_and_run_system(
        ComputeUnitReynolds(low_speed_aero=False), __file__, xml_file_name
    )
    assert problem["data:aerodynamics:cruise:mach"] == pytest.approx(mach_high_speed, abs=1e-4)
    assert problem.get_val(
        "data:aerodynamics:cruise:unit_reynolds", units="m**-1"
    ) == pytest.approx(reynolds_high_speed, abs=1)

    problem.check_partials(compact_print=True)


def cd0_high_speed(
    xml_file_name: str,
    engine_wrapper_id: str,
    cd0_wing: float,
    cd0_fus: float,
    cd0_ht: float,
    cd0_vt: float,
    cd0_nac: float,
    cd0_lg: float,
    cd0_other: float,
    cd0_total: float,
):
    """Tests drag coefficient @ high speed!"""
    # Research independent input value in .xml file
    # noinspection PyTypeChecker
    ivc = get_indep_var_comp(
        list_inputs(Cd0(propulsion_id=engine_wrapper_id)), __file__, xml_file_name
    )

    # noinspection PyTypeChecker
    problem = run_system(Cd0(propulsion_id=engine_wrapper_id), ivc)
    assert problem["data:aerodynamics:wing:cruise:CD0"] == pytest.approx(cd0_wing, abs=1e-5)
    assert problem["data:aerodynamics:fuselage:cruise:CD0"] == pytest.approx(cd0_fus, abs=1e-5)
    assert problem["data:aerodynamics:horizontal_tail:cruise:CD0"] == pytest.approx(
        cd0_ht, abs=1e-5
    )
    assert problem["data:aerodynamics:vertical_tail:cruise:CD0"] == pytest.approx(cd0_vt, abs=1e-5)
    assert problem["data:aerodynamics:nacelles:cruise:CD0"] == pytest.approx(cd0_nac, abs=1e-5)
    assert problem["data:aerodynamics:landing_gear:cruise:CD0"] == pytest.approx(cd0_lg, abs=1e-5)
    assert problem["data:aerodynamics:other:cruise:CD0"] == pytest.approx(cd0_other, abs=1e-5)
    cd0_total_cal = 1.25 * (
        problem["data:aerodynamics:wing:cruise:CD0"]
        + problem["data:aerodynamics:fuselage:cruise:CD0"]
        + problem["data:aerodynamics:horizontal_tail:cruise:CD0"]
        + problem["data:aerodynamics:vertical_tail:cruise:CD0"]
        + problem["data:aerodynamics:nacelles:cruise:CD0"]
        + problem["data:aerodynamics:landing_gear:cruise:CD0"]
        + problem["data:aerodynamics:other:cruise:CD0"]
    )
    assert cd0_total_cal == pytest.approx(cd0_total, abs=1e-5)

    # Exclude check on wing, ht and nacelles Cd0 partials as it is computed by fd for now
    problem.check_partials(
        compact_print=True,
        excludes=["*cd0_wing*", "*cd0_ht*", "*cd0_nacelle*"],
    )


def cd0_low_speed(
    xml_file_name: str,
    engine_wrapper_id: str,
    cd0_wing: float,
    cd0_fus: float,
    cd0_ht: float,
    cd0_vt: float,
    cd0_nac: float,
    cd0_lg: float,
    cd0_other: float,
    cd0_total: float,
):
    """Tests drag coefficient @ low speed!"""
    # Research independent input value in .xml file
    # noinspection PyTypeChecker
    ivc = get_indep_var_comp(
        list_inputs(Cd0(propulsion_id=engine_wrapper_id, low_speed_aero=True)),
        __file__,
        xml_file_name,
    )

    # noinspection PyTypeChecker
    problem = run_system(Cd0(propulsion_id=engine_wrapper_id, low_speed_aero=True), ivc)
    assert problem["data:aerodynamics:wing:low_speed:CD0"] == pytest.approx(cd0_wing, abs=1e-5)
    assert problem["data:aerodynamics:fuselage:low_speed:CD0"] == pytest.approx(cd0_fus, abs=1e-5)
    assert problem["data:aerodynamics:horizontal_tail:low_speed:CD0"] == pytest.approx(
        cd0_ht, abs=1e-5
    )
    assert problem["data:aerodynamics:vertical_tail:low_speed:CD0"] == pytest.approx(
        cd0_vt, abs=1e-5
    )
    assert problem["data:aerodynamics:nacelles:low_speed:CD0"] == pytest.approx(cd0_nac, abs=1e-5)
    assert problem["data:aerodynamics:landing_gear:low_speed:CD0"] == pytest.approx(
        cd0_lg, abs=1e-5
    )
    assert problem["data:aerodynamics:other:low_speed:CD0"] == pytest.approx(cd0_other, abs=1e-5)
    cd0_total_cal = 1.25 * (
        problem["data:aerodynamics:wing:low_speed:CD0"]
        + problem["data:aerodynamics:fuselage:low_speed:CD0"]
        + problem["data:aerodynamics:horizontal_tail:low_speed:CD0"]
        + problem["data:aerodynamics:vertical_tail:low_speed:CD0"]
        + problem["data:aerodynamics:nacelles:low_speed:CD0"]
        + problem["data:aerodynamics:landing_gear:low_speed:CD0"]
        + problem["data:aerodynamics:other:low_speed:CD0"]
    )
    assert cd0_total_cal == pytest.approx(cd0_total, abs=1e-5)

    # Exclude check on wing, ht and nacelles Cd0 partials as it is computed by fd for now
    problem.check_partials(
        compact_print=True,
        excludes=["*cd0_wing*", "*cd0_ht*", "*cd0_nacelle*"],
    )


def polar_xfoil(
    xml_file_name: str,
    mach_high_speed: float,
    reynolds_high_speed: float,
    mach_low_speed: float,
    reynolds_low_speed: float,
    cdp_1_high_speed: float,
    cl_max_2d: float,
    cdp_1_low_speed: float,
):
    """Tests polar execution (XFOIL) @ high and low speed!"""
    # Transfer saved polar results to temporary folder
    tmp_folder = polar_result_transfer()

    # Define high-speed parameters (with .xml file and additional inputs)
    ivc = get_indep_var_comp(list_inputs(XfoilPolar()), __file__, xml_file_name)
    ivc.add_output("mach", mach_high_speed, units="unitless")
    ivc.add_output("reynolds", reynolds_high_speed, units="unitless")

    # Run problem
    xfoil_comp = XfoilPolar(
        alpha_start=0.0, alpha_end=20.0, iter_limit=20, xfoil_exe_path=xfoil_path
    )
    problem = run_system(xfoil_comp, ivc)

    # Retrieve polar results from temporary folder
    polar_result_retrieve(tmp_folder)

    # Check obtained value(s) is/(are) correct
    cl = problem["CL"]
    cdp = problem["CDp"]
    cl, cdp = reshape_polar(cl, cdp)
    assert np.interp(1.0, cl, cdp) == pytest.approx(cdp_1_high_speed, abs=1e-4)

    # Transfer saved polar results to temporary folder
    tmp_folder = polar_result_transfer()

    # Define low-speed parameters (with .xml file and additional inputs)
    ivc = get_indep_var_comp(list_inputs(XfoilPolar()), __file__, xml_file_name)
    ivc.add_output("mach", mach_low_speed, units="unitless")
    ivc.add_output("reynolds", reynolds_low_speed, units="unitless")

    # Run problem
    xfoil_comp = XfoilPolar(
        alpha_start=0.0, alpha_end=25.0, iter_limit=20, xfoil_exe_path=xfoil_path
    )
    problem = run_system(xfoil_comp, ivc)

    # Retrieve polar results from temporary folder
    polar_result_retrieve(tmp_folder)

    # Check obtained value(s) is/(are) correct
    cl = problem["CL"]
    cdp = problem["CDp"]
    assert problem["CL_max_2D"] == pytest.approx(cl_max_2d, abs=1e-4)
    cl, cdp = reshape_polar(cl, cdp)
    assert np.interp(1.0, cl, cdp) == pytest.approx(cdp_1_low_speed, abs=1e-4)


def polar_neuralfoil(
    xml_file_name: str,
    mach_high_speed: float,
    reynolds_high_speed: float,
    mach_low_speed: float,
    reynolds_low_speed: float,
    cdp_1_high_speed: float,
    cl_max_2d: float,
    cdp_1_low_speed: float,
):
    """Tests polar execution (NeuralFOIL) @ high and low speed!"""
    # Define high-speed parameters (with .xml file and additional inputs)
    ivc = get_indep_var_comp(list_inputs(NeuralfoilPolar()), __file__, xml_file_name)
    ivc.add_output("mach", mach_high_speed, units="unitless")
    ivc.add_output("reynolds", reynolds_high_speed, units="unitless")

    # Run problem
    neuralfoil_comp = NeuralfoilPolar(alpha_start=0.0, alpha_end=20.0)
    problem = run_system(neuralfoil_comp, ivc)

    # Check obtained value(s) is/(are) correct
    cl = problem["CL"]
    cdp = problem["CDp"]
    cl, cdp = reshape_polar(cl, cdp)
    assert np.interp(1.0, cl, cdp) == pytest.approx(cdp_1_high_speed, abs=1e-4)

    # Define low-speed parameters (with .xml file and additional inputs)
    ivc = get_indep_var_comp(list_inputs(NeuralfoilPolar()), __file__, xml_file_name)
    ivc.add_output("mach", mach_low_speed, units="unitless")
    ivc.add_output("reynolds", reynolds_low_speed, units="unitless")

    # Run problem
    neuralfoil_comp = NeuralfoilPolar(alpha_start=0.0, alpha_end=25.0)
    problem = run_system(neuralfoil_comp, ivc)
    # Check obtained value(s) is/(are) correct
    cl = problem["CL"]
    cdp = problem["CDp"]
    assert problem["CL_max_2D"] == pytest.approx(cl_max_2d, abs=1e-4)
    cl, cdp = reshape_polar(cl, cdp)
    assert np.interp(1.0, cl, cdp) == pytest.approx(cdp_1_low_speed, abs=1e-4)


def polar_interpolation(mach: float):
    """
    Tests the interpolation mechanism from XfoilPolar. To do so we will run XfoilPolar twice and
    then a third time a Reynolds number in between, it should trigger the interpolation mechanism.
    """

    # Transfer saved polar results to temporary folder
    tmp_folder = polar_result_transfer()

    ivc = om.IndepVarComp()
    ivc.add_output("mach", mach, units="unitless")
    ivc.add_output("reynolds", 5e6, units="unitless")

    # Run problem
    xfoil_comp = XfoilPolar(
        alpha_start=0.0, alpha_end=20.0, iter_limit=20, xfoil_exe_path=xfoil_path
    )
    t1_start = time.time()
    _ = run_system(xfoil_comp, ivc)
    t1_end = time.time()
    t1_duration = t1_end - t1_start

    ivc = om.IndepVarComp()
    ivc.add_output("mach", mach, units="unitless")
    ivc.add_output("reynolds", 7e6, units="unitless")
    t2_start = time.time()
    _ = run_system(xfoil_comp, ivc)
    t2_end = time.time()
    t2_duration = t2_end - t2_start

    # Run a third time between the two other Reynolds

    ivc = om.IndepVarComp()
    ivc.add_output("mach", mach, units="unitless")
    ivc.add_output("reynolds", 6e6, units="unitless")

    # Run problem
    t3_start = time.time()
    _ = run_system(xfoil_comp, ivc)
    t3_end = time.time()
    t3_duration = t3_end - t3_start

    # Retrieve polar results from temporary folder
    polar_result_retrieve(tmp_folder)

    assert t3_duration < (t1_duration + t2_duration) / 2


def polar_single_aoa_xfoil(
    xml_file_name: str,
    mach_low_speed: float,
    reynolds_low_speed: float,
):
    """
    Tests polar execution (XFOIL) @ low speed! Run Xfoil once with multiple AOA then extract the
    value for 5.0 degree and run Xfoil for a single AoA at 5.0 degree and compare results
    """

    # Transfer saved polar results to temporary folder
    tmp_folder = polar_result_transfer()

    # Define low-speed parameters (with .xml file and additional inputs)
    ivc = get_indep_var_comp(list_inputs(XfoilPolar()), __file__, xml_file_name)
    ivc.add_output("mach", mach_low_speed, units="unitless")
    ivc.add_output("reynolds", reynolds_low_speed, units="unitless")

    # Run problem
    xfoil_comp = XfoilPolar(
        alpha_start=2.0, alpha_end=7.5, iter_limit=20, xfoil_exe_path=xfoil_path
    )
    problem = run_system(xfoil_comp, ivc)

    # Retrieve polar results from temporary folder
    polar_result_retrieve(tmp_folder)

    # Extract value for 5.0 deg
    cl = problem["CL"]
    cdp = problem["CDp"]

    alpha = problem["alpha"]
    index_5_deg = list(alpha).index(5.0)

    cl, cdp = reshape_polar(cl, cdp)

    cl_5 = cl[index_5_deg]
    cdp_5 = cdp[index_5_deg]

    # Transfer saved polar results to temporary folder
    tmp_folder = polar_result_transfer()

    # Define high-speed parameters (with .xml file and additional inputs)
    ivc = get_indep_var_comp(list_inputs(XfoilPolar()), __file__, xml_file_name)
    ivc.add_output("mach", mach_low_speed, units="unitless")
    ivc.add_output("reynolds", reynolds_low_speed, units="unitless")
    # Run problem
    xfoil_comp = XfoilPolar(
        alpha_start=5.0, iter_limit=20, xfoil_exe_path=xfoil_path, single_AoA=True
    )
    problem = run_system(xfoil_comp, ivc)

    # Retrieve polar results from temporary folder
    polar_result_retrieve(tmp_folder)

    # Check obtained value(s) is/(are) correct
    cl_s = problem["CL"]
    cdp_s = problem["CDp"]
    assert cl_5 == pytest.approx(cl_s, abs=1e-4)
    assert cdp_5 == pytest.approx(cdp_s, abs=1e-4)


def polar_single_aoa_neuralfoil(
    xml_file_name: str,
    mach_low_speed: float,
    reynolds_low_speed: float,
    alpha: float,
    cl: float,
):
    """
    Tests polar execution (NeuralFoil) @ low speed! Run Neuralfoil once with multiple AOA then
    extract the value for 5.0 degree and run it for a single AoA at 5.0 degree and compare
    results
    """

    # Transfer saved polar results to temporary folder
    tmp_folder = polar_result_transfer()

    # Define high-speed parameters (with .xml file and additional inputs)
    ivc = get_indep_var_comp(list_inputs(NeuralfoilPolar()), __file__, xml_file_name)
    ivc.add_output("mach", mach_low_speed, units="unitless")
    ivc.add_output("reynolds", reynolds_low_speed, units="unitless")
    # Run problem
    nfoil_comp = NeuralfoilPolar(alpha_start=alpha, single_AoA=True)
    problem = run_system(nfoil_comp, ivc)

    # Retrieve polar results from temporary folder
    polar_result_retrieve(tmp_folder)

    # Check obtained value(s) is/(are) correct
    cl_s = problem["CL"]
    assert cl == pytest.approx(cl_s, abs=1e-3)  # tolerance increased


def polar_single_aoa_inv(
    xml_file_name: str,
    mach_low_speed: float,
    reynolds_low_speed: float,
):
    """
    Tests polar execution (XFOIL) @ low speed and inviscid! Run Xfoil once with multiple AOA then
    extract the value for 5.0 degree and run Xfoil for a single Aoa at 5.0 degree and compare
    results
    """

    # Transfer saved polar results to temporary folder
    tmp_folder = polar_result_transfer()

    # Define low-speed parameters (with .xml file and additional inputs)
    ivc = get_indep_var_comp(list_inputs(XfoilPolar()), __file__, xml_file_name)
    ivc.add_output("mach", mach_low_speed, units="unitless")
    ivc.add_output("reynolds", reynolds_low_speed, units="unitless")

    # Run problem
    xfoil_comp = XfoilPolar(
        alpha_start=2.5,
        alpha_end=7.5,
        iter_limit=20,
        xfoil_exe_path=xfoil_path,
        inviscid_calculation=True,
    )
    problem = run_system(xfoil_comp, ivc)
    # Retrieve polar results from temporary folder
    polar_result_retrieve(tmp_folder)

    # Check obtained value(s) is/(are) correct
    cl = problem["CL"]
    cdp = problem["CDp"]

    alpha = problem["alpha"]
    index_5_deg = list(alpha).index(5.0)

    cl, cdp = reshape_polar(cl, cdp)

    cl_5 = cl[index_5_deg]
    cdp_5 = cdp[index_5_deg]

    # Transfer saved polar results to temporary folder
    tmp_folder = polar_result_transfer()

    # Define high-speed parameters (with .xml file and additional inputs)
    ivc = get_indep_var_comp(list_inputs(XfoilPolar()), __file__, xml_file_name)
    ivc.add_output("mach", mach_low_speed, units="unitless")
    ivc.add_output("reynolds", reynolds_low_speed, units="unitless")
    # Run problem
    xfoil_comp = XfoilPolar(
        alpha_start=5.0,
        iter_limit=20,
        xfoil_exe_path=xfoil_path,
        inviscid_calculation=True,
        single_AoA=True,
    )
    problem = run_system(xfoil_comp, ivc)

    # Retrieve polar results from temporary folder
    polar_result_retrieve(tmp_folder)

    # Check obtained value(s) is/(are) correct
    cl_s = problem["CL"]
    cdp_s = problem["CDp"]
    assert cl_5 == pytest.approx(cl_s, abs=1e-4)
    assert cdp_5 == pytest.approx(cdp_s, abs=1e-4)


def polar_ext_folder(
    xml_file_name: str,
    mach_high_speed: float,
    reynolds_high_speed: float,
    mach_low_speed: float,
    reynolds_low_speed: float,
    cdp_1_high_speed: float,
    cl_max_2d: float,
    cdp_1_low_speed: float,
):
    """Tests polar execution (XFOIL) @ high and low speed! with the option airfoil_folder_path"""
    # Transfer saved polar results to temporary folder
    tmp_folder = polar_result_transfer()
    shutil.copy(
        DATA_FOLDER / "sample_airfoil.af", pathlib.Path(tmp_folder.name) / "sample_airfoil.af"
    )

    # Define high-speed parameters (with .xml file and additional inputs)
    ivc = get_indep_var_comp(list_inputs(XfoilPolar()), __file__, xml_file_name)
    ivc.add_output("mach", mach_high_speed, units="unitless")
    ivc.add_output("reynolds", reynolds_high_speed, units="unitless")

    # Run problem
    xfoil_comp = XfoilPolar(
        alpha_start=0.0,
        alpha_end=25.0,
        iter_limit=20,
        xfoil_exe_path=xfoil_path,
        airfoil_folder_path=tmp_folder.name,
        airfoil_file="sample_airfoil.af",
    )
    problem = run_system(xfoil_comp, ivc)

    # Retrieve polar results from temporary folder
    polar_result_retrieve(tmp_folder)

    # Check obtained value(s) is/(are) correct
    cl = problem["CL"]
    cdp = problem["CDp"]
    cl, cdp = reshape_polar(cl, cdp)
    assert np.interp(1.0, cl, cdp) == pytest.approx(cdp_1_high_speed, abs=1e-4)

    # Transfer saved polar results to temporary folder
    tmp_folder = polar_result_transfer()
    shutil.copy(
        DATA_FOLDER / "sample_airfoil.af", pathlib.Path(tmp_folder.name) / "sample_airfoil.af"
    )

    # Define low-speed parameters (with .xml file and additional inputs)
    ivc = get_indep_var_comp(list_inputs(XfoilPolar()), __file__, xml_file_name)
    ivc.add_output("mach", mach_low_speed, units="unitless")
    ivc.add_output("reynolds", reynolds_low_speed, units="unitless")

    # Run problem
    xfoil_comp = XfoilPolar(
        alpha_start=0.0,
        alpha_end=25.0,
        iter_limit=20,
        xfoil_exe_path=xfoil_path,
        airfoil_folder_path=tmp_folder.name,
        airfoil_file="sample_airfoil.af",
    )
    problem = run_system(xfoil_comp, ivc)

    # Retrieve polar results from temporary folder
    polar_result_retrieve(tmp_folder)

    # Check obtained value(s) is/(are) correct
    cl = problem["CL"]
    cdp = problem["CDp"]
    assert problem["CL_max_2D"] == pytest.approx(cl_max_2d, abs=1e-4)
    cl, cdp = reshape_polar(cl, cdp)
    assert np.interp(1.0, cl, cdp) == pytest.approx(cdp_1_low_speed, abs=1e-4)


def polar_ext_folder_inv(
    xml_file_name: str,
    mach_low_speed: float,
    reynolds_low_speed: float,
):
    """Tests polar execution (XFOIL) @ high and low speed! with the option airfoil_folder_path"""
    # Transfer saved polar results to temporary folder
    tmp_folder = polar_result_transfer()
    shutil.copy(
        DATA_FOLDER / "sample_airfoil.af", pathlib.Path(tmp_folder.name) / "sample_airfoil.af"
    )

    # Define high-speed parameters (with .xml file and additional inputs)
    ivc = get_indep_var_comp(list_inputs(XfoilPolar()), __file__, xml_file_name)
    ivc.add_output("mach", mach_low_speed, units="unitless")
    ivc.add_output("reynolds", reynolds_low_speed, units="unitless")

    # Run problem
    xfoil_comp = XfoilPolar(
        alpha_start=5.0,
        alpha_end=25.0,
        iter_limit=20,
        xfoil_exe_path=xfoil_path,
        airfoil_folder_path=tmp_folder.name,
        airfoil_file="sample_airfoil.af",
    )
    xfoil_comp.options["inviscid_calculation"] = True
    problem = run_system(xfoil_comp, ivc)

    # Retrieve polar results from temporary folder
    polar_result_retrieve(tmp_folder)

    # Check obtained value(s) is/(are) correct
    cl_1 = problem["CL"]
    cdp_1 = problem["CDp"]

    # Transfer saved polar results to temporary folder
    tmp_folder = polar_result_transfer()
    shutil.copy(
        DATA_FOLDER / "sample_airfoil.af", pathlib.Path(tmp_folder.name) / "sample_airfoil.af"
    )

    # Define high-speed parameters (with .xml file and additional inputs)
    ivc = get_indep_var_comp(list_inputs(XfoilPolar()), __file__, xml_file_name)
    ivc.add_output("mach", mach_low_speed, units="unitless")
    ivc.add_output("reynolds", reynolds_low_speed, units="unitless")

    # Run problem
    xfoil_comp = XfoilPolar(
        alpha_start=5.0,
        iter_limit=20,
        xfoil_exe_path=xfoil_path,
        airfoil_folder_path=tmp_folder.name,
        airfoil_file="sample_airfoil.af",
    )
    xfoil_comp.options["inviscid_calculation"] = True
    xfoil_comp.options["single_AoA"] = True
    problem = run_system(xfoil_comp, ivc)

    # Retrieve polar results from temporary folder
    polar_result_retrieve(tmp_folder)

    # Check obtained value(s) is/(are) correct
    cl_2 = problem["CL"]
    cdp_2 = problem["CDp"]
    assert cl_1[0] == pytest.approx(cl_2, abs=1e-4)
    assert cdp_1[0] == pytest.approx(cdp_2, abs=1e-4)


def polar_ext_folder_neuralfoil(
    xml_file_name: str,
    mach_high_speed: float,
    reynolds_high_speed: float,
    mach_low_speed: float,
    reynolds_low_speed: float,
    cdp_1_high_speed: float,
    cl_max_2d: float,
    cdp_1_low_speed: float,
):
    """Tests polar execution (NeuralFoil) @ high and low speed! with the option
    airfoil_folder_path"""
    # Transfer saved polar results to temporary folder
    tmp_folder = polar_result_transfer()
    shutil.copy(
        DATA_FOLDER / "sample_airfoil.af", pathlib.Path(tmp_folder.name) / "sample_airfoil.af"
    )

    # Define high-speed parameters (with .xml file and additional inputs)
    ivc = get_indep_var_comp(list_inputs(NeuralfoilPolar()), __file__, xml_file_name)
    ivc.add_output("mach", mach_high_speed, units="unitless")
    ivc.add_output("reynolds", reynolds_high_speed, units="unitless")

    # Run problem
    nfoil_comp = NeuralfoilPolar(
        alpha_start=0.0,
        alpha_end=25.0,
        airfoil_folder_path=tmp_folder.name,
        airfoil_file="sample_airfoil.af",
    )
    problem = run_system(nfoil_comp, ivc)

    # Retrieve polar results from temporary folder
    polar_result_retrieve(tmp_folder)

    # Check obtained value(s) is/(are) correct
    cl = problem["CL"]
    cdp = problem["CDp"]
    cl, cdp = reshape_polar(cl, cdp)
    assert np.interp(1.0, cl, cdp) == pytest.approx(cdp_1_high_speed, abs=1e-4)

    # Transfer saved polar results to temporary folder
    tmp_folder = polar_result_transfer()
    shutil.copy(
        DATA_FOLDER / "sample_airfoil.af", pathlib.Path(tmp_folder.name) / "sample_airfoil.af"
    )

    # Define low-speed parameters (with .xml file and additional inputs)
    ivc = get_indep_var_comp(list_inputs(NeuralfoilPolar()), __file__, xml_file_name)
    ivc.add_output("mach", mach_low_speed, units="unitless")
    ivc.add_output("reynolds", reynolds_low_speed, units="unitless")

    # Run problem
    nfoil_comp = NeuralfoilPolar(
        alpha_start=0.0,
        alpha_end=25.0,
        airfoil_folder_path=tmp_folder.name,
        airfoil_file="sample_airfoil.af",
    )
    problem = run_system(nfoil_comp, ivc)

    # Retrieve polar results from temporary folder
    polar_result_retrieve(tmp_folder)

    # Check obtained value(s) is/(are) correct
    cl = problem["CL"]
    cdp = problem["CDp"]
    assert problem["CL_max_2D"] == pytest.approx(cl_max_2d, abs=1e-4)
    cl, cdp = reshape_polar(cl, cdp)
    assert np.interp(1.0, cl, cdp) == pytest.approx(cdp_1_low_speed, abs=1e-4)


def airfoil_slope_wt_xfoil(
    xml_file_name: str,
    wing_airfoil_file: str,
    htp_airfoil_file: str,
    vtp_airfoil_file: str,
):
    """Tests polar execution (XFOIL) @ high speed!"""
    # Define high-speed parameters (with .xml file and additional inputs)
    ivc = get_indep_var_comp(
        list_inputs(
            ComputeAirfoilLiftCurveSlope(
                wing_airfoil_file=wing_airfoil_file,
                htp_airfoil_file=htp_airfoil_file,
                vtp_airfoil_file=vtp_airfoil_file,
            )
        ),
        __file__,
        xml_file_name,
    )

    # Run problem and return for complementary values check
    return run_system(
        ComputeAirfoilLiftCurveSlope(
            wing_airfoil_file=wing_airfoil_file,
            htp_airfoil_file=htp_airfoil_file,
            vtp_airfoil_file=vtp_airfoil_file,
        ),
        ivc,
    )


def airfoil_slope_wt_neuralfoil(
    xml_file_name: str,
    wing_airfoil_file: str,
    htp_airfoil_file: str,
    vtp_airfoil_file: str,
):
    """Tests polar execution (NeuralFoil) @ high speed!"""
    # Define high-speed parameters (with .xml file and additional inputs)
    ivc = get_indep_var_comp(
        list_inputs(
            ComputeAirfoilLiftCurveSlope(
                wing_airfoil_file=wing_airfoil_file,
                htp_airfoil_file=htp_airfoil_file,
                vtp_airfoil_file=vtp_airfoil_file,
                use_neuralfoil=True,
            )
        ),
        __file__,
        xml_file_name,
    )

    # Run problem and return for complementary values check
    return run_system(
        ComputeAirfoilLiftCurveSlope(
            wing_airfoil_file=wing_airfoil_file,
            htp_airfoil_file=htp_airfoil_file,
            vtp_airfoil_file=vtp_airfoil_file,
            use_neuralfoil=True,
        ),
        ivc,
    )


def airfoil_slope_xfoil(
    xml_file_name: str,
    wing_airfoil_file: str,
    htp_airfoil_file: str,
    vtp_airfoil_file: str,
    cl_alpha_wing: float,
    cl_alpha_htp: float,
    cl_alpha_vtp: float,
):
    """Tests polar reading @ high speed!"""
    # Transfer saved polar results to temporary folder
    tmp_folder = polar_result_transfer()

    problem = airfoil_slope_wt_xfoil(
        xml_file_name,
        wing_airfoil_file,
        htp_airfoil_file,
        vtp_airfoil_file,
    )

    # Retrieve polar results from temporary folder
    polar_result_retrieve(tmp_folder)

    # Check obtained value(s) is/(are) correct
    assert problem.get_val(
        "data:aerodynamics:wing:airfoil:CL_alpha", units="rad**-1"
    ) == pytest.approx(cl_alpha_wing, abs=1e-4)
    assert problem.get_val(
        "data:aerodynamics:horizontal_tail:airfoil:CL_alpha", units="rad**-1"
    ) == pytest.approx(cl_alpha_htp, abs=1e-4)
    assert problem.get_val(
        "data:aerodynamics:vertical_tail:airfoil:CL_alpha", units="rad**-1"
    ) == pytest.approx(cl_alpha_vtp, abs=1e-4)


def airfoil_slope_neuralfoil(
    xml_file_name: str,
    wing_airfoil_file: str,
    htp_airfoil_file: str,
    vtp_airfoil_file: str,
    cl_alpha_wing: float,
    cl_alpha_htp: float,
    cl_alpha_vtp: float,
):
    """Tests polar reading @ high speed!"""
    # Transfer saved polar results to temporary folder
    tmp_folder = polar_result_transfer()

    problem = airfoil_slope_wt_neuralfoil(
        xml_file_name,
        wing_airfoil_file,
        htp_airfoil_file,
        vtp_airfoil_file,
    )

    # Retrieve polar results from temporary folder
    polar_result_retrieve(tmp_folder)

    # Check obtained value(s) is/(are) correct
    assert problem.get_val(
        "data:aerodynamics:wing:airfoil:CL_alpha", units="rad**-1"
    ) == pytest.approx(cl_alpha_wing, abs=1e-4)
    assert problem.get_val(
        "data:aerodynamics:horizontal_tail:airfoil:CL_alpha", units="rad**-1"
    ) == pytest.approx(cl_alpha_htp, abs=1e-4)
    assert problem.get_val(
        "data:aerodynamics:vertical_tail:airfoil:CL_alpha", units="rad**-1"
    ) == pytest.approx(cl_alpha_vtp, abs=1e-4)


def compute_aero(
    xml_file_name: str,
    *,
    use_openvsp: bool,
    mach_interpolation: bool,
    low_speed_aero: bool,
):
    """Compute aero components!"""
    # Create result temporary directory
    results_folder = _create_tmp_directory()

    # Transfer saved polar results to temporary folder
    tmp_folder = polar_result_transfer()

    # Research independent input value in .xml file
    if use_openvsp:
        OpenVSPSimpleGeometry._cache.clear()
        # noinspection PyTypeChecker
        ivc = get_indep_var_comp(
            list_inputs(ComputeAeroOpenVSP(low_speed_aero=low_speed_aero)), __file__, xml_file_name
        )

        # Run problem twice
        start = time.time()
        # noinspection PyTypeChecker
        openvsp_comp = ComputeAeroOpenVSP(
            low_speed_aero=low_speed_aero,
            result_folder_path=results_folder.name,
            compute_mach_interpolation=mach_interpolation,
        )

        _ = run_system(openvsp_comp, ivc)
        stop = time.time()
        duration_1st_run = stop - start
        start = time.time()
        # noinspection PyTypeChecker
        openvsp_comp = ComputeAeroOpenVSP(
            low_speed_aero=low_speed_aero,
            result_folder_path=results_folder.name,
            compute_mach_interpolation=mach_interpolation,
        )

        problem = run_system(openvsp_comp, ivc)
        stop = time.time()
    else:
        VLMSimpleGeometry._cache.clear()
        # noinspection PyTypeChecker
        ivc = get_indep_var_comp(
            list_inputs(ComputeAeroVLM(low_speed_aero=low_speed_aero)), __file__, xml_file_name
        )

        # Run problem twice
        start = time.time()
        # noinspection PyTypeChecker
        vlm_comp = ComputeAeroVLM(
            low_speed_aero=low_speed_aero,
            result_folder_path=results_folder.name,
            compute_mach_interpolation=mach_interpolation,
        )

        _ = run_system(vlm_comp, ivc)
        stop = time.time()
        duration_1st_run = stop - start
        start = time.time()
        # noinspection PyTypeChecker
        vlm_comp = ComputeAeroVLM(
            low_speed_aero=low_speed_aero,
            result_folder_path=results_folder.name,
            compute_mach_interpolation=mach_interpolation,
        )

        problem = run_system(vlm_comp, ivc)
        stop = time.time()
    duration_2nd_run = stop - start

    # Retrieve polar results from temporary folder
    polar_result_retrieve(tmp_folder)

    # Remove existing result files
    results_folder.cleanup()

    # Check obtained value(s) is/(are) correct
    if use_openvsp:
        assert (duration_2nd_run / duration_1st_run) <= 0.5  # original 0.1
    else:
        assert (duration_2nd_run / duration_1st_run) <= 1
    # Return problem for complementary values check
    return problem


def compute_aero_neuralfoil(
    xml_file_name: str,
    *,
    mach_interpolation: bool,
    low_speed_aero: bool,
):
    """Compute aero components!"""
    # Create result temporary directory
    results_folder = _create_tmp_directory()

    # Transfer saved polar results to temporary folder
    tmp_folder = polar_result_transfer()

    # Research independent input value in .xml file
    VLMSimpleGeometry._cache.clear()
    ivc = get_indep_var_comp(
        list_inputs(ComputeAeroVLM(low_speed_aero=low_speed_aero, use_neuralfoil=True)),
        __file__,
        xml_file_name,
    )

    # noinspection PyTypeChecker
    vlm_comp = ComputeAeroVLM(
        low_speed_aero=low_speed_aero,
        use_neuralfoil=True,
        result_folder_path=results_folder.name,
        compute_mach_interpolation=mach_interpolation,
    )

    problem = run_system(vlm_comp, ivc)

    # Retrieve polar results from temporary folder
    polar_result_retrieve(tmp_folder)

    # Remove existing result files
    results_folder.cleanup()

    # Return problem for complementary values check
    return problem


def comp_aero_input_aoa(
    xml_file_name: str,
    *,
    use_openvsp: bool,
    mach_interpolation: bool,
    low_speed_aero: bool,
):
    """Compute aero components!"""
    # Create result temporary directory
    results_folder = _create_tmp_directory()

    # Transfer saved polar results to temporary folder
    tmp_folder = polar_result_transfer()

    # Research independent input value in .xml file
    if use_openvsp:
        OpenVSPSimpleGeometry._cache.clear()
        # noinspection PyTypeChecker
        ivc = get_indep_var_comp(
            list_inputs(ComputeAeroOpenVSP(low_speed_aero=low_speed_aero)), __file__, xml_file_name
        )

        # Run problem twice
        start = time.time()
        # noinspection PyTypeChecker
        openvsp_comp = ComputeAeroOpenVSP(
            low_speed_aero=low_speed_aero,
            result_folder_path=results_folder.name,
            compute_mach_interpolation=mach_interpolation,
            input_angle_of_attack=10.5,
        )

        _ = run_system(openvsp_comp, ivc)
        stop = time.time()
        duration_1st_run = stop - start
        start = time.time()
        # noinspection PyTypeChecker
        openvsp_comp = ComputeAeroOpenVSP(
            low_speed_aero=low_speed_aero,
            result_folder_path=results_folder.name,
            compute_mach_interpolation=mach_interpolation,
            input_angle_of_attack=10.5,
        )

        problem = run_system(openvsp_comp, ivc)
        stop = time.time()
    else:
        VLMSimpleGeometry._cache.clear()
        # noinspection PyTypeChecker
        ivc = get_indep_var_comp(
            list_inputs(ComputeAeroVLM(low_speed_aero=low_speed_aero)), __file__, xml_file_name
        )

        # Run problem twice
        start = time.time()
        # noinspection PyTypeChecker
        vlm_comp = ComputeAeroVLM(
            low_speed_aero=low_speed_aero,
            result_folder_path=results_folder.name,
            compute_mach_interpolation=mach_interpolation,
            input_angle_of_attack=10.5,
        )

        _ = run_system(vlm_comp, ivc)
        stop = time.time()
        duration_1st_run = stop - start
        start = time.time()
        # noinspection PyTypeChecker
        vlm_comp = ComputeAeroVLM(
            low_speed_aero=low_speed_aero,
            result_folder_path=results_folder.name,
            compute_mach_interpolation=mach_interpolation,
            input_angle_of_attack=10.5,
        )

        problem = run_system(vlm_comp, ivc)
        stop = time.time()
    duration_2nd_run = stop - start

    # Retrieve polar results from temporary folder
    polar_result_retrieve(tmp_folder)

    # Remove existing result files
    results_folder.cleanup()

    # Check obtained value(s) is/(are) correct
    if use_openvsp:
        assert (duration_2nd_run / duration_1st_run) <= 0.5  # original 0.1
    else:
        assert (duration_2nd_run / duration_1st_run) <= 1
    # Return problem for complementary values check
    return problem


def comp_aero_input_aoa_neuralfoil(
    xml_file_name: str,
    *,
    mach_interpolation: bool,
    low_speed_aero: bool,
):
    """Compute aero components!"""
    # Create result temporary directory
    results_folder = _create_tmp_directory()

    # Transfer saved polar results to temporary folder
    tmp_folder = polar_result_transfer()

    VLMSimpleGeometry._cache.clear()

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(ComputeAeroVLM(low_speed_aero=low_speed_aero)), __file__, xml_file_name
    )

    # Run problem twice
    start = time.time()
    # noinspection PyTypeChecker
    vlm_comp = ComputeAeroVLM(
        low_speed_aero=low_speed_aero,
        result_folder_path=results_folder.name,
        compute_mach_interpolation=mach_interpolation,
        input_angle_of_attack=10.5,
        use_neuralfoil=True,
    )

    _ = run_system(vlm_comp, ivc)
    stop = time.time()
    duration_1st_run = stop - start
    start = time.time()
    # noinspection PyTypeChecker
    vlm_comp = ComputeAeroVLM(
        low_speed_aero=low_speed_aero,
        result_folder_path=results_folder.name,
        compute_mach_interpolation=mach_interpolation,
        input_angle_of_attack=10.5,
        use_neuralfoil=True,
    )

    problem = run_system(vlm_comp, ivc)
    stop = time.time()
    duration_2nd_run = stop - start

    # Retrieve polar results from temporary folder
    polar_result_retrieve(tmp_folder)

    # Remove existing result files
    results_folder.cleanup()

    # Check obtained value(s) is/(are) correct

    assert (duration_2nd_run / duration_1st_run) <= 1
    # Return problem for complementary values check
    return problem


def comp_high_speed_xfoil(
    xml_file_name: str,
    cl0_wing: float,
    cl_ref_wing: float,
    cl_alpha_wing: float,
    cm0: float,
    coeff_k_wing: float,
    cl0_htp: float,
    cl_alpha_htp: float,
    cl_alpha_htp_isolated: float,
    coeff_k_htp: float,
    cl_alpha_vector: np.ndarray,
    mach_vector: np.ndarray,
    *,
    use_openvsp: bool,
):
    """Tests components @ high speed!"""
    for mach_interpolation in [False, True]:
        problem = compute_aero(
            xml_file_name,
            use_openvsp=use_openvsp,
            mach_interpolation=mach_interpolation,
            low_speed_aero=False,
        )

        # Check obtained value(s) is/(are) correct
        if mach_interpolation:
            assert problem[
                "data:aerodynamics:aircraft:mach_interpolation:CL_alpha_vector"
            ] == pytest.approx(cl_alpha_vector, abs=1e-2)
            assert problem[
                "data:aerodynamics:aircraft:mach_interpolation:mach_vector"
            ] == pytest.approx(mach_vector, abs=1e-2)
        else:
            assert problem["data:aerodynamics:wing:cruise:CL0_clean"] == pytest.approx(
                cl0_wing, abs=1e-4
            )
            assert problem["data:aerodynamics:wing:cruise:CL_ref"] == pytest.approx(
                cl_ref_wing, abs=1e-4
            )
            assert problem.get_val(
                "data:aerodynamics:wing:cruise:CL_alpha", units="rad**-1"
            ) == pytest.approx(cl_alpha_wing, abs=1e-3)
            assert problem["data:aerodynamics:wing:cruise:CM0_clean"] == pytest.approx(
                cm0, abs=1e-4
            )
            assert problem[
                "data:aerodynamics:wing:cruise:induced_drag_coefficient"
            ] == pytest.approx(coeff_k_wing, abs=1e-4)
            assert problem["data:aerodynamics:horizontal_tail:cruise:CL0"] == pytest.approx(
                cl0_htp, abs=1e-4
            )
            assert problem.get_val(
                "data:aerodynamics:horizontal_tail:cruise:CL_alpha", units="rad**-1"
            ) == pytest.approx(cl_alpha_htp, abs=1e-4)
            assert problem.get_val(
                "data:aerodynamics:horizontal_tail:cruise:CL_alpha_isolated", units="rad**-1"
            ) == pytest.approx(cl_alpha_htp_isolated, abs=1e-4)
            assert problem[
                "data:aerodynamics:horizontal_tail:cruise:induced_drag_coefficient"
            ] == pytest.approx(coeff_k_htp, abs=1e-4)


def comp_high_speed_neuralfoil(
    xml_file_name: str,
    cl0_wing: float,
    cl_ref_wing: float,
    cl_alpha_wing: float,
    cm0: float,
    coeff_k_wing: float,
    cl0_htp: float,
    cl_alpha_htp: float,
    cl_alpha_htp_isolated: float,
    coeff_k_htp: float,
    cl_alpha_vector: np.ndarray,
    mach_vector: np.ndarray,
):
    """Tests components @ high speed!"""
    problem = compute_aero_neuralfoil(xml_file_name, mach_interpolation=True, low_speed_aero=False)

    # Check obtained value(s) is/(are) correct
    assert problem.get_val(
        "data:aerodynamics:aircraft:mach_interpolation:CL_alpha_vector", units="rad**-1"
    ) == pytest.approx(cl_alpha_vector, abs=1e-2)
    assert problem.get_val(
        "data:aerodynamics:aircraft:mach_interpolation:mach_vector"
    ) == pytest.approx(mach_vector, abs=1e-2)
    assert problem.get_val("data:aerodynamics:wing:cruise:CL0_clean") == pytest.approx(
        cl0_wing, abs=1e-4
    )
    assert problem.get_val("data:aerodynamics:wing:cruise:CL_ref") == pytest.approx(
        cl_ref_wing, abs=1e-4
    )
    assert problem.get_val(
        "data:aerodynamics:wing:cruise:CL_alpha", units="rad**-1"
    ) == pytest.approx(cl_alpha_wing, abs=1e-3)
    assert problem.get_val("data:aerodynamics:wing:cruise:CM0_clean") == pytest.approx(
        cm0, abs=1e-4
    )
    assert problem.get_val(
        "data:aerodynamics:wing:cruise:induced_drag_coefficient"
    ) == pytest.approx(coeff_k_wing, abs=1e-4)
    assert problem.get_val("data:aerodynamics:horizontal_tail:cruise:CL0") == pytest.approx(
        cl0_htp, abs=1e-4
    )
    assert problem.get_val(
        "data:aerodynamics:horizontal_tail:cruise:CL_alpha", units="rad**-1"
    ) == pytest.approx(cl_alpha_htp, abs=1e-4)
    assert problem.get_val(
        "data:aerodynamics:horizontal_tail:cruise:CL_alpha_isolated", units="rad**-1"
    ) == pytest.approx(cl_alpha_htp_isolated, abs=1e-4)
    assert problem.get_val(
        "data:aerodynamics:horizontal_tail:cruise:induced_drag_coefficient"
    ) == pytest.approx(coeff_k_htp, abs=1e-4)


def comp_high_speed_input_aoa_xfoil(
    xml_file_name: str,
    *,
    use_openvsp: bool,
):
    """Tests components @ high speed!"""
    for mach_interpolation in [True, False]:
        problem = compute_aero(
            xml_file_name,
            use_openvsp=use_openvsp,
            mach_interpolation=mach_interpolation,
            low_speed_aero=False,
        )
        problem_input_aoa = comp_aero_input_aoa(
            xml_file_name,
            use_openvsp=use_openvsp,
            mach_interpolation=mach_interpolation,
            low_speed_aero=False,
        )
        # Check obtained value(s) is/(are) correct
        if mach_interpolation:
            assert problem[
                "data:aerodynamics:aircraft:mach_interpolation:CL_alpha_vector"
            ] == pytest.approx(
                problem_input_aoa["data:aerodynamics:aircraft:mach_interpolation:CL_alpha_vector"],
                abs=1e-2,
            )
            assert problem[
                "data:aerodynamics:aircraft:mach_interpolation:mach_vector"
            ] == pytest.approx(
                problem_input_aoa["data:aerodynamics:aircraft:mach_interpolation:mach_vector"],
                abs=1e-2,
            )
        else:
            assert problem.get_val(
                "data:aerodynamics:wing:cruise:CL_alpha", units="rad**-1"
            ) == pytest.approx(
                problem_input_aoa.get_val(
                    "data:aerodynamics:wing:cruise:CL_alpha", units="rad**-1"
                ),
                abs=1e-2,
            )
            assert problem.get_val(
                "data:aerodynamics:horizontal_tail:cruise:CL_alpha", units="rad**-1"
            ) == pytest.approx(
                problem_input_aoa.get_val(
                    "data:aerodynamics:horizontal_tail:cruise:CL_alpha", units="rad**-1"
                ),
                abs=1e-2,
            )
            assert problem.get_val(
                "data:aerodynamics:horizontal_tail:cruise:CL_alpha_isolated", units="rad**-1"
            ) == pytest.approx(
                problem_input_aoa.get_val(
                    "data:aerodynamics:horizontal_tail:cruise:CL_alpha_isolated", units="rad**-1"
                ),
                abs=1e-2,
            )


def comp_high_speed_input_aoa_neuralfoil(xml_file_name: str):
    """Tests components @ high speed!"""
    for mach_interpolation in [True, False]:
        problem = compute_aero_neuralfoil(
            xml_file_name, mach_interpolation=mach_interpolation, low_speed_aero=False
        )
        problem_input_aoa = comp_aero_input_aoa_neuralfoil(
            xml_file_name, mach_interpolation=mach_interpolation, low_speed_aero=False
        )
        # Check obtained value(s) is/(are) correct
        if mach_interpolation:
            assert problem[
                "data:aerodynamics:aircraft:mach_interpolation:CL_alpha_vector"
            ] == pytest.approx(
                problem_input_aoa["data:aerodynamics:aircraft:mach_interpolation:CL_alpha_vector"],
                abs=1e-2,
            )
            assert problem[
                "data:aerodynamics:aircraft:mach_interpolation:mach_vector"
            ] == pytest.approx(
                problem_input_aoa["data:aerodynamics:aircraft:mach_interpolation:mach_vector"],
                abs=1e-2,
            )
        else:
            assert problem.get_val(
                "data:aerodynamics:wing:cruise:CL_alpha", units="rad**-1"
            ) == pytest.approx(
                problem_input_aoa.get_val(
                    "data:aerodynamics:wing:cruise:CL_alpha", units="rad**-1"
                ),
                abs=1e-2,
            )
            assert problem.get_val(
                "data:aerodynamics:horizontal_tail:cruise:CL_alpha", units="rad**-1"
            ) == pytest.approx(
                problem_input_aoa.get_val(
                    "data:aerodynamics:horizontal_tail:cruise:CL_alpha", units="rad**-1"
                ),
                abs=1e-2,
            )
            assert problem.get_val(
                "data:aerodynamics:horizontal_tail:cruise:CL_alpha_isolated", units="rad**-1"
            ) == pytest.approx(
                problem_input_aoa.get_val(
                    "data:aerodynamics:horizontal_tail:cruise:CL_alpha_isolated", units="rad**-1"
                ),
                abs=1e-2,
            )


def comp_low_speed_xfoil(
    xml_file_name: str,
    cl0_wing: float,
    cl_ref_wing: float,
    cl_alpha_wing: float,
    cm0: float,
    coeff_k_wing: float,
    cl0_htp: float,
    cl_alpha_htp: float,
    cl_alpha_htp_isolated: float,
    coeff_k_htp: float,
    y_vector_wing: np.ndarray,
    cl_vector_wing: np.ndarray,
    chord_vector_wing: np.ndarray,
    cl_ref_htp: float,
    y_vector_htp: np.ndarray,
    cl_vector_htp: np.ndarray,
    *,
    use_openvsp: bool,
):
    """Tests components @ low speed!"""
    problem = compute_aero(
        xml_file_name, use_openvsp=use_openvsp, mach_interpolation=False, low_speed_aero=True
    )

    # Check obtained value(s) is/(are) correct
    assert problem["data:aerodynamics:wing:low_speed:CL0_clean"] == pytest.approx(
        cl0_wing, abs=1e-4
    )
    assert problem["data:aerodynamics:wing:low_speed:CL_ref"] == pytest.approx(
        cl_ref_wing, abs=1e-4
    )
    assert problem.get_val(
        "data:aerodynamics:wing:low_speed:CL_alpha", units="rad**-1"
    ) == pytest.approx(cl_alpha_wing, abs=1e-3)
    assert problem["data:aerodynamics:wing:low_speed:CM0_clean"] == pytest.approx(cm0, abs=1e-4)
    assert problem["data:aerodynamics:wing:low_speed:induced_drag_coefficient"] == pytest.approx(
        coeff_k_wing, abs=1e-4
    )
    assert problem["data:aerodynamics:horizontal_tail:low_speed:CL0"] == pytest.approx(
        cl0_htp, abs=1e-4
    )
    assert problem.get_val(
        "data:aerodynamics:horizontal_tail:low_speed:CL_alpha", units="rad**-1"
    ) == pytest.approx(cl_alpha_htp, abs=1e-4)
    assert problem.get_val(
        "data:aerodynamics:horizontal_tail:low_speed:CL_alpha_isolated", units="rad**-1"
    ) == pytest.approx(cl_alpha_htp_isolated, abs=1e-4)
    assert problem[
        "data:aerodynamics:horizontal_tail:low_speed:induced_drag_coefficient"
    ] == pytest.approx(coeff_k_htp, abs=1e-4)
    y, cl = reshape_curve(
        problem.get_val("data:aerodynamics:wing:low_speed:Y_vector", "m"),
        problem["data:aerodynamics:wing:low_speed:CL_vector"],
    )
    _, chord = reshape_curve(
        problem.get_val("data:aerodynamics:wing:low_speed:Y_vector", "m"),
        problem.get_val("data:aerodynamics:wing:low_speed:chord_vector", "m"),
    )
    assert np.max(np.abs(y_vector_wing - y)) <= 1e-3
    assert np.max(np.abs(cl_vector_wing - cl)) <= 1e-3
    assert np.max(np.abs(chord_vector_wing - chord)) <= 1e-3
    assert problem["data:aerodynamics:horizontal_tail:low_speed:CL_ref"] == pytest.approx(
        cl_ref_htp, abs=1e-4
    )
    y, cl = reshape_curve(
        problem.get_val("data:aerodynamics:horizontal_tail:low_speed:Y_vector", "m"),
        problem["data:aerodynamics:horizontal_tail:low_speed:CL_vector"],
    )
    assert np.max(np.abs(y_vector_htp - y)) <= 1e-3
    assert np.max(np.abs(cl_vector_htp - cl)) <= 1e-3


def comp_low_speed_neuralfoil(
    xml_file_name: str,
    cl0_wing: float,
    cl_ref_wing: float,
    cl_alpha_wing: float,
    cm0: float,
    coeff_k_wing: float,
    cl0_htp: float,
    cl_alpha_htp: float,
    cl_alpha_htp_isolated: float,
    coeff_k_htp: float,
    y_vector_wing: np.ndarray,
    cl_vector_wing: np.ndarray,
    chord_vector_wing: np.ndarray,
    cl_ref_htp: float,
    y_vector_htp: np.ndarray,
    cl_vector_htp: np.ndarray,
):
    """Tests components @ low speed!"""
    problem = compute_aero_neuralfoil(xml_file_name, mach_interpolation=False, low_speed_aero=True)

    # Check obtained value(s) is/(are) correct
    assert problem.get_val("data:aerodynamics:wing:low_speed:CL0_clean") == pytest.approx(
        cl0_wing, abs=1e-4
    )
    assert problem.get_val("data:aerodynamics:wing:low_speed:CL_ref") == pytest.approx(
        cl_ref_wing, abs=1e-4
    )
    assert problem.get_val(
        "data:aerodynamics:wing:low_speed:CL_alpha", units="rad**-1"
    ) == pytest.approx(cl_alpha_wing, abs=1e-3)
    assert problem.get_val("data:aerodynamics:wing:low_speed:CM0_clean") == pytest.approx(
        cm0, abs=1e-4
    )
    assert problem.get_val(
        "data:aerodynamics:wing:low_speed:induced_drag_coefficient"
    ) == pytest.approx(coeff_k_wing, abs=1e-4)
    assert problem.get_val("data:aerodynamics:horizontal_tail:low_speed:CL0") == pytest.approx(
        cl0_htp, abs=1e-4
    )
    assert problem.get_val(
        "data:aerodynamics:horizontal_tail:low_speed:CL_alpha", units="rad**-1"
    ) == pytest.approx(cl_alpha_htp, abs=1e-4)
    assert problem.get_val(
        "data:aerodynamics:horizontal_tail:low_speed:CL_alpha_isolated", units="rad**-1"
    ) == pytest.approx(cl_alpha_htp_isolated, abs=1e-4)
    assert problem.get_val(
        "data:aerodynamics:horizontal_tail:low_speed:induced_drag_coefficient"
    ) == pytest.approx(coeff_k_htp, abs=1e-4)
    y, cl = reshape_curve(
        problem.get_val("data:aerodynamics:wing:low_speed:Y_vector", "m"),
        problem.get_val("data:aerodynamics:wing:low_speed:CL_vector"),
    )
    _, chord = reshape_curve(
        problem.get_val("data:aerodynamics:wing:low_speed:Y_vector", "m"),
        problem.get_val("data:aerodynamics:wing:low_speed:chord_vector", "m"),
    )
    assert np.max(np.abs(y_vector_wing - y)) <= 1e-3
    assert np.max(np.abs(cl_vector_wing - cl)) <= 1e-3
    assert np.max(np.abs(chord_vector_wing - chord)) <= 1e-3
    assert problem.get_val("data:aerodynamics:horizontal_tail:low_speed:CL_ref") == pytest.approx(
        cl_ref_htp, abs=1e-4
    )
    y, cl = reshape_curve(
        problem.get_val("data:aerodynamics:horizontal_tail:low_speed:Y_vector", "m"),
        problem.get_val("data:aerodynamics:horizontal_tail:low_speed:CL_vector"),
    )
    assert np.max(np.abs(y_vector_htp - y)) <= 1e-3
    assert np.max(np.abs(cl_vector_htp - cl)) <= 1e-3


def comp_low_speed_input_aoa_xfoil(
    xml_file_name: str,
    *,
    use_openvsp: bool,
):
    """Tests components @ low speed!"""
    problem = compute_aero(
        xml_file_name, use_openvsp=use_openvsp, mach_interpolation=False, low_speed_aero=True
    )
    problem_input_aoa = comp_aero_input_aoa(
        xml_file_name, use_openvsp=use_openvsp, mach_interpolation=False, low_speed_aero=True
    )
    # Check obtained value(s) is/(are) correct

    assert problem.get_val(
        "data:aerodynamics:wing:low_speed:CL_alpha", units="rad**-1"
    ) == pytest.approx(
        problem_input_aoa.get_val("data:aerodynamics:wing:low_speed:CL_alpha", units="rad**-1"),
        abs=1e-2,
    )
    assert problem.get_val(
        "data:aerodynamics:horizontal_tail:low_speed:CL_alpha", units="rad**-1"
    ) == pytest.approx(
        problem_input_aoa.get_val(
            "data:aerodynamics:horizontal_tail:low_speed:CL_alpha", units="rad**-1"
        ),
        abs=1e-2,
    )
    assert problem.get_val(
        "data:aerodynamics:horizontal_tail:low_speed:CL_alpha_isolated", units="rad**-1"
    ) == pytest.approx(
        problem_input_aoa.get_val(
            "data:aerodynamics:horizontal_tail:low_speed:CL_alpha_isolated", units="rad**-1"
        ),
        abs=1e-2,
    )


def comp_low_speed_input_aoa_neuralfoil(xml_file_name: str):
    """Tests components @ low speed!"""
    problem = compute_aero_neuralfoil(xml_file_name, mach_interpolation=False, low_speed_aero=True)
    problem_input_aoa = comp_aero_input_aoa_neuralfoil(
        xml_file_name, mach_interpolation=False, low_speed_aero=True
    )
    # Check obtained value(s) is/(are) correct

    assert problem.get_val(
        "data:aerodynamics:wing:low_speed:CL_alpha", units="rad**-1"
    ) == pytest.approx(
        problem_input_aoa.get_val("data:aerodynamics:wing:low_speed:CL_alpha", units="rad**-1"),
        abs=1e-2,
    )
    assert problem.get_val(
        "data:aerodynamics:horizontal_tail:low_speed:CL_alpha", units="rad**-1"
    ) == pytest.approx(
        problem_input_aoa.get_val(
            "data:aerodynamics:horizontal_tail:low_speed:CL_alpha", units="rad**-1"
        ),
        abs=1e-2,
    )
    assert problem.get_val(
        "data:aerodynamics:horizontal_tail:low_speed:CL_alpha_isolated", units="rad**-1"
    ) == pytest.approx(
        problem_input_aoa.get_val(
            "data:aerodynamics:horizontal_tail:low_speed:CL_alpha_isolated", units="rad**-1"
        ),
        abs=1e-2,
    )


def hinge_moment_2d(xml_file_name: str, ch_alpha_2d: float, ch_delta_2d: float):
    """Tests tail hinge-moments"""
    # Research independent input value in .xml file
    problem = setup_and_run_system(Compute2DHingeMomentsTail(), __file__, xml_file_name)
    assert problem.get_val(
        "data:aerodynamics:horizontal_tail:cruise:hinge_moment:CH_alpha_2D", units="rad**-1"
    ) == pytest.approx(ch_alpha_2d, abs=1e-4)
    assert problem.get_val(
        "data:aerodynamics:horizontal_tail:cruise:hinge_moment:CH_delta_2D", units="rad**-1"
    ) == pytest.approx(ch_delta_2d, abs=1e-4)


def hinge_moment_3d(xml_file_name: str, ch_alpha: float, ch_delta: float):
    """Tests tail hinge-moments!"""
    # Research independent input value in .xml file
    problem = setup_and_run_system(Compute3DHingeMomentsTail(), __file__, xml_file_name)
    assert problem.get_val(
        "data:aerodynamics:horizontal_tail:cruise:hinge_moment:CH_alpha", units="rad**-1"
    ) == pytest.approx(ch_alpha, abs=1e-4)
    assert problem.get_val(
        "data:aerodynamics:horizontal_tail:cruise:hinge_moment:CH_delta", units="rad**-1"
    ) == pytest.approx(ch_delta, abs=1e-4)

    problem.check_partials(compact_print=True)


def hinge_moments(xml_file_name: str, ch_alpha: float, ch_delta: float):
    """Tests tail hinge-moments complete computation!"""

    # Research independent input value in .xml file
    problem = setup_and_run_system(ComputeHingeMomentsTail(), __file__, xml_file_name)
    assert problem.get_val(
        "data:aerodynamics:horizontal_tail:cruise:hinge_moment:CH_alpha", units="rad**-1"
    ) == pytest.approx(ch_alpha, abs=1e-4)
    assert problem.get_val(
        "data:aerodynamics:horizontal_tail:cruise:hinge_moment:CH_delta", units="rad**-1"
    ) == pytest.approx(ch_delta, abs=1e-4)


def elevator(
    xml_file_name: str,
    cl_delta_elev: float,
    cd_delta_elev: float,
):
    problem = setup_and_run_system(ComputeDeltaElevator(), __file__, xml_file_name)

    assert problem.get_val(
        "data:aerodynamics:elevator:low_speed:CL_delta", units="rad**-1"
    ) == pytest.approx(cl_delta_elev, abs=1e-4)
    assert problem.get_val(
        "data:aerodynamics:elevator:low_speed:CD_delta", units="rad**-2"
    ) == pytest.approx(cd_delta_elev, abs=1e-4)


def high_lift(
    xml_file_name: str,
    delta_cl0_landing: float,
    delta_cl0_landing_2d: float,
    delta_clmax_landing: float,
    delta_cm_landing: float,
    delta_cm_landing_2d: float,
    delta_cd_landing: float,
    delta_cd_landing_2d: float,
    delta_cl0_takeoff: float,
    delta_cl0_takeoff_2d: float,
    delta_clmax_takeoff: float,
    delta_cm_takeoff: float,
    delta_cm_takeoff_2d: float,
    delta_cd_takeoff: float,
    delta_cd_takeoff_2d: float,
):
    """Tests high-lift contribution!"""
    # Research independent input value in .xml file
    problem = setup_and_run_system(ComputeDeltaHighLift(), __file__, xml_file_name)
    assert problem["data:aerodynamics:flaps:landing:CL"] == pytest.approx(
        delta_cl0_landing, abs=1e-4
    )
    assert problem["data:aerodynamics:flaps:landing:CL_2D"] == pytest.approx(
        delta_cl0_landing_2d, abs=1e-4
    )
    assert problem["data:aerodynamics:flaps:landing:CL_max"] == pytest.approx(
        delta_clmax_landing, abs=1e-4
    )
    assert problem["data:aerodynamics:flaps:landing:CM"] == pytest.approx(
        delta_cm_landing, abs=1e-4
    )
    assert problem["data:aerodynamics:flaps:landing:CM_2D"] == pytest.approx(
        delta_cm_landing_2d, abs=1e-4
    )
    assert problem["data:aerodynamics:flaps:landing:CD"] == pytest.approx(
        delta_cd_landing, abs=1e-4
    )
    assert problem["data:aerodynamics:flaps:landing:CD_2D"] == pytest.approx(
        delta_cd_landing_2d, abs=1e-4
    )
    assert problem["data:aerodynamics:flaps:takeoff:CL"] == pytest.approx(
        delta_cl0_takeoff, abs=1e-4
    )
    assert problem["data:aerodynamics:flaps:takeoff:CL_2D"] == pytest.approx(
        delta_cl0_takeoff_2d, abs=1e-4
    )
    assert problem["data:aerodynamics:flaps:takeoff:CL_max"] == pytest.approx(
        delta_clmax_takeoff, abs=1e-4
    )
    assert problem["data:aerodynamics:flaps:takeoff:CM"] == pytest.approx(
        delta_cm_takeoff, abs=1e-4
    )
    assert problem["data:aerodynamics:flaps:takeoff:CM_2D"] == pytest.approx(
        delta_cm_takeoff_2d, abs=1e-4
    )
    assert problem["data:aerodynamics:flaps:takeoff:CD"] == pytest.approx(
        delta_cd_takeoff, abs=1e-4
    )
    assert problem["data:aerodynamics:flaps:takeoff:CD_2D"] == pytest.approx(
        delta_cd_takeoff_2d, abs=1e-4
    )


def wing_extreme_cl_clean_xfoil(
    xml_file_name: str, cl_max_clean_wing: float, cl_min_clean_wing: float
):
    """Tests maximum minimum lift coefficient for clean wing."""

    # Transfer saved polar results to temporary folder
    tmp_folder = polar_result_transfer()

    # Research independent input value in .xml file for Openvsp test
    ivc = get_indep_var_comp(list_inputs(ComputeExtremeCLWing()), __file__, xml_file_name)

    # Run problem
    problem = run_system(ComputeExtremeCLWing(), ivc)

    # Retrieve polar results from temporary folder
    polar_result_retrieve(tmp_folder)

    assert problem["data:aerodynamics:wing:low_speed:CL_max_clean"] == pytest.approx(
        cl_max_clean_wing, abs=1e-2
    )
    assert problem["data:aerodynamics:wing:low_speed:CL_min_clean"] == pytest.approx(
        cl_min_clean_wing, abs=1e-2
    )


def wing_extreme_cl_clean_neuralfoil(
    xml_file_name: str, cl_max_clean_wing: float, cl_min_clean_wing: float
):
    """Tests maximum minimum lift coefficient for clean wing."""

    # Transfer saved polar results to temporary folder
    tmp_folder = polar_result_transfer()

    # Research independent input value in .xml file for Openvsp test
    ivc = get_indep_var_comp(
        list_inputs(ComputeExtremeCLWing(use_neuralfoil=True)), __file__, xml_file_name
    )

    # Run problem
    problem = run_system(ComputeExtremeCLWing(use_neuralfoil=True), ivc)

    # Retrieve polar results from temporary folder
    polar_result_retrieve(tmp_folder)

    assert problem["data:aerodynamics:wing:low_speed:CL_max_clean"] == pytest.approx(
        cl_max_clean_wing, abs=1e-2
    )
    assert problem["data:aerodynamics:wing:low_speed:CL_min_clean"] == pytest.approx(
        cl_min_clean_wing, abs=1e-2
    )


def htp_extreme_cl_clean_xfoil(
    xml_file_name: str,
    cl_max_clean_htp: float,
    cl_min_clean_htp: float,
    alpha_max_clean_htp: float,
    alpha_min_clean_htp: float,
):
    """Tests maximum minimum lift coefficient for clean htp."""

    # Transfer saved polar results to temporary folder
    tmp_folder = polar_result_transfer()

    # Research independent input value in .xml file for Openvsp test
    ivc = get_indep_var_comp(list_inputs(ComputeExtremeCLHtp()), __file__, xml_file_name)

    # Run problem
    problem = run_system(ComputeExtremeCLHtp(), ivc)

    # Retrieve polar results from temporary folder
    polar_result_retrieve(tmp_folder)

    assert problem["data:aerodynamics:horizontal_tail:low_speed:CL_max_clean"] == pytest.approx(
        cl_max_clean_htp, abs=1e-2
    )
    assert problem["data:aerodynamics:horizontal_tail:low_speed:CL_min_clean"] == pytest.approx(
        cl_min_clean_htp, abs=1e-2
    )
    assert problem.get_val(
        "data:aerodynamics:horizontal_tail:low_speed:clean:alpha_aircraft_max", units="deg"
    ) == pytest.approx(alpha_max_clean_htp, abs=1e-2)
    assert problem.get_val(
        "data:aerodynamics:horizontal_tail:low_speed:clean:alpha_aircraft_min", units="deg"
    ) == pytest.approx(alpha_min_clean_htp, abs=1e-2)


def htp_extreme_cl_clean_neuralfoil(
    xml_file_name: str,
    cl_max_clean_htp: float,
    cl_min_clean_htp: float,
    alpha_max_clean_htp: float,
    alpha_min_clean_htp: float,
):
    """Tests maximum minimum lift coefficient for clean htp."""

    # Transfer saved polar results to temporary folder
    tmp_folder = polar_result_transfer()

    # Research independent input value in .xml file for Openvsp test
    ivc = get_indep_var_comp(
        list_inputs(ComputeExtremeCLHtp(use_neuralfoil=True)), __file__, xml_file_name
    )

    # Run problem
    problem = run_system(ComputeExtremeCLHtp(use_neuralfoil=True), ivc)

    # Retrieve polar results from temporary folder
    polar_result_retrieve(tmp_folder)

    assert problem["data:aerodynamics:horizontal_tail:low_speed:CL_max_clean"] == pytest.approx(
        cl_max_clean_htp, abs=1e-2
    )
    assert problem["data:aerodynamics:horizontal_tail:low_speed:CL_min_clean"] == pytest.approx(
        cl_min_clean_htp, abs=1e-2
    )
    assert problem.get_val(
        "data:aerodynamics:horizontal_tail:low_speed:clean:alpha_aircraft_max", units="deg"
    ) == pytest.approx(alpha_max_clean_htp, abs=1e-2)
    assert problem.get_val(
        "data:aerodynamics:horizontal_tail:low_speed:clean:alpha_aircraft_min", units="deg"
    ) == pytest.approx(alpha_min_clean_htp, abs=1e-2)


def extreme_cl(
    xml_file_name: str,
    cl_max_takeoff_wing: float,
    cl_max_landing_wing: float,
):
    """Tests maximum/minimum cl component with default result cl=f(y) curve!"""
    # Transfer saved polar results to temporary folder
    tmp_folder = polar_result_transfer()

    # Research independent input value in .xml file for Openvsp test
    ivc = get_indep_var_comp(list_inputs(ComputeAircraftMaxCl()), __file__, xml_file_name)

    # Run problem
    problem = run_system(ComputeAircraftMaxCl(), ivc)

    # Retrieve polar results from temporary folder
    polar_result_retrieve(tmp_folder)

    # Check obtained value(s) is/(are) correct
    assert problem["data:aerodynamics:aircraft:takeoff:CL_max"] == pytest.approx(
        cl_max_takeoff_wing, abs=1e-2
    )
    assert problem["data:aerodynamics:aircraft:landing:CL_max"] == pytest.approx(
        cl_max_landing_wing, abs=1e-2
    )

    problem.check_partials(compact_print=True)


def l_d_max(
    xml_file_name: str, l_d_max_: float, optimal_cl: float, optimal_cd: float, optimal_alpha: float
):
    """Tests best lift/drag component!"""
    # Define independent input value (openVSP)
    problem = setup_and_run_system(ComputeLDMax(), __file__, xml_file_name)
    assert problem["data:aerodynamics:aircraft:cruise:L_D_max"] == pytest.approx(l_d_max_, abs=1e-1)
    assert problem["data:aerodynamics:aircraft:cruise:optimal_CL"] == pytest.approx(
        optimal_cl, abs=1e-4
    )
    assert problem["data:aerodynamics:aircraft:cruise:optimal_CD"] == pytest.approx(
        optimal_cd, abs=1e-4
    )
    assert problem.get_val(
        "data:aerodynamics:aircraft:cruise:optimal_alpha", units="deg"
    ) == pytest.approx(optimal_alpha, abs=1e-2)

    problem.check_partials(compact_print=True)


def cnbeta(xml_file_name: str, cn_beta_fus: float):
    """Tests cn beta fuselage"""
    # Research independent input value in .xml file
    problem = setup_and_run_system(ComputeCnBetaFuselage(), __file__, xml_file_name)
    assert problem["data:aerodynamics:fuselage:Cn_beta"] == pytest.approx(cn_beta_fus, rel=1e-3)

    problem.check_partials(compact_print=True)


def slipstream_openvsp(
    xml_file_name: str,
    engine_wrapper_id: str,
    *,
    low_speed_aero: bool,
):
    # Create result temporary directory
    results_folder = _create_tmp_directory()

    # Research independent input value in .xml file
    # noinspection PyTypeChecker
    ivc = get_indep_var_comp(
        list_inputs(
            ComputeSlipstreamOpenvsp(
                low_speed_aero=low_speed_aero,
                propulsion_id=engine_wrapper_id,
                result_folder_path=results_folder.name,
            )
        ),
        __file__,
        xml_file_name,
    )
    # Run problem and check obtained value(s) is/(are) correct and return for complementary values
    # check
    # noinspection PyTypeChecker
    return run_system(
        ComputeSlipstreamOpenvsp(
            low_speed_aero=low_speed_aero,
            propulsion_id=engine_wrapper_id,
            result_folder_path=results_folder.name,
        ),
        ivc,
    )


def slipstream_openvsp_cruise(
    xml_file_name: str,
    engine_wrapper_id: str,
    y_vector_prop_on: np.ndarray,
    cl_vector_prop_on: np.ndarray,
    ct: float,
    delta_cl: float,
):
    # Compute slipstream @ high speed
    problem = slipstream_openvsp(xml_file_name, engine_wrapper_id, low_speed_aero=False)

    # Check obtained value(s) is/(are) correct
    y_result_prop_on = problem.get_val(
        "data:aerodynamics:slipstream:wing:cruise:prop_on:Y_vector", units="m"
    )
    assert y_result_prop_on == pytest.approx(y_vector_prop_on, abs=1e-2)
    cl_result_prop_on = problem.get_val(
        "data:aerodynamics:slipstream:wing:cruise:prop_on:CL_vector"
    )
    assert cl_vector_prop_on == pytest.approx(cl_result_prop_on, abs=1e-2)
    assert problem.get_val(
        "data:aerodynamics:slipstream:wing:cruise:prop_on:CT_ref"
    ) == pytest.approx(ct, abs=1e-4)
    delta_cl_result = problem.get_val(
        "data:aerodynamics:slipstream:wing:cruise:prop_on:CL"
    ) - problem.get_val("data:aerodynamics:slipstream:wing:cruise:prop_off:CL")
    assert delta_cl_result == pytest.approx(delta_cl, abs=1e-4)


def slipstream_openvsp_low_speed(
    xml_file_name: str,
    engine_wrapper_id: str,
    y_vector_prop_on: np.ndarray,
    cl_vector_prop_on: np.ndarray,
    ct: float,
    delta_cl: float,
):
    # Compute slipstream @ high speed
    problem = slipstream_openvsp(xml_file_name, engine_wrapper_id, low_speed_aero=True)

    # Check obtained value(s) is/(are) correct
    y_result_prop_on = problem.get_val(
        "data:aerodynamics:slipstream:wing:low_speed:prop_on:Y_vector", units="m"
    )
    assert np.max(np.abs(y_vector_prop_on - y_result_prop_on)) <= 1e-2
    cl_result_prop_on = problem.get_val(
        "data:aerodynamics:slipstream:wing:low_speed:prop_on:CL_vector"
    )
    assert np.max(np.abs(cl_vector_prop_on - cl_result_prop_on)) <= 1e-2
    assert problem.get_val(
        "data:aerodynamics:slipstream:wing:low_speed:prop_on:CT_ref"
    ) == pytest.approx(ct, abs=1e-4)
    delta_cl_result = problem.get_val(
        "data:aerodynamics:slipstream:wing:low_speed:prop_on:CL"
    ) - problem.get_val("data:aerodynamics:slipstream:wing:low_speed:prop_off:CL")
    assert delta_cl_result == pytest.approx(delta_cl, abs=1e-4)


def compute_mach_interpolation_roskam_xfoil(
    xml_file_name: str, cl_alpha_vector: np.ndarray, mach_vector: np.ndarray
):
    """Tests computation of the mach interpolation vector using Roskam's approach!"""
    # Research independent input value in .xml file
    problem = setup_and_run_system(ComputeMachInterpolation(), __file__, xml_file_name)
    cl_alpha_result = problem["data:aerodynamics:aircraft:mach_interpolation:CL_alpha_vector"]
    assert np.max(np.abs(cl_alpha_vector - cl_alpha_result)) <= 1e-2
    mach_result = problem["data:aerodynamics:aircraft:mach_interpolation:mach_vector"]
    assert np.max(np.abs(mach_vector - mach_result)) <= 1e-2


def compute_mach_interpolation_roskam_neuralfoil(
    xml_file_name: str, cl_alpha_vector: np.ndarray, mach_vector: np.ndarray
):
    """Tests computation of the mach interpolation vector using Roskam's approach!"""
    # Research independent input value in .xml file
    problem = setup_and_run_system(
        ComputeMachInterpolation(use_neuralfoil=True), __file__, xml_file_name
    )
    cl_alpha_result = problem["data:aerodynamics:aircraft:mach_interpolation:CL_alpha_vector"]
    assert np.max(np.abs(cl_alpha_vector - cl_alpha_result)) <= 1e-2
    mach_result = problem["data:aerodynamics:aircraft:mach_interpolation:mach_vector"]
    assert np.max(np.abs(mach_vector - mach_result)) <= 1e-2


def cl_alpha_vt(
    xml_file_name: str, cl_alpha_vt_ls: float, k_ar_effective: float, cl_alpha_vt_cruise: float
):
    """Tests Cl alpha vt!"""
    # Research independent input value in .xml file
    problem = setup_and_run_system(
        ComputeClAlphaVerticalTail(low_speed_aero=True), __file__, xml_file_name
    )
    assert problem.get_val(
        "data:aerodynamics:vertical_tail:low_speed:CL_alpha", units="rad**-1"
    ) == pytest.approx(cl_alpha_vt_ls, rel=1e-3)
    assert problem.get_val("data:aerodynamics:vertical_tail:k_ar_effective") == pytest.approx(
        k_ar_effective, rel=1e-3
    )

    # Research independent input value in .xml file
    problem = setup_and_run_system(ComputeClAlphaVerticalTail(), __file__, xml_file_name)
    assert problem.get_val(
        "data:aerodynamics:vertical_tail:cruise:CL_alpha", units="rad**-1"
    ) == pytest.approx(cl_alpha_vt_cruise, rel=1e-3)


def cy_delta_r(xml_file_name: str, cy_delta_r_: float, cy_delta_r_cruise):
    """Tests cy delta of the rudder!"""
    # Research independent input value in .xml file
    problem = setup_and_run_system(
        ComputeCyDeltaRudder(low_speed_aero=True), __file__, xml_file_name
    )
    assert problem.get_val(
        "data:aerodynamics:rudder:low_speed:Cy_delta_r", units="rad**-1"
    ) == pytest.approx(cy_delta_r_, abs=1e-4)

    # Research independent input value in .xml file
    problem = setup_and_run_system(ComputeCyDeltaRudder(), __file__, xml_file_name)
    assert problem.get_val(
        "data:aerodynamics:rudder:cruise:Cy_delta_r", units="rad**-1"
    ) == pytest.approx(cy_delta_r_cruise, abs=1e-4)


def effective_efficiency(
    xml_file_name: str, effective_efficiency_low_speed: float, effective_efficiency_cruise: float
):
    """Tests effective efficiency of the propeller!"""
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(ComputeEffectiveEfficiencyPropeller(low_speed_aero=True)),
        __file__,
        xml_file_name,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeEffectiveEfficiencyPropeller(low_speed_aero=True), ivc)
    assert problem.get_val(
        "data:aerodynamics:propeller:installation_effect:effective_efficiency:low_speed",
    ) == pytest.approx(effective_efficiency_low_speed, abs=1e-4)

    # Research independent input value in .xml file
    problem = setup_and_run_system(ComputeEffectiveEfficiencyPropeller(), __file__, xml_file_name)
    assert problem.get_val(
        "data:aerodynamics:propeller:installation_effect:effective_efficiency:cruise",
    ) == pytest.approx(effective_efficiency_cruise, abs=1e-4)


def cm_alpha_fus(xml_file_name: str, cm_alpha_fus_: float):
    """Tests cm alpha of the fuselage"""
    # Research independent input value in .xml file
    problem = setup_and_run_system(ComputeCmAlphaFuselage(), __file__, xml_file_name)
    assert problem.get_val("data:aerodynamics:fuselage:cm_alpha", units="rad**-1") == pytest.approx(
        cm_alpha_fus_, abs=1e-4
    )

    problem.check_partials(compact_print=True)


def high_speed_connection(xml_file_name: str, engine_wrapper_id: str, *, use_openvsp: bool):
    """Tests high speed components connection!"""
    # Transfer saved polar results to temporary folder
    tmp_folder = polar_result_transfer()

    # load all inputs
    ivc = get_indep_var_comp(
        list_inputs(
            AerodynamicsHighSpeed(propulsion_id=engine_wrapper_id, use_openvsp=use_openvsp)
        ),
        __file__,
        xml_file_name,
    )

    # noinspection PyTypeChecker
    run_system(AerodynamicsHighSpeed(propulsion_id=engine_wrapper_id, use_openvsp=use_openvsp), ivc)

    # Retrieve polar results from temporary folder
    polar_result_retrieve(tmp_folder)


def low_speed_connection(xml_file_name: str, engine_wrapper_id: str, *, use_openvsp: bool):
    """Tests low speed components connection!"""
    # Transfer saved polar results to temporary folder
    tmp_folder = polar_result_transfer()

    # load all inputs
    ivc = get_indep_var_comp(
        list_inputs(AerodynamicsLowSpeed(propulsion_id=engine_wrapper_id, use_openvsp=use_openvsp)),
        __file__,
        xml_file_name,
    )

    # noinspection PyTypeChecker
    run_system(AerodynamicsLowSpeed(propulsion_id=engine_wrapper_id, use_openvsp=use_openvsp), ivc)

    # Retrieve polar results from temporary folder
    polar_result_retrieve(tmp_folder)


def v_n_diagram(
    xml_file_name: str,
    engine_wrapper_id: str,
    velocity_vect: np.ndarray,
    load_factor_vect: np.ndarray,
):
    # load all inputs
    ivc = get_indep_var_comp(
        list_inputs(ComputeVNAndVH(propulsion_id=engine_wrapper_id)), __file__, xml_file_name
    )
    # Run problem with VLM and check obtained value(s) is/(are) correct
    # noinspection PyTypeChecker
    problem = run_system(ComputeVNAndVH(propulsion_id=engine_wrapper_id), ivc)
    assert velocity_vect == pytest.approx(
        problem.get_val("data:mission:sizing:cs23:flight_domain:mtow:velocity", units="m/s"),
        abs=1e-3,
    )
    assert load_factor_vect == pytest.approx(
        problem["data:mission:sizing:cs23:flight_domain:mtow:load_factor"], abs=1e-3
    )


def load_factor(
    xml_file_name: str,
    engine_wrapper_id: str,
    load_factor_ultimate: float,
    load_factor_ultimate_mtow: float,
    load_factor_ultimate_mzfw: float,
    vh: float,
    va: float,
    vc: float,
    vd: float,
):
    # load all inputs
    ivc = get_indep_var_comp(
        list_inputs(LoadFactor(propulsion_id=engine_wrapper_id)), __file__, xml_file_name
    )

    problem = run_system(LoadFactor(propulsion_id=engine_wrapper_id), ivc)

    assert problem.get_val(
        "data:mission:sizing:cs23:sizing_factor:ultimate_aircraft"
    ) == pytest.approx(load_factor_ultimate, abs=1e-1)
    assert max(
        problem.get_val("data:mission:sizing:cs23:sizing_factor:ultimate_mtow:positive"),
        problem.get_val("data:mission:sizing:cs23:sizing_factor:ultimate_mtow:negative"),
    ) == pytest.approx(load_factor_ultimate_mtow, abs=1e-1)
    assert max(
        problem.get_val("data:mission:sizing:cs23:sizing_factor:ultimate_mzfw:positive"),
        problem.get_val("data:mission:sizing:cs23:sizing_factor:ultimate_mzfw:negative"),
    ) == pytest.approx(load_factor_ultimate_mzfw, abs=1e-1)
    assert problem.get_val("data:TLAR:v_max_sl", units="m/s") == pytest.approx(vh, abs=1e-2)
    assert problem.get_val(
        "data:mission:sizing:cs23:characteristic_speed:va", units="m/s"
    ) == pytest.approx(va, abs=1e-2)
    assert problem.get_val(
        "data:mission:sizing:cs23:characteristic_speed:vc", units="m/s"
    ) == pytest.approx(vc, abs=1e-2)
    assert problem.get_val(
        "data:mission:sizing:cs23:characteristic_speed:vd", units="m/s"
    ) == pytest.approx(vd, abs=1e-2)


def propeller_xfoil(
    xml_file_name: str,
    thrust_sl: np.ndarray,
    thrust_sl_limit: np.ndarray,
    efficiency_sl: np.ndarray,
    thrust_cl: np.ndarray,
    thrust_cl_limit: np.ndarray,
    efficiency_cl: np.ndarray,
    speed: np.ndarray,
):
    # Transfer saved polar results to temporary folder
    tmp_folder = polar_result_transfer()

    # load all inputs and add missing ones
    ivc = get_indep_var_comp(
        list_inputs(
            ComputePropellerPerformance(
                sections_profile_name_list=["naca4430"],
                sections_profile_position_list=[0],
                elements_number=3,
            )
        ),
        __file__,
        xml_file_name,
    )

    # Run problem
    problem = run_system(
        ComputePropellerPerformance(
            sections_profile_name_list=["naca4430"],
            sections_profile_position_list=[0],
            elements_number=3,
        ),
        ivc,
    )

    # Retrieve polar results from temporary folder
    polar_result_retrieve(tmp_folder)

    # Check obtained value(s) is/(are) correct
    assert problem.get_val(
        "data:aerodynamics:propeller:sea_level:thrust", units="N"
    ) == pytest.approx(thrust_sl, abs=1)
    assert problem.get_val(
        "data:aerodynamics:propeller:sea_level:thrust_limit", units="N"
    ) == pytest.approx(thrust_sl_limit, abs=1)
    assert problem.get_val(
        "data:aerodynamics:propeller:sea_level:speed", units="m/s"
    ) == pytest.approx(speed, abs=1e-2)
    assert problem.get_val("data:aerodynamics:propeller:sea_level:efficiency") == pytest.approx(
        efficiency_sl, abs=1e-5
    )

    assert problem.get_val(
        "data:aerodynamics:propeller:cruise_level:thrust", units="N"
    ) == pytest.approx(thrust_cl, abs=1)
    assert problem.get_val(
        "data:aerodynamics:propeller:cruise_level:thrust_limit", units="N"
    ) == pytest.approx(thrust_cl_limit, abs=1)
    assert problem.get_val(
        "data:aerodynamics:propeller:cruise_level:speed", units="m/s"
    ) == pytest.approx(speed, abs=1e-2)
    assert problem.get_val("data:aerodynamics:propeller:cruise_level:efficiency") == pytest.approx(
        efficiency_cl, abs=1e-5
    )


def propeller_neuralfoil(
    xml_file_name: str,
    thrust_sl: np.ndarray,
    thrust_sl_limit: np.ndarray,
    efficiency_sl: np.ndarray,
    thrust_cl: np.ndarray,
    thrust_cl_limit: np.ndarray,
    efficiency_cl: np.ndarray,
    speed: np.ndarray,
):
    # Transfer saved polar results to temporary folder
    tmp_folder = polar_result_transfer()

    # load all inputs and add missing ones
    ivc = get_indep_var_comp(
        list_inputs(
            ComputePropellerPerformance(
                sections_profile_name_list=["naca4430"],
                sections_profile_position_list=[0],
                elements_number=3,
                use_neuralfoil=True,
            )
        ),
        __file__,
        xml_file_name,
    )

    # Run problem
    problem = run_system(
        ComputePropellerPerformance(
            sections_profile_name_list=["naca4430"],
            sections_profile_position_list=[0],
            elements_number=3,
            use_neuralfoil=True,
        ),
        ivc,
    )

    # Retrieve polar results from temporary folder
    polar_result_retrieve(tmp_folder)

    # Check obtained value(s) is/(are) correct
    assert problem.get_val(
        "data:aerodynamics:propeller:sea_level:thrust", units="N"
    ) == pytest.approx(thrust_sl, abs=1)
    assert problem.get_val(
        "data:aerodynamics:propeller:sea_level:thrust_limit", units="N"
    ) == pytest.approx(thrust_sl_limit, abs=1)
    assert problem.get_val(
        "data:aerodynamics:propeller:sea_level:speed", units="m/s"
    ) == pytest.approx(speed, abs=1e-2)
    assert problem.get_val("data:aerodynamics:propeller:sea_level:efficiency") == pytest.approx(
        efficiency_sl, abs=1e-5
    )

    assert problem.get_val(
        "data:aerodynamics:propeller:cruise_level:thrust", units="N"
    ) == pytest.approx(thrust_cl, abs=1)
    assert problem.get_val(
        "data:aerodynamics:propeller:cruise_level:thrust_limit", units="N"
    ) == pytest.approx(thrust_cl_limit, abs=1)
    assert problem.get_val(
        "data:aerodynamics:propeller:cruise_level:speed", units="m/s"
    ) == pytest.approx(speed, abs=1e-2)
    assert problem.get_val("data:aerodynamics:propeller:cruise_level:efficiency") == pytest.approx(
        efficiency_cl, abs=1e-5
    )


def non_equilibrated_cl_cd_polar(
    xml_file_name: str,
    cl_polar_ls_: np.ndarray,
    cd_polar_ls_: np.ndarray,
    cl_polar_cruise_: np.ndarray,
    cd_polar_cruise_: np.ndarray,
):
    """Tests non-equilibrated cl/cd polar of the aircraft"""
    # Research independent input value in .xml file
    problem = setup_and_run_system(
        ComputeNonEquilibratedPolar(low_speed_aero=True), __file__, xml_file_name
    )
    assert problem.get_val("data:aerodynamics:aircraft:low_speed:CD")[::10] == pytest.approx(
        cd_polar_ls_, abs=1e-4
    )
    assert problem.get_val("data:aerodynamics:aircraft:low_speed:CL")[::10] == pytest.approx(
        cl_polar_ls_, abs=1e-2
    )

    problem = setup_and_run_system(
        ComputeNonEquilibratedPolar(low_speed_aero=False), __file__, xml_file_name
    )
    assert problem.get_val("data:aerodynamics:aircraft:cruise:CD")[::10] == pytest.approx(
        cd_polar_cruise_, abs=1e-4
    )
    assert problem.get_val("data:aerodynamics:aircraft:cruise:CL")[::10] == pytest.approx(
        cl_polar_cruise_, abs=1e-2
    )


def equilibrated_cl_cd_polar(
    xml_file_name: str,
    cl_polar_ls_: np.ndarray,
    cd_polar_ls_: np.ndarray,
    cl_polar_cruise_: np.ndarray,
    cd_polar_cruise_: np.ndarray,
):
    """Tests equilibrated cl/cd polar of the aircraft"""
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(ComputeEquilibratedPolar(low_speed_aero=True, cg_ratio=0.5)),
        __file__,
        xml_file_name,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeEquilibratedPolar(low_speed_aero=True, cg_ratio=0.5), ivc)
    polar_cd = np.array(problem.get_val("data:aerodynamics:aircraft:low_speed:equilibrated:CD"))
    valid_polar_cd = polar_cd[np.where(polar_cd < FIRST_INVALID_COEFF)[0]]
    assert list(valid_polar_cd)[::10] == pytest.approx(cd_polar_ls_, abs=1e-4)
    polar_cl = np.array(problem.get_val("data:aerodynamics:aircraft:low_speed:equilibrated:CL"))
    valid_polar_cl = polar_cl[np.where(polar_cl < FIRST_INVALID_COEFF)[0]]
    assert list(valid_polar_cl)[::10] == pytest.approx(cl_polar_ls_, abs=1e-2)

    ivc = get_indep_var_comp(
        list_inputs(ComputeEquilibratedPolar(low_speed_aero=False, cg_ratio=0.5)),
        __file__,
        xml_file_name,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeEquilibratedPolar(low_speed_aero=False, cg_ratio=0.5), ivc)
    polar_cd = np.array(problem.get_val("data:aerodynamics:aircraft:cruise:equilibrated:CD"))
    valid_polar_cd = polar_cd[np.where(polar_cd < FIRST_INVALID_COEFF)[0]]
    assert list(valid_polar_cd)[::10] == pytest.approx(cd_polar_cruise_, abs=1e-4)
    polar_cl = np.array(problem.get_val("data:aerodynamics:aircraft:cruise:equilibrated:CL"))
    valid_polar_cl = polar_cl[np.where(polar_cl < FIRST_INVALID_COEFF)[0]]
    assert list(valid_polar_cl)[::10] == pytest.approx(cl_polar_cruise_, abs=1e-2)


def cy_beta_fus(
    xml_file_name: str,
    cy_beta_fus_: float,
):
    """Tests cy beta of the fuselage"""
    # Research independent input value in .xml file
    problem = setup_and_run_system(ComputeCyBetaFuselage(), __file__, xml_file_name)
    assert problem.get_val("data:aerodynamics:fuselage:Cy_beta", units="rad**-1") == pytest.approx(
        cy_beta_fus_, rel=1e-3
    )

    problem.check_partials(compact_print=True)


def downwash_gradient(
    xml_file_name: str,
    downwash_gradient_ls_: float,
    downwash_gradient_cruise_: float,
):
    """Tests cy beta of the fuselage"""
    # Research independent input value in .xml file
    problem = setup_and_run_system(
        DownWashGradientComputation(low_speed_aero=True), __file__, xml_file_name
    )
    assert problem.get_val(
        "data:aerodynamics:horizontal_tail:low_speed:downwash_gradient"
    ) == pytest.approx(downwash_gradient_ls_, rel=1e-3)

    problem.check_partials(compact_print=True)

    """Tests cy beta of the fuselage"""
    # Research independent input value in .xml file
    problem = setup_and_run_system(
        DownWashGradientComputation(low_speed_aero=False), __file__, xml_file_name
    )
    assert problem.get_val(
        "data:aerodynamics:horizontal_tail:cruise:downwash_gradient"
    ) == pytest.approx(downwash_gradient_cruise_, rel=1e-3)

    problem.check_partials(compact_print=True)


def lift_aoa_rate_derivative(
    xml_file_name: str,
    cl_aoa_dot_low_speed_: float,
    cl_aoa_dot_cruise_: float,
):
    # Research independent input value in .xml file
    problem = setup_and_run_system(
        ComputeCLAlphaDotAircraft(low_speed_aero=True), __file__, xml_file_name
    )
    assert problem.get_val("data:aerodynamics:aircraft:low_speed:CL_alpha_dot") == pytest.approx(
        cl_aoa_dot_low_speed_, rel=1e-3
    )

    problem.check_partials(compact_print=True)

    # Research independent input value in .xml file
    problem = setup_and_run_system(
        ComputeCLAlphaDotAircraft(low_speed_aero=False), __file__, xml_file_name
    )
    assert problem.get_val("data:aerodynamics:aircraft:cruise:CL_alpha_dot") == pytest.approx(
        cl_aoa_dot_cruise_, rel=1e-3
    )

    problem.check_partials(compact_print=True)


def lift_pitch_velocity_derivative_ht(
    xml_file_name: str,
    cl_q_ht_low_speed_: float,
    cl_q_ht_cruise_: float,
):
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(ComputeCLPitchVelocityHorizontalTail(low_speed_aero=True)),
        __file__,
        xml_file_name,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeCLPitchVelocityHorizontalTail(low_speed_aero=True), ivc)
    assert problem.get_val("data:aerodynamics:horizontal_tail:low_speed:CL_q") == pytest.approx(
        cl_q_ht_low_speed_, rel=1e-3
    )

    problem.check_partials(compact_print=True)

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(ComputeCLPitchVelocityHorizontalTail(low_speed_aero=False)),
        __file__,
        xml_file_name,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeCLPitchVelocityHorizontalTail(low_speed_aero=False), ivc)
    assert problem.get_val("data:aerodynamics:horizontal_tail:cruise:CL_q") == pytest.approx(
        cl_q_ht_cruise_, rel=1e-3
    )

    problem.check_partials(compact_print=True)


def lift_pitch_velocity_derivative_wing(
    xml_file_name: str,
    cl_q_wing_low_speed_: float,
    cl_q_wing_cruise_: float,
):
    # Research independent input value in .xml file
    problem = setup_and_run_system(
        ComputeCLPitchVelocityWing(low_speed_aero=True), __file__, xml_file_name
    )
    assert problem.get_val("data:aerodynamics:wing:low_speed:CL_q") == pytest.approx(
        cl_q_wing_low_speed_, rel=1e-3
    )

    problem.check_partials(compact_print=True)

    # Research independent input value in .xml file
    problem = setup_and_run_system(
        ComputeCLPitchVelocityWing(low_speed_aero=False), __file__, xml_file_name
    )
    assert problem.get_val("data:aerodynamics:wing:cruise:CL_q") == pytest.approx(
        cl_q_wing_cruise_, rel=1e-3
    )

    problem.check_partials(compact_print=True)


def lift_pitch_velocity_derivative_aircraft(
    xml_file_name: str,
    cl_q_low_speed_: float,
    cl_q_cruise_: float,
):
    # Research independent input value in .xml file
    problem = setup_and_run_system(
        ComputeCLPitchVelocityAircraft(low_speed_aero=True), __file__, xml_file_name
    )
    assert problem.get_val("data:aerodynamics:aircraft:low_speed:CL_q") == pytest.approx(
        cl_q_low_speed_, rel=1e-3
    )

    problem.check_partials(compact_print=True)

    # Research independent input value in .xml file
    problem = setup_and_run_system(
        ComputeCLPitchVelocityAircraft(low_speed_aero=False), __file__, xml_file_name
    )
    assert problem.get_val("data:aerodynamics:aircraft:cruise:CL_q") == pytest.approx(
        cl_q_cruise_, rel=1e-3
    )

    problem.check_partials(compact_print=True)


def side_force_sideslip_derivative_wing(
    xml_file_name: str,
    cy_beta_wing_: float,
):
    # Research independent input value in .xml file
    problem = setup_and_run_system(ComputeCyBetaWing(), __file__, xml_file_name)
    assert problem.get_val("data:aerodynamics:wing:Cy_beta", units="rad**-1") == pytest.approx(
        cy_beta_wing_, rel=1e-3
    )

    problem.check_partials(compact_print=True)


def side_force_sideslip_derivative_vt(
    xml_file_name: str,
    cy_beta_vt_low_speed_: float,
    cy_beta_vt_cruise_: float,
):
    # Research independent input value in .xml file
    problem = setup_and_run_system(
        ComputeCyBetaVerticalTail(low_speed_aero=True), __file__, xml_file_name
    )
    assert problem.get_val(
        "data:aerodynamics:vertical_tail:low_speed:Cy_beta", units="rad**-1"
    ) == pytest.approx(cy_beta_vt_low_speed_, rel=1e-3)

    problem.check_partials(compact_print=True)

    # Research independent input value in .xml file
    problem = setup_and_run_system(
        ComputeCyBetaVerticalTail(low_speed_aero=False), __file__, xml_file_name
    )
    assert problem.get_val(
        "data:aerodynamics:vertical_tail:cruise:Cy_beta", units="rad**-1"
    ) == pytest.approx(cy_beta_vt_cruise_, rel=1e-3)

    problem.check_partials(compact_print=True)


def side_force_sideslip_aircraft(
    xml_file_name: str,
    cy_beta_low_speed_: float,
):
    # Only testing the low speed case since the high can't run on its own (fuselage and wing
    # contribution are independent of mach number and are thus only computed at low speed)
    # Research independent input value in .xml file
    problem = setup_and_run_system(
        ComputeCYBetaAircraft(low_speed_aero=True), __file__, xml_file_name
    )
    assert problem.get_val(
        "data:aerodynamics:aircraft:low_speed:Cy_beta", units="rad**-1"
    ) == pytest.approx(cy_beta_low_speed_, rel=1e-3)

    problem.check_partials(compact_print=True)


def side_force_yaw_rate_aircraft(
    xml_file_name: str,
    cy_yaw_rate_low_speed_: float,
    cy_yaw_rate_cruise_: float,
):
    # Research independent input value in .xml file
    problem = setup_and_run_system(
        ComputeCyYawRateAircraft(low_speed_aero=True), __file__, xml_file_name
    )
    assert problem.get_val(
        "data:aerodynamics:aircraft:low_speed:Cy_r", units="rad**-1"
    ) == pytest.approx(cy_yaw_rate_low_speed_, rel=1e-3)

    problem.check_partials(compact_print=True)

    # Research independent input value in .xml file
    problem = setup_and_run_system(
        ComputeCyYawRateAircraft(low_speed_aero=False), __file__, xml_file_name
    )
    assert problem.get_val(
        "data:aerodynamics:aircraft:cruise:Cy_r", units="rad**-1"
    ) == pytest.approx(cy_yaw_rate_cruise_, rel=1e-3)

    problem.check_partials(compact_print=True)


def side_force_roll_rate_aircraft(
    xml_file_name: str,
    cy_roll_rate_low_speed_: float,
    cy_roll_rate_cruise_: float,
):
    # Research independent input value in .xml file
    problem = setup_and_run_system(
        ComputeCyRollRateAircraft(low_speed_aero=True), __file__, xml_file_name
    )
    assert problem.get_val(
        "data:aerodynamics:aircraft:low_speed:Cy_p", units="rad**-1"
    ) == pytest.approx(cy_roll_rate_low_speed_, rel=1e-3)

    problem.check_partials(compact_print=True)

    # Research independent input value in .xml file
    problem = setup_and_run_system(
        ComputeCyRollRateAircraft(low_speed_aero=False), __file__, xml_file_name
    )
    assert problem.get_val(
        "data:aerodynamics:aircraft:cruise:Cy_p", units="rad**-1"
    ) == pytest.approx(cy_roll_rate_cruise_, rel=1e-3)

    problem.check_partials(compact_print=True)


def roll_moment_side_slip_wing(
    xml_file_name: str,
    cl_beta_wing_low_speed_: float,
    cl_beta_wing_cruise_: float,
):
    # Research independent input value in .xml file
    problem = setup_and_run_system(ComputeClBetaWing(low_speed_aero=True), __file__, xml_file_name)
    assert problem.get_val(
        "data:aerodynamics:wing:low_speed:Cl_beta", units="rad**-1"
    ) == pytest.approx(cl_beta_wing_low_speed_, rel=1e-3)

    # Research independent input value in .xml file
    problem = setup_and_run_system(ComputeClBetaWing(low_speed_aero=False), __file__, xml_file_name)
    assert problem.get_val(
        "data:aerodynamics:wing:cruise:Cl_beta", units="rad**-1"
    ) == pytest.approx(cl_beta_wing_cruise_, rel=1e-3)


def roll_moment_side_slip_ht(
    xml_file_name: str,
    cl_beta_ht_low_speed_: float,
    cl_beta_ht_cruise_: float,
):
    # Research independent input value in .xml file
    problem = setup_and_run_system(
        ComputeClBetaHorizontalTail(low_speed_aero=True), __file__, xml_file_name
    )
    assert problem.get_val(
        "data:aerodynamics:horizontal_tail:low_speed:Cl_beta", units="rad**-1"
    ) == pytest.approx(cl_beta_ht_low_speed_, rel=1e-3)

    # Research independent input value in .xml file
    problem = setup_and_run_system(
        ComputeClBetaHorizontalTail(low_speed_aero=False), __file__, xml_file_name
    )
    assert problem.get_val(
        "data:aerodynamics:horizontal_tail:cruise:Cl_beta", units="rad**-1"
    ) == pytest.approx(cl_beta_ht_cruise_, rel=1e-3)


def roll_moment_side_slip_vt(
    xml_file_name: str,
    cl_beta_vt_low_speed_: float,
    cl_beta_vt_cruise_: float,
):
    # Research independent input value in .xml file
    problem = setup_and_run_system(
        ComputeClBetaVerticalTail(low_speed_aero=True), __file__, xml_file_name
    )
    assert problem.get_val(
        "data:aerodynamics:vertical_tail:low_speed:Cl_beta", units="rad**-1"
    ) == pytest.approx(cl_beta_vt_low_speed_, rel=1e-3)

    problem.check_partials(compact_print=True)

    # Research independent input value in .xml file
    problem = setup_and_run_system(
        ComputeClBetaVerticalTail(low_speed_aero=False), __file__, xml_file_name
    )
    assert problem.get_val(
        "data:aerodynamics:vertical_tail:cruise:Cl_beta", units="rad**-1"
    ) == pytest.approx(cl_beta_vt_cruise_, rel=1e-3)

    problem.check_partials(compact_print=True)


def roll_moment_side_slip_aircraft(
    xml_file_name: str,
    cl_beta_low_speed_: float,
    cl_beta_cruise_: float,
):
    # Research independent input value in .xml file
    problem = setup_and_run_system(
        ComputeClBetaAircraft(low_speed_aero=True), __file__, xml_file_name
    )
    assert problem.get_val(
        "data:aerodynamics:aircraft:low_speed:Cl_beta", units="rad**-1"
    ) == pytest.approx(cl_beta_low_speed_, rel=1e-3)

    # We already check them individually
    problem.check_partials(
        compact_print=True,
        excludes=["*wing_contribution*", "*ht_contribution*", "*vt_contribution*"],
    )

    # Research independent input value in .xml file
    problem = setup_and_run_system(
        ComputeClBetaAircraft(low_speed_aero=False), __file__, xml_file_name
    )
    assert problem.get_val(
        "data:aerodynamics:aircraft:cruise:Cl_beta", units="rad**-1"
    ) == pytest.approx(cl_beta_cruise_, rel=1e-3)

    # We already check them individually
    problem.check_partials(
        compact_print=True,
        excludes=["*wing_contribution*", "*ht_contribution*", "*vt_contribution*"],
    )


def roll_moment_roll_rate_wing(
    xml_file_name: str,
    cl_p_wing_low_speed_: float,
    cl_p_wing_cruise_: float,
):
    # Research independent input value in .xml file
    problem = setup_and_run_system(
        ComputeClRollRateWing(low_speed_aero=True), __file__, xml_file_name
    )
    assert problem.get_val(
        "data:aerodynamics:wing:low_speed:Cl_p", units="rad**-1"
    ) == pytest.approx(cl_p_wing_low_speed_, rel=1e-3)

    # Research independent input value in .xml file
    problem = setup_and_run_system(
        ComputeClRollRateWing(low_speed_aero=False), __file__, xml_file_name
    )
    assert problem.get_val("data:aerodynamics:wing:cruise:Cl_p", units="rad**-1") == pytest.approx(
        cl_p_wing_cruise_, rel=1e-3
    )


def roll_moment_roll_rate_ht(
    xml_file_name: str,
    cl_p_ht_low_speed_: float,
    cl_p_ht_cruise_: float,
):
    # Research independent input value in .xml file
    problem = setup_and_run_system(
        ComputeClRollRateHorizontalTail(low_speed_aero=True), __file__, xml_file_name
    )
    assert problem.get_val(
        "data:aerodynamics:horizontal_tail:low_speed:Cl_p", units="rad**-1"
    ) == pytest.approx(cl_p_ht_low_speed_, rel=1e-3)

    # Research independent input value in .xml file
    problem = setup_and_run_system(
        ComputeClRollRateHorizontalTail(low_speed_aero=False), __file__, xml_file_name
    )
    assert problem.get_val(
        "data:aerodynamics:horizontal_tail:cruise:Cl_p", units="rad**-1"
    ) == pytest.approx(cl_p_ht_cruise_, rel=1e-3)


def roll_moment_roll_rate_vt(
    xml_file_name: str,
    cl_p_vt_low_speed_: float,
    cl_p_vt_cruise_: float,
):
    # Research independent input value in .xml file
    problem = setup_and_run_system(
        ComputeClRollRateVerticalTail(low_speed_aero=True), __file__, xml_file_name
    )
    assert problem.get_val(
        "data:aerodynamics:vertical_tail:low_speed:Cl_p", units="rad**-1"
    ) == pytest.approx(cl_p_vt_low_speed_, rel=1e-3)

    problem.check_partials(compact_print=True)

    # Research independent input value in .xml file
    problem = setup_and_run_system(
        ComputeClRollRateVerticalTail(low_speed_aero=False), __file__, xml_file_name
    )
    assert problem.get_val(
        "data:aerodynamics:vertical_tail:cruise:Cl_p", units="rad**-1"
    ) == pytest.approx(cl_p_vt_cruise_, rel=1e-3)

    problem.check_partials(compact_print=True)


def roll_moment_roll_rate_aircraft(
    xml_file_name: str,
    cl_p_low_speed_: float,
    cl_p_cruise_: float,
):
    # Research independent input value in .xml file
    problem = setup_and_run_system(
        ComputeClRollRateAircraft(low_speed_aero=True), __file__, xml_file_name
    )
    assert problem.get_val(
        "data:aerodynamics:aircraft:low_speed:Cl_p", units="rad**-1"
    ) == pytest.approx(cl_p_low_speed_, rel=1e-3)

    # Checked individually
    problem.check_partials(
        compact_print=True,
        excludes=["*wing_contribution*", "*ht_contribution*", "*vt_contribution*"],
    )

    # Research independent input value in .xml file
    problem = setup_and_run_system(
        ComputeClRollRateAircraft(low_speed_aero=False), __file__, xml_file_name
    )
    assert problem.get_val(
        "data:aerodynamics:aircraft:cruise:Cl_p", units="rad**-1"
    ) == pytest.approx(cl_p_cruise_, rel=1e-3)

    # Checked individually
    problem.check_partials(
        compact_print=True,
        excludes=["*wing_contribution*", "*ht_contribution*", "*vt_contribution*"],
    )


def roll_moment_yaw_rate_wing(
    xml_file_name: str,
    cl_r_wing_low_speed_: float,
    cl_r_wing_cruise_: float,
):
    # Research independent input value in .xml file
    problem = setup_and_run_system(
        ComputeClYawRateWing(low_speed_aero=True), __file__, xml_file_name
    )

    assert problem.get_val(
        "data:aerodynamics:wing:low_speed:Cl_r", units="rad**-1"
    ) == pytest.approx(cl_r_wing_low_speed_, rel=1e-3)

    problem.check_partials(compact_print=True)

    # Research independent input value in .xml file
    problem = setup_and_run_system(
        ComputeClYawRateWing(low_speed_aero=False), __file__, xml_file_name
    )

    assert problem.get_val("data:aerodynamics:wing:cruise:Cl_r", units="rad**-1") == pytest.approx(
        cl_r_wing_cruise_, rel=1e-3
    )

    problem.check_partials(compact_print=True)


def roll_moment_yaw_rate_vt(
    xml_file_name: str,
    cl_r_vt_low_speed_: float,
    cl_r_vt_cruise_: float,
):
    # Research independent input value in .xml file
    problem = setup_and_run_system(
        ComputeClYawRateVerticalTail(low_speed_aero=True), __file__, xml_file_name
    )
    assert problem.get_val(
        "data:aerodynamics:vertical_tail:low_speed:Cl_r", units="rad**-1"
    ) == pytest.approx(cl_r_vt_low_speed_, rel=1e-3)

    problem.check_partials(compact_print=True)

    # Research independent input value in .xml file
    problem = setup_and_run_system(
        ComputeClYawRateVerticalTail(low_speed_aero=False), __file__, xml_file_name
    )
    assert problem.get_val(
        "data:aerodynamics:vertical_tail:cruise:Cl_r", units="rad**-1"
    ) == pytest.approx(cl_r_vt_cruise_, rel=1e-3)
    problem.check_partials(compact_print=True)


def roll_moment_yaw_rate_aircraft(
    xml_file_name: str,
    cl_r_low_speed_: float,
    cl_r_cruise_: float,
):
    # Research independent input value in .xml file
    problem = setup_and_run_system(
        ComputeClYawRateAircraft(low_speed_aero=True), __file__, xml_file_name
    )
    assert problem.get_val(
        "data:aerodynamics:aircraft:low_speed:Cl_r", units="rad**-1"
    ) == pytest.approx(cl_r_low_speed_, rel=1e-3)

    # Checked individually
    problem.check_partials(
        compact_print=True, excludes=["*wing_contribution*", "*vt_contribution*"]
    )

    # Research independent input value in .xml file
    problem = setup_and_run_system(
        ComputeClYawRateAircraft(low_speed_aero=False), __file__, xml_file_name
    )
    assert problem.get_val(
        "data:aerodynamics:aircraft:cruise:Cl_r", units="rad**-1"
    ) == pytest.approx(cl_r_cruise_, rel=1e-3)

    # Checked individually
    problem.check_partials(
        compact_print=True, excludes=["*wing_contribution*", "*vt_contribution*"]
    )


def roll_authority_aileron(
    xml_file_name: str,
    cl_delta_a_low_speed_: float,
    cl_delta_a_cruise_: float,
):
    # Research independent input value in .xml file
    problem = setup_and_run_system(
        ComputeClDeltaAileron(low_speed_aero=True), __file__, xml_file_name
    )
    assert problem.get_val(
        "data:aerodynamics:aileron:low_speed:Cl_delta_a", units="rad**-1"
    ) == pytest.approx(cl_delta_a_low_speed_, rel=1e-3)

    problem.check_partials(compact_print=True)

    # Research independent input value in .xml file
    problem = setup_and_run_system(
        ComputeClDeltaAileron(low_speed_aero=False), __file__, xml_file_name
    )
    assert problem.get_val(
        "data:aerodynamics:aileron:cruise:Cl_delta_a", units="rad**-1"
    ) == pytest.approx(cl_delta_a_cruise_, rel=1e-3)

    problem.check_partials(compact_print=True)


def roll_moment_rudder(
    xml_file_name: str,
    cl_delta_r_low_speed_: float,
    cl_delta_r_cruise_: float,
):
    # Research independent input value in .xml file
    problem = setup_and_run_system(
        ComputeClDeltaRudder(low_speed_aero=True), __file__, xml_file_name
    )
    assert problem.get_val(
        "data:aerodynamics:rudder:low_speed:Cl_delta_r", units="rad**-1"
    ) == pytest.approx(cl_delta_r_low_speed_, rel=1e-3)

    problem.check_partials(compact_print=True)

    # Research independent input value in .xml file
    problem = setup_and_run_system(
        ComputeClDeltaRudder(low_speed_aero=False), __file__, xml_file_name
    )
    assert problem.get_val(
        "data:aerodynamics:rudder:cruise:Cl_delta_r", units="rad**-1"
    ) == pytest.approx(cl_delta_r_cruise_, rel=1e-3)

    problem.check_partials(compact_print=True)


def pitch_moment_pitch_rate_wing(
    xml_file_name: str,
    cm_q_wing_low_speed_: float,
    cm_q_wing_cruise_: float,
):
    # Research independent input value in .xml file
    problem = setup_and_run_system(
        ComputeCMPitchVelocityWing(low_speed_aero=True), __file__, xml_file_name
    )
    assert problem.get_val(
        "data:aerodynamics:wing:low_speed:Cm_q", units="rad**-1"
    ) == pytest.approx(cm_q_wing_low_speed_, rel=1e-3)

    # Research independent input value in .xml file
    problem = setup_and_run_system(
        ComputeCMPitchVelocityWing(low_speed_aero=False), __file__, xml_file_name
    )
    assert problem.get_val("data:aerodynamics:wing:cruise:Cm_q", units="rad**-1") == pytest.approx(
        cm_q_wing_cruise_, rel=1e-3
    )


def pitch_moment_pitch_rate_ht(
    xml_file_name: str,
    cm_q_ht_low_speed_: float,
    cm_q_ht_cruise_: float,
):
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(ComputeCMPitchVelocityHorizontalTail(low_speed_aero=True)),
        __file__,
        xml_file_name,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeCMPitchVelocityHorizontalTail(low_speed_aero=True), ivc)
    assert problem.get_val(
        "data:aerodynamics:horizontal_tail:low_speed:Cm_q", units="rad**-1"
    ) == pytest.approx(cm_q_ht_low_speed_, rel=1e-3)

    problem.check_partials(compact_print=True)

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(ComputeCMPitchVelocityHorizontalTail(low_speed_aero=False)),
        __file__,
        xml_file_name,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeCMPitchVelocityHorizontalTail(low_speed_aero=False), ivc)
    assert problem.get_val(
        "data:aerodynamics:horizontal_tail:cruise:Cm_q", units="rad**-1"
    ) == pytest.approx(cm_q_ht_cruise_, rel=1e-3)

    problem.check_partials(compact_print=True)


def pitch_moment_pitch_rate_aircraft(
    xml_file_name: str,
    cm_q_low_speed_: float,
    cm_q_cruise_: float,
):
    # Research independent input value in .xml file
    problem = setup_and_run_system(
        ComputeCMPitchVelocityAircraft(low_speed_aero=True), __file__, xml_file_name
    )
    assert problem.get_val(
        "data:aerodynamics:aircraft:low_speed:Cm_q", units="rad**-1"
    ) == pytest.approx(cm_q_low_speed_, rel=1e-3)

    # Checked individually
    problem.check_partials(
        compact_print=True, excludes=["*wing_contribution*", "*ht_contribution*"]
    )

    # Research independent input value in .xml file
    problem = setup_and_run_system(
        ComputeCMPitchVelocityAircraft(low_speed_aero=False), __file__, xml_file_name
    )
    assert problem.get_val(
        "data:aerodynamics:aircraft:cruise:Cm_q", units="rad**-1"
    ) == pytest.approx(cm_q_cruise_, rel=1e-3)

    # Checked individually
    problem.check_partials(
        compact_print=True, excludes=["*wing_contribution*", "*ht_contribution*"]
    )


def pitch_moment_aoa_rate_derivative(
    xml_file_name: str,
    cm_aoa_dot_low_speed_: float,
    cm_aoa_dot_cruise_: float,
):
    # Research independent input value in .xml file
    problem = setup_and_run_system(
        ComputeCMAlphaDotAircraft(low_speed_aero=True), __file__, xml_file_name
    )
    assert problem.get_val("data:aerodynamics:aircraft:low_speed:Cm_alpha_dot") == pytest.approx(
        cm_aoa_dot_low_speed_, rel=1e-3
    )

    problem.check_partials(compact_print=True)

    # Research independent input value in .xml file
    problem = setup_and_run_system(
        ComputeCMAlphaDotAircraft(low_speed_aero=False), __file__, xml_file_name
    )
    assert problem.get_val("data:aerodynamics:aircraft:cruise:Cm_alpha_dot") == pytest.approx(
        cm_aoa_dot_cruise_, rel=1e-3
    )

    problem.check_partials(compact_print=True)


def yaw_moment_sideslip_derivative_vt(
    xml_file_name: str,
    cn_beta_vt_low_speed_: float,
    cn_beta_vt_cruise_: float,
):
    # Research independent input value in .xml file
    problem = setup_and_run_system(
        ComputeCnBetaVerticalTail(low_speed_aero=True), __file__, xml_file_name
    )
    assert problem.get_val(
        "data:aerodynamics:vertical_tail:low_speed:Cn_beta", units="rad**-1"
    ) == pytest.approx(cn_beta_vt_low_speed_, rel=1e-3)

    problem.check_partials(compact_print=True)

    # Research independent input value in .xml file
    problem = setup_and_run_system(
        ComputeCnBetaVerticalTail(low_speed_aero=False), __file__, xml_file_name
    )
    assert problem.get_val(
        "data:aerodynamics:vertical_tail:cruise:Cn_beta", units="rad**-1"
    ) == pytest.approx(cn_beta_vt_cruise_, rel=1e-3)

    problem.check_partials(compact_print=True)


def yaw_moment_sideslip_aircraft(
    xml_file_name: str,
    cn_beta_low_speed_: float,
):
    # Only testing the low speed case since the high can't run on its own (fuselage is
    # independent of mach number and are thus only computed at low speed)
    # Research independent input value in .xml file
    problem = setup_and_run_system(
        ComputeCnBetaAircraft(low_speed_aero=True), __file__, xml_file_name
    )
    assert problem.get_val(
        "data:aerodynamics:aircraft:low_speed:Cn_beta", units="rad**-1"
    ) == pytest.approx(cn_beta_low_speed_, rel=1e-3)

    problem.check_partials(compact_print=True)


def yaw_moment_aileron(
    xml_file_name: str,
    cn_delta_a_low_speed_: float,
    cn_delta_a_cruise_: float,
):
    # Research independent input value in .xml file
    problem = setup_and_run_system(
        ComputeCnDeltaAileron(low_speed_aero=True), __file__, xml_file_name
    )
    assert problem.get_val(
        "data:aerodynamics:aileron:low_speed:Cn_delta_a", units="rad**-1"
    ) == pytest.approx(cn_delta_a_low_speed_, rel=1e-3)

    problem.check_partials(compact_print=True)

    # Research independent input value in .xml file
    problem = setup_and_run_system(
        ComputeCnDeltaAileron(low_speed_aero=False), __file__, xml_file_name
    )
    assert problem.get_val(
        "data:aerodynamics:aileron:cruise:Cn_delta_a", units="rad**-1"
    ) == pytest.approx(cn_delta_a_cruise_, rel=1e-3)

    problem.check_partials(compact_print=True)


def yaw_moment_rudder(
    xml_file_name: str,
    cn_delta_r_low_speed_: float,
    cn_delta_r_cruise_: float,
):
    # Research independent input value in .xml file
    problem = setup_and_run_system(
        ComputeCnDeltaRudder(low_speed_aero=True), __file__, xml_file_name
    )
    assert problem.get_val(
        "data:aerodynamics:rudder:low_speed:Cn_delta_r", units="rad**-1"
    ) == pytest.approx(cn_delta_r_low_speed_, rel=1e-3)

    problem.check_partials(compact_print=True)

    # Research independent input value in .xml file
    problem = setup_and_run_system(
        ComputeCnDeltaRudder(low_speed_aero=False), __file__, xml_file_name
    )
    assert problem.get_val(
        "data:aerodynamics:rudder:cruise:Cn_delta_r", units="rad**-1"
    ) == pytest.approx(cn_delta_r_cruise_, rel=1e-3)

    problem.check_partials(compact_print=True)


def yaw_moment_roll_rate_wing(
    xml_file_name: str,
    cn_p_wing_low_speed_: float,
    cn_p_wing_cruise_: float,
):
    # Research independent input value in .xml file
    problem = setup_and_run_system(
        ComputeCnRollRateWing(low_speed_aero=True), __file__, xml_file_name
    )
    assert problem.get_val(
        "data:aerodynamics:wing:low_speed:Cn_p", units="rad**-1"
    ) == pytest.approx(cn_p_wing_low_speed_, rel=1e-3)

    # Research independent input value in .xml file
    problem = setup_and_run_system(
        ComputeCnRollRateWing(low_speed_aero=False), __file__, xml_file_name
    )
    assert problem.get_val("data:aerodynamics:wing:cruise:Cn_p", units="rad**-1") == pytest.approx(
        cn_p_wing_cruise_, rel=1e-3
    )

    problem.check_partials(compact_print=True)


def yaw_moment_roll_rate_vt(
    xml_file_name: str,
    cn_p_vt_low_speed_: float,
    cn_p_vt_cruise_: float,
):
    # Research independent input value in .xml file
    problem = setup_and_run_system(
        ComputeCnRollRateVerticalTail(low_speed_aero=True), __file__, xml_file_name
    )
    assert problem.get_val(
        "data:aerodynamics:vertical_tail:low_speed:Cn_p", units="rad**-1"
    ) == pytest.approx(cn_p_vt_low_speed_, rel=1e-3)

    problem.check_partials(compact_print=True)

    # Research independent input value in .xml file
    problem = setup_and_run_system(
        ComputeCnRollRateVerticalTail(low_speed_aero=False), __file__, xml_file_name
    )
    assert problem.get_val(
        "data:aerodynamics:vertical_tail:cruise:Cn_p", units="rad**-1"
    ) == pytest.approx(cn_p_vt_cruise_, rel=1e-3)

    problem.check_partials(compact_print=True)


def yaw_moment_roll_rate_aircraft(
    xml_file_name: str,
    cn_p_low_speed_: float,
    cn_p_cruise_: float,
):
    # Research independent input value in .xml file
    problem = setup_and_run_system(
        ComputeCnRollRateAircraft(low_speed_aero=True), __file__, xml_file_name
    )
    assert problem.get_val(
        "data:aerodynamics:aircraft:low_speed:Cn_p", units="rad**-1"
    ) == pytest.approx(cn_p_low_speed_, rel=1e-3)

    # Individually checked
    problem.check_partials(
        compact_print=True, excludes=["*wing_contribution*", "*vt_contribution*"]
    )

    # Research independent input value in .xml file
    problem = setup_and_run_system(
        ComputeCnRollRateAircraft(low_speed_aero=False), __file__, xml_file_name
    )
    assert problem.get_val(
        "data:aerodynamics:aircraft:cruise:Cn_p", units="rad**-1"
    ) == pytest.approx(cn_p_cruise_, rel=1e-3)

    # Individually checked
    problem.check_partials(
        compact_print=True, excludes=["*wing_contribution*", "*vt_contribution*"]
    )


def yaw_moment_yaw_rate_wing(
    xml_file_name: str,
    cn_r_wing_low_speed_: float,
    cn_r_wing_cruise_: float,
):
    # Research independent input value in .xml file
    problem = setup_and_run_system(
        ComputeCnYawRateWing(low_speed_aero=True), __file__, xml_file_name
    )
    assert problem.get_val(
        "data:aerodynamics:wing:low_speed:Cn_r", units="rad**-1"
    ) == pytest.approx(cn_r_wing_low_speed_, rel=1e-3)

    problem.check_partials(compact_print=True)

    # Research independent input value in .xml file
    problem = setup_and_run_system(
        ComputeCnYawRateWing(low_speed_aero=False), __file__, xml_file_name
    )
    assert problem.get_val("data:aerodynamics:wing:cruise:Cn_r", units="rad**-1") == pytest.approx(
        cn_r_wing_cruise_, rel=1e-3
    )

    problem.check_partials(compact_print=True)


def yaw_moment_yaw_rate_vt(
    xml_file_name: str,
    cn_r_vt_low_speed_: float,
    cn_r_vt_cruise_: float,
):
    # Research independent input value in .xml file
    problem = setup_and_run_system(
        ComputeCnYawRateVerticalTail(low_speed_aero=True), __file__, xml_file_name
    )
    assert problem.get_val(
        "data:aerodynamics:vertical_tail:low_speed:Cn_r", units="rad**-1"
    ) == pytest.approx(cn_r_vt_low_speed_, rel=1e-3)

    problem.check_partials(compact_print=True)

    # Research independent input value in .xml file
    problem = setup_and_run_system(
        ComputeCnYawRateVerticalTail(low_speed_aero=False), __file__, xml_file_name
    )
    assert problem.get_val(
        "data:aerodynamics:vertical_tail:cruise:Cn_r", units="rad**-1"
    ) == pytest.approx(cn_r_vt_cruise_, rel=1e-3)

    problem.check_partials(compact_print=True)


def yaw_moment_yaw_rate_aircraft(
    xml_file_name: str,
    cn_r_low_speed_: float,
    cn_r_cruise_: float,
):
    # Research independent input value in .xml file
    problem = setup_and_run_system(
        ComputeCnYawRateAircraft(low_speed_aero=True), __file__, xml_file_name
    )
    assert problem.get_val(
        "data:aerodynamics:aircraft:low_speed:Cn_r", units="rad**-1"
    ) == pytest.approx(cn_r_low_speed_, rel=1e-3)

    # Checked individually
    problem.check_partials(
        compact_print=True, excludes=["*wing_contribution*", "*vt_contribution*"]
    )

    # Research independent input value in .xml file
    problem = setup_and_run_system(
        ComputeCnYawRateAircraft(low_speed_aero=False), __file__, xml_file_name
    )
    assert problem.get_val(
        "data:aerodynamics:aircraft:cruise:Cn_r", units="rad**-1"
    ) == pytest.approx(cn_r_cruise_, rel=1e-3)

    # Checked individually
    problem.check_partials(
        compact_print=True, excludes=["*wing_contribution*", "*vt_contribution*"]
    )
