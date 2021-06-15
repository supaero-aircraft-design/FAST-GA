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
import pandas as pd
from openmdao.core.component import Component
import numpy as np
from platform import system
import tempfile
from pathlib import Path
from tempfile import TemporaryDirectory
import pytest
from typing import Union
# import time

from fastoad.module_management.service_registry import RegisterPropulsion
from fastoad.model_base import FlightPoint
from fastoad.model_base.propulsion import IOMPropulsionWrapper

from ..external.xfoil import resources
from ..external.openvsp.compute_aero_slipstream_x57 import ComputeSlipstreamOpenvspX57

from tests.testing_utilities import run_system, get_indep_var_comp, list_inputs
from tests.xfoil_exe.get_xfoil import get_xfoil_path

from fastga.models.propulsion.fuel_propulsion.base import AbstractFuelPropulsion
from fastga.models.propulsion.propulsion import IPropulsion

from .dummy_engines import ENGINE_WRAPPER_X57 as ENGINE_WRAPPER

from ..external.vlm.compute_aero import DEFAULT_WING_AIRFOIL, DEFAULT_HTP_AIRFOIL

RESULTS_FOLDER = pth.join(pth.dirname(__file__), "results")
xfoil_path = None if system() == "Windows" else get_xfoil_path()

XML_FILE = "maxwell_x57.xml"


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
        if np.sum(y[idx:len(y)] == 0) == (len(y) - idx):
            y = y[0:idx]
            cl = cl[0:idx]
            break

    return y, cl


def reshape_polar(cl, cdp):
    """ Reshape data from xfoil polar vectors """
    for idx in range(len(cl)):
        if np.sum(cl[idx:len(cl)] == 0) == (len(cl) - idx):
            cl = cl[0:idx]
            cdp = cdp[0:idx]
            break
    return cl, cdp


def clear_polar_results():
    # Clear saved polar results (for wing and htp airfoils)
    if pth.exists(pth.join(resources.__path__[0], DEFAULT_WING_AIRFOIL.replace('af', 'csv'))):
        os.remove(pth.join(resources.__path__[0], DEFAULT_WING_AIRFOIL.replace('af', 'csv')))
    if pth.exists(pth.join(resources.__path__[0], DEFAULT_WING_AIRFOIL.replace('.af', '_sym.csv'))):
        os.remove(pth.join(resources.__path__[0], DEFAULT_WING_AIRFOIL.replace('.af', '_sym.csv')))
    if pth.exists(pth.join(resources.__path__[0], DEFAULT_HTP_AIRFOIL.replace('af', 'csv'))):
        os.remove(pth.join(resources.__path__[0], DEFAULT_HTP_AIRFOIL.replace('af', 'csv')))


def test_slipstream_openvsp():

    # Create result temporary directory
    results_folder = _create_tmp_directory()

    # Research independent input value in .xml file
    # noinspection PyTypeChecker
    ivc = get_indep_var_comp(list_inputs(ComputeSlipstreamOpenvspX57(
        propulsion_id=ENGINE_WRAPPER,
        result_folder_path=results_folder.name,
    )), __file__, XML_FILE)
    # Run problem and check obtained value(s) is/(are) correct
    # start = time.time()
    # noinspection PyTypeChecker
    problem = run_system(ComputeSlipstreamOpenvspX57(propulsion_id=ENGINE_WRAPPER,
                                                     result_folder_path=results_folder.name,
                                                     ), ivc)
    # stop = time.time()
    # duration_1st_run = stop - start
    y_vector_prop_on = problem.get_val("data:aerodynamics:slipstream:wing:prop_on:Y_vector", units="m")
    y_result_prop_on = np.array([0.04, 0.13, 0.22, 0.3, 0.39, 0.48, 0.57, 0.68, 0.81, 0.94, 1.07,
                                 1.21, 1.34, 1.48, 1.61, 1.75, 1.88, 2.02, 2.15, 2.29, 2.42, 2.56,
                                 2.69, 2.83, 2.96, 3.09, 3.23, 3.36, 3.49, 3.62, 3.75, 3.88, 4.,
                                 4.13, 4.26, 4.38, 4.5, 4.62, 4.74, 0., 0., 0., 0., 0.,
                                 0., 0., 0., 0., 0., 0.])
    assert np.max(np.abs(y_vector_prop_on - y_result_prop_on)) <= 1e-2
    cl_vector_prop_on = problem.get_val("data:aerodynamics:slipstream:wing:prop_on:CL_vector")
    cl_result_prop_on = np.array([1.67, 1.64, 1.64, 1.63, 1.62, 1.59, 1.57, 1.55, 1.61, 1.75, 1.82,
                                  1.67, 1.66, 1.78, 1.87, 1.68, 1.6, 1.59, 1.81, 1.71, 1.6, 1.63,
                                  1.82, 1.78, 1.57, 1.6, 1.72, 1.78, 1.55, 1.47, 1.48, 1.6, 1.6,
                                  1.38, 1.24, 1.14, 1.03, 0.88, 0.64, 0., 0., 0., 0., 0.,
                                  0., 0., 0., 0., 0., 0.])
    assert np.max(np.abs(cl_vector_prop_on - cl_result_prop_on)) <= 1e-2
    ct = problem.get_val("data:aerodynamics:slipstream:wing:prop_on:CT_ref")
    ct_result = [0.185, 0.185, 0.185, 0.185, 0.185, 0.185, 0.096]
    assert np.max(np.abs(ct - ct_result)) <= 1e-2
    delta_cl = problem.get_val("data:aerodynamics:slipstream:wing:prop_on:CL") - \
               problem.get_val("data:aerodynamics:slipstream:wing:prop_off:CL")
    assert delta_cl == pytest.approx(-0.00129, abs=1e-4)
