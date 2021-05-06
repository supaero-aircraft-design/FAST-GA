"""
Test module for XFOIL component
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

# pylint: disable=redefined-outer-name  # needed for fixtures

import os.path as pth
import os
import shutil
from platform import system
import warnings

import pytest
from openmdao.core.indepvarcomp import IndepVarComp

from fastga.tests.testing_utilities import run_system
from fastga.tests.xfoil_exe.get_xfoil import get_xfoil_path
from ..xfoil_polar import XfoilPolar, DEFAULT_2D_CL_MAX, DEFAULT_2D_CL_MIN, _DEFAULT_AIRFOIL_FILE
from .. import resources

XFOIL_RESULTS = pth.join(pth.dirname(__file__), "results")

xfoil_path = None if system() == "Windows" else get_xfoil_path()


@pytest.mark.skipif(
    system() != "Windows" and xfoil_path is None, reason="No XFOIL executable available"
)
def test_compute():
    """ Tests a simple XFOIL run"""

    if pth.exists(XFOIL_RESULTS):
        shutil.rmtree(XFOIL_RESULTS)

    ivc = IndepVarComp()
    ivc.add_output("xfoil:reynolds", 1e6, units="m**-1")
    ivc.add_output("xfoil:mach", 0.60)

    xfoil_comp = XfoilPolar(
        alpha_start=0.0, alpha_end=15.0, iter_limit=100, symmetrical=True, xfoil_exe_path=xfoil_path
    )
    problem = run_system(xfoil_comp, ivc)
    assert problem["xfoil:CL_max_2D"] == pytest.approx(1.31, 1e-2)
    assert problem["xfoil:CL_min_2D"] == pytest.approx(-0.59, 1e-2)
    assert not pth.exists(XFOIL_RESULTS)

    # Deactivate warnings for wished crash of xfoil
    warnings.simplefilter("ignore")

    xfoil_comp = XfoilPolar(
        alpha_start=50.0, alpha_end=60.0, iter_limit=2, xfoil_exe_path=xfoil_path
    )  # will not converge
    problem = run_system(xfoil_comp, ivc)
    assert problem["xfoil:CL_max_2D"] == pytest.approx(DEFAULT_2D_CL_MAX, 1e-2)
    assert problem["xfoil:CL_min_2D"] == pytest.approx(DEFAULT_2D_CL_MIN, 1e-2)
    assert not pth.exists(XFOIL_RESULTS)

    # Reactivate warnings
    warnings.simplefilter("default")

    xfoil_comp = XfoilPolar(
        iter_limit=20, result_folder_path=XFOIL_RESULTS, xfoil_exe_path=xfoil_path
    )
    run_system(xfoil_comp, ivc)
    assert pth.exists(XFOIL_RESULTS)
    assert pth.exists(pth.join(XFOIL_RESULTS, "polar_result.txt"))
    
    # remove folder
    if pth.exists(XFOIL_RESULTS):
        shutil.rmtree(XFOIL_RESULTS)


@pytest.mark.skipif(
    system() != "Windows" and xfoil_path is None, reason="No XFOIL executable available"
)
def test_compute_with_provided_path():
    """ Test that option "use_exe_path" works """
    ivc = IndepVarComp()
    ivc.add_output("xfoil:reynolds", 1e6)
    ivc.add_output("xfoil:mach", 0.20)

    # Clear saved polar results
    if pth.exists(pth.join(resources.__path__[0], _DEFAULT_AIRFOIL_FILE.replace('af', 'csv'))):
        os.remove(pth.join(resources.__path__[0], _DEFAULT_AIRFOIL_FILE.replace('af', 'csv')))
    if pth.exists(pth.join(resources.__path__[0], _DEFAULT_AIRFOIL_FILE.replace('.af', '_sym.csv'))):
        os.remove(pth.join(resources.__path__[0], _DEFAULT_AIRFOIL_FILE.replace('.af', '_sym.csv')))

    xfoil_comp = XfoilPolar(alpha_start=0.0, alpha_end=20.0, iter_limit=20)
    xfoil_comp.options["xfoil_exe_path"] = "Dummy"  # bad name
    with pytest.raises(ValueError):
        _ = run_system(xfoil_comp, ivc)

    xfoil_comp.options["xfoil_exe_path"] = (
        xfoil_path
        if xfoil_path
        else pth.join(pth.dirname(__file__), pth.pardir, "xfoil699", "xfoil.exe")
    )
    problem = run_system(xfoil_comp, ivc)
    assert problem["xfoil:CL_max_2D"] == pytest.approx(1.48, 1e-2)
