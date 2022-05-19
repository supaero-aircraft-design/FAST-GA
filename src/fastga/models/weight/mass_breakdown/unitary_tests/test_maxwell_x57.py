"""
Test module for mass breakdown functions.
"""
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

import pytest

from ..a_airframe.wing_components import (
    ComputeWebMass,
    ComputeLowerFlange,
    ComputeUpperFlange,
    ComputeSkinMass,
    ComputeMiscMass,
    ComputeRibsMass,
    ComputePrimaryMass,
    ComputeSecondaryMass,
    UpdateWingMass,
)
from ..a_airframe import ComputeWingMassAnalytical

from tests.testing_utilities import run_system, get_indep_var_comp, list_inputs

XML_FILE = "maxwell_x57.xml"


def test_compute_web_mass():
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(ComputeWebMass()), __file__, XML_FILE)
    ivc.add_output("data:weight:airframe:wing:mass", val=174.348, units="kg")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeWebMass(), ivc)
    assert problem["data:weight:airframe:wing:web:mass:max_fuel_in_wing"] == pytest.approx(
        2.408, abs=1e-2
    )

    ivc2 = get_indep_var_comp(
        list_inputs(ComputeWebMass(min_fuel_in_wing=True)), __file__, XML_FILE
    )
    ivc2.add_output("data:weight:airframe:wing:mass", val=174.348, units="kg")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeWebMass(min_fuel_in_wing=True), ivc2)
    assert problem["data:weight:airframe:wing:web:mass:min_fuel_in_wing"] == pytest.approx(
        2.408, abs=1e-2
    )


def test_compute_upper_flange_mass():
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(ComputeUpperFlange()), __file__, XML_FILE)
    ivc.add_output("data:weight:airframe:wing:mass", val=174.348, units="kg")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeUpperFlange(), ivc)
    assert problem["data:weight:airframe:wing:upper_flange:mass:max_fuel_in_wing"] == pytest.approx(
        27.465, abs=1e-2
    )

    ivc2 = get_indep_var_comp(
        list_inputs(ComputeUpperFlange(min_fuel_in_wing=True)), __file__, XML_FILE
    )
    ivc2.add_output("data:weight:airframe:wing:mass", val=174.348, units="kg")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeUpperFlange(min_fuel_in_wing=True), ivc2)
    assert problem["data:weight:airframe:wing:upper_flange:mass:min_fuel_in_wing"] == pytest.approx(
        27.465, abs=1e-2
    )


def test_compute_lower_flange_mass():
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(ComputeLowerFlange()), __file__, XML_FILE)
    ivc.add_output("data:weight:airframe:wing:mass", val=174.348, units="kg")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeLowerFlange(), ivc)
    assert problem["data:weight:airframe:wing:lower_flange:mass:max_fuel_in_wing"] == pytest.approx(
        20.584, abs=1e-2
    )

    ivc2 = get_indep_var_comp(
        list_inputs(ComputeLowerFlange(min_fuel_in_wing=True)), __file__, XML_FILE
    )
    ivc2.add_output("data:weight:airframe:wing:mass", val=174.348, units="kg")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeLowerFlange(min_fuel_in_wing=True), ivc2)
    assert problem["data:weight:airframe:wing:lower_flange:mass:min_fuel_in_wing"] == pytest.approx(
        20.584, abs=1e-2
    )


def test_compute_skin_mass():
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(ComputeSkinMass()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeSkinMass(), ivc)
    assert problem["data:weight:airframe:wing:skin:mass"] == pytest.approx(68.501, abs=1e-2)


def test_compute_ribs_mass():
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(ComputeRibsMass()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeRibsMass(), ivc)
    assert problem["data:weight:airframe:wing:ribs:mass"] == pytest.approx(3.205, abs=1e-2)


def test_compute_misc_mass():
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(ComputeMiscMass()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeMiscMass(), ivc)
    assert problem["data:weight:airframe:wing:misc:mass"] == pytest.approx(8.639, abs=1e-2)


def test_compute_primary_mass():
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(ComputePrimaryMass()), __file__, XML_FILE)
    ivc.add_output("data:weight:airframe:wing:web:mass:max_fuel_in_wing", val=2.408, units="kg")
    ivc.add_output("data:weight:airframe:wing:web:mass:min_fuel_in_wing", val=2.408, units="kg")
    ivc.add_output(
        "data:weight:airframe:wing:lower_flange:mass:max_fuel_in_wing", val=20.584, units="kg"
    )
    ivc.add_output(
        "data:weight:airframe:wing:lower_flange:mass:min_fuel_in_wing", val=20.584, units="kg"
    )
    ivc.add_output(
        "data:weight:airframe:wing:upper_flange:mass:max_fuel_in_wing", val=27.465, units="kg"
    )
    ivc.add_output(
        "data:weight:airframe:wing:upper_flange:mass:min_fuel_in_wing", val=27.465, units="kg"
    )
    ivc.add_output("data:weight:airframe:wing:skin:mass", val=68.501, units="kg")
    ivc.add_output("data:weight:airframe:wing:ribs:mass", val=3.205, units="kg")
    ivc.add_output("data:weight:airframe:wing:misc:mass", val=8.639, units="kg")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputePrimaryMass(), ivc)
    assert problem["data:weight:airframe:wing:primary_structure:mass"] == pytest.approx(
        130.802, abs=1e-2
    )


def test_compute_secondary_mass():

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(ComputeSecondaryMass()), __file__, XML_FILE)
    ivc.add_output("data:weight:airframe:wing:primary_structure:mass", val=130.802, units="kg")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeSecondaryMass(), ivc)
    assert problem["data:weight:airframe:wing:secondary_structure:mass"] == pytest.approx(
        43.600, abs=1e-2
    )


def test_update_wing_mass():
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(UpdateWingMass()), __file__, XML_FILE)
    ivc.add_output("data:weight:airframe:wing:primary_structure:mass", val=130.802, units="kg")
    ivc.add_output("data:weight:airframe:wing:secondary_structure:mass", val=43.600, units="kg")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(UpdateWingMass(), ivc)
    assert problem["data:weight:airframe:wing:mass"] == pytest.approx(174.402, abs=1e-2)


def test_compute_wing_mass():
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(ComputeWingMassAnalytical()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeWingMassAnalytical(), ivc)
    assert problem["data:weight:airframe:wing:mass"] == pytest.approx(174.402, abs=1e-2)
