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

import fastoad.api as oad

from tests.dummy_plugins.dummy_plugin_1.models.dummy_group import DummyGroup, DUMMY_SERVICE


def test_fails_when_none_and_required(with_dummy_plugin_1):
    """
    Test that the problem can't setup when the propulsion id is specified as None and is required in
    the correct format.
    """

    oad.RegisterSubmodel.active_models[DUMMY_SERVICE] = "dummy_submodel_1"

    problem = oad.FASTOADProblem()
    model = problem.model
    model.add_subsystem(name="dummy_group", subsys=DummyGroup(propulsion_id=None), promotes=["*"])

    with pytest.raises(ValueError) as e:
        problem.setup()

    # Check the error message
    assert e.value.args[0] == "Option 'propulsion_id' is expected to be a string, not None"


def test_fails_when_not_specified_and_required(with_dummy_plugin_1):
    """
    Test that the problem can't setup when the propulsion id is not specified and is required in
    the correct format.
    """

    oad.RegisterSubmodel.active_models[DUMMY_SERVICE] = "dummy_submodel_1"

    problem = oad.FASTOADProblem()
    model = problem.model
    model.add_subsystem(name="dummy_group", subsys=DummyGroup(), promotes=["*"])

    with pytest.raises(ValueError) as e:
        problem.setup()

    # Check the error message
    assert e.value.args[0] == "Option 'propulsion_id' is expected to be a string, not None"


def test_fails_when_invalid_and_required(with_dummy_plugin_1):
    """
    Test that the problem can't setup when the propulsion id is not set correctly and is required in
    the correct format.
    """

    oad.RegisterSubmodel.active_models[DUMMY_SERVICE] = "dummy_submodel_1"

    problem = oad.FASTOADProblem()
    model = problem.model
    model.add_subsystem(
        name="dummy_group", subsys=DummyGroup(propulsion_id=(48.5734053, 7.7521113)), promotes=["*"]
    )

    with pytest.raises(ValueError) as e:
        problem.setup()

    # Check the error message
    assert e.value.args[0] == "Option 'propulsion_id' is expected to be a string, not a tuple"


def test_does_not_fail_when_not_required(with_dummy_plugin_1):
    oad.RegisterSubmodel.active_models[DUMMY_SERVICE] = "dummy_submodel_2"

    problem = oad.FASTOADProblem()
    model = problem.model
    model.add_subsystem(name="dummy_group", subsys=DummyGroup(), promotes=["*"])
    problem.setup()
