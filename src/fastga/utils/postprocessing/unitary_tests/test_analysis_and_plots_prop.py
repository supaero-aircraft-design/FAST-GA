"""Tests for analysis and plots functions"""
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

import os.path as pth

from ..propeller.analysis_and_plots_propeller import (
    propeller_coeff_map_plot,
    propeller_efficiency_map_plot,
)

DATA_FOLDER_PATH = pth.join(pth.dirname(__file__), "data")


def test_efficiency_map_plot():
    """Basic tests for testing the plotting."""

    filename = pth.join(DATA_FOLDER_PATH, "problem_outputs_propeller.xml")

    # First plot
    # This is a rudimentary test as plot are difficult to verify
    # The test will fail if an error is raised by the following line
    fig = propeller_efficiency_map_plot(filename)

    # Second plot with sea_level option
    # This is a rudimentary test as plot are difficult to verify
    # The test will fail if an error is raised by the following line
    fig = propeller_efficiency_map_plot(filename, sea_level=True)


def test_coefficient_map_plot():
    """Basic tests for testing the plotting."""

    filename = pth.join(DATA_FOLDER_PATH, "problem_outputs_propeller.xml")

    # First plot
    # This is a rudimentary test as plot are difficult to verify
    # The test will fail if an error is raised by the following line
    fig = propeller_coeff_map_plot(filename)

    # First plot with name
    # This is a rudimentary test as plot are difficult to verify
    # The test will fail if an error is raised by the following line
    fig = propeller_coeff_map_plot(filename, name="First plot")

    # Adding a plot to the previous fig
    # This is a rudimentary test as plot are difficult to verify
    # The test will fail if an error is raised by the following line
    fig = propeller_coeff_map_plot(filename, name="Second plot", fig=fig)
