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

from ..load_analysis.analysis_and_plots_la import (
    force_repartition_diagram,
    shear_diagram,
    rbm_diagram,
)

DATA_FOLDER_PATH = pth.join(pth.dirname(__file__), "data")


def test_force_repartition_diagram():
    """Basic tests for testing the plotting."""

    filename = pth.join(DATA_FOLDER_PATH, "problem_outputs_loads.xml")

    # First plot
    # This is a rudimentary test as plot are difficult to verify
    # The test will fail if an error is raised by the following line
    fig = force_repartition_diagram(filename)

    # First plot with name
    # This is a rudimentary test as plot are difficult to verify
    # The test will fail if an error is raised by the following line
    fig = force_repartition_diagram(filename, name="First plot")

    # Adding a plot to the previous fig
    # This is a rudimentary test as plot are difficult to verify
    # The test will fail if an error is raised by the following line
    fig = force_repartition_diagram(filename, name="Second plot", fig=fig)


def test_shear_diagram():
    """Basic tests for testing the plotting."""

    filename = pth.join(DATA_FOLDER_PATH, "problem_outputs_loads.xml")

    # First plot
    # This is a rudimentary test as plot are difficult to verify
    # The test will fail if an error is raised by the following line
    fig = shear_diagram(filename)

    # First plot with name
    # This is a rudimentary test as plot are difficult to verify
    # The test will fail if an error is raised by the following line
    fig = shear_diagram(filename, name="First plot")

    # Adding a plot to the previous fig
    # This is a rudimentary test as plot are difficult to verify
    # The test will fail if an error is raised by the following line
    fig = shear_diagram(filename, name="Second plot", fig=fig)


def test_rbm_diagram():
    """Basic tests for testing the plotting."""

    filename = pth.join(DATA_FOLDER_PATH, "problem_outputs_loads.xml")

    # First plot
    # This is a rudimentary test as plot are difficult to verify
    # The test will fail if an error is raised by the following line
    fig = rbm_diagram(filename)

    # First plot with name
    # This is a rudimentary test as plot are difficult to verify
    # The test will fail if an error is raised by the following line
    fig = rbm_diagram(filename, name="First plot")

    # Adding a plot to the previous fig
    # This is a rudimentary test as plot are difficult to verify
    # The test will fail if an error is raised by the following line
    fig = rbm_diagram(filename, name="Second plot", fig=fig)
