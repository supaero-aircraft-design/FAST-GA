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

from ..analysis_and_plots import (
    aircraft_geometry_plot,
    evolution_diagram,
    compressibility_effects_diagram,
    cl_wing_diagram,
    cg_lateral_diagram,
    mass_breakdown_bar_plot,
    mass_breakdown_sun_plot,
    drag_breakdown_diagram,
    payload_range,
    aircraft_polar,
)

DATA_FOLDER_PATH = pth.join(pth.dirname(__file__), "data")


def test_aircraft_geometry_plot():
    """Basic tests for testing the plotting."""

    filename = pth.join(DATA_FOLDER_PATH, "problem_outputs.xml")

    # First plot
    # This is a rudimentary test as plot are difficult to verify
    # The test will fail if an error is raised by the following line
    fig = aircraft_geometry_plot(filename)

    # First plot with name
    # This is a rudimentary test as plot are difficult to verify
    # The test will fail if an error is raised by the following line
    fig = aircraft_geometry_plot(filename, name="First plot")

    # Adding a plot to the previous fig
    # This is a rudimentary test as plot are difficult to verify
    # The test will fail if an error is raised by the following line
    fig = aircraft_geometry_plot(filename, name="Second plot", fig=fig)


def test_evolution_diagram_plot():
    """Basic tests for testing the plotting."""

    filename = pth.join(DATA_FOLDER_PATH, "problem_outputs.xml")

    # First plot
    # This is a rudimentary test as plot are difficult to verify
    # The test will fail if an error is raised by the following line
    fig = evolution_diagram(filename)

    # First plot with name
    # This is a rudimentary test as plot are difficult to verify
    # The test will fail if an error is raised by the following line
    fig = evolution_diagram(filename, name="First plot")

    # Adding a plot to the previous fig
    # This is a rudimentary test as plot are difficult to verify
    # The test will fail if an error is raised by the following line
    fig = evolution_diagram(filename, name="Second plot", fig=fig)


def test_compressibility_effect_plot():
    """Basic tests for testing the plotting."""

    filename = pth.join(DATA_FOLDER_PATH, "problem_outputs.xml")

    # First plot
    # This is a rudimentary test as plot are difficult to verify
    # The test will fail if an error is raised by the following line
    fig = compressibility_effects_diagram(filename)

    # First plot with name
    # This is a rudimentary test as plot are difficult to verify
    # The test will fail if an error is raised by the following line
    fig = compressibility_effects_diagram(filename, name="First plot")

    # Adding a plot to the previous fig
    # This is a rudimentary test as plot are difficult to verify
    # The test will fail if an error is raised by the following line
    fig = compressibility_effects_diagram(filename, name="Second plot", fig=fig)


def test_cl_wing_diagram_plot():
    """Basic tests for testing the plotting."""

    filename = pth.join(DATA_FOLDER_PATH, "problem_outputs.xml")

    # First plot
    # This is a rudimentary test as plot are difficult to verify
    # The test will fail if an error is raised by the following line
    fig = cl_wing_diagram(filename)

    # First plot with name
    # This is a rudimentary test as plot are difficult to verify
    # The test will fail if an error is raised by the following line
    fig = cl_wing_diagram(filename, name="First plot")

    # Adding a plot to the previous fig
    # This is a rudimentary test as plot are difficult to verify
    # The test will fail if an error is raised by the following line
    fig = cl_wing_diagram(filename, name="Second plot", fig=fig)

    # adding a new plot with prop on tag
    # This is a rudimentary test as plot are difficult to verify
    # The test will fail if an error is raised by the following line
    fig = cl_wing_diagram(filename, name="Third plot", fig=fig, prop_on=True)


def test_cg_lateral_diagram_plot():
    """Basic tests for testing the plotting."""

    filename = pth.join(DATA_FOLDER_PATH, "problem_outputs.xml")

    # First plot
    # This is a rudimentary test as plot are difficult to verify
    # The test will fail if an error is raised by the following line
    fig = cg_lateral_diagram(filename)

    # First plot with name
    # This is a rudimentary test as plot are difficult to verify
    # The test will fail if an error is raised by the following line
    fig = cg_lateral_diagram(filename, name="First plot")

    # Adding a plot to the previous fig
    # This is a rudimentary test as plot are difficult to verify
    # The test will fail if an error is raised by the following line
    fig = cg_lateral_diagram(filename, name="Second plot", fig=fig)


def test_mass_breakdown_bar_plot():
    """Basic tests for testing the plotting."""

    filename = pth.join(DATA_FOLDER_PATH, "problem_outputs.xml")

    # First plot
    # This is a rudimentary test as plot are difficult to verify
    # The test will fail if an error is raised by the following line
    fig = mass_breakdown_bar_plot(filename)

    # First plot with name
    # This is a rudimentary test as plot are difficult to verify
    # The test will fail if an error is raised by the following line
    fig = mass_breakdown_bar_plot(filename, name="First plot")

    # Adding a plot to the previous fig
    # This is a rudimentary test as plot are difficult to verify
    # The test will fail if an error is raised by the following line
    fig = mass_breakdown_bar_plot(filename, name="Second plot", fig=fig)


def test_mass_breakdown_sun_plot():
    """Basic tests for testing the plotting."""

    filename = pth.join(DATA_FOLDER_PATH, "problem_outputs.xml")

    # First plot
    # This is a rudimentary test as plot are difficult to verify
    # The test will fail if an error is raised by the following line
    fig = mass_breakdown_sun_plot(filename)


def test_drag_breakdown_diagram_plot():
    """Basic tests for testing the plotting."""

    filename = pth.join(DATA_FOLDER_PATH, "problem_outputs.xml")

    # First plot
    # This is a rudimentary test as plot are difficult to verify
    # The test will fail if an error is raised by the following line
    fig = drag_breakdown_diagram(filename)


def test_payload_range_plot():
    """Basic tests for testing the plotting."""

    filename = pth.join(DATA_FOLDER_PATH, "problem_outputs.xml")

    # First plot
    # This is a rudimentary test as plot are difficult to verify
    # The test will fail if an error is raised by the following line
    fig = payload_range(filename)

    # First plot with name
    # This is a rudimentary test as plot are difficult to verify
    # The test will fail if an error is raised by the following line
    fig = payload_range(filename, name="First plot")

    # Adding a plot to the previous fig
    # This is a rudimentary test as plot are difficult to verify
    # The test will fail if an error is raised by the following line
    fig = payload_range(filename, name="Second plot", fig=fig)


def test_aircraft_polar_plot():
    """Basic tests for testing the plotting."""

    filename = pth.join(DATA_FOLDER_PATH, "problem_outputs.xml")

    # First plot
    # This is a rudimentary test as plot are difficult to verify
    # The test will fail if an error is raised by the following line
    fig = aircraft_polar(filename)

    # First plot with name
    # This is a rudimentary test as plot are difficult to verify
    # The test will fail if an error is raised by the following line
    fig = aircraft_polar(filename, name="First plot")

    # Adding a plot to the previous fig
    # This is a rudimentary test as plot are difficult to verify
    # The test will fail if an error is raised by the following line
    fig = aircraft_polar(filename, name="Second plot", fig=fig)

    # Same with equilibrated tag
    fig_2 = aircraft_polar(filename, equilibrated=True)

    fig_2 = aircraft_polar(filename, name="First plot", equilibrated=True)

    fig_2 = aircraft_polar(filename, name="Second plot", fig=fig, equilibrated=True)
