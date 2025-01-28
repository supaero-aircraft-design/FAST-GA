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

import os.path as pth
import fastoad.api as oad
import logging

DATA_FOLDER_PATH = pth.join(pth.dirname(__file__), "data")


def test_tail_mass_registry():
    """Tests tail mass submodel registry with integration ."""
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("fastoad.module_management._bundle_loader").disabled = True
    logging.getLogger("fastoad.openmdao.variables.variable").disabled = True

    # Define used files depending on options
    xml_file_name = "cirrus_sr22.xml"
    process_file_name = "dummy_conf.yml"

    configurator = oad.FASTOADProblemConfigurator(pth.join(DATA_FOLDER_PATH, process_file_name))

    # Create inputs
    ref_inputs = pth.join(DATA_FOLDER_PATH, xml_file_name)

    # Create problems with inputs
    problem = configurator.get_problem()
    problem.write_needed_inputs(ref_inputs)
    problem.read_inputs()
    problem.setup()

    assert (
        oad.RegisterSubmodel.active_models["submodel.weight.mass.airframe.htp"]
        == "fastga.submodel.weight.mass.airframe.htp.torenbeek"
    )
    assert (
        oad.RegisterSubmodel.active_models["submodel.weight.mass.airframe.vtp"]
        == "fastga.submodel.weight.mass.airframe.vtp.legacy"
    )
