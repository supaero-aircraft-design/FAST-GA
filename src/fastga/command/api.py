"""
API
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

import logging
import os.path as pth
import os
import types
import copy
from importlib.resources import path
from typing import Union, List
from openmdao.core.explicitcomponent import ExplicitComponent
from openmdao.core.implicitcomponent import ImplicitComponent
from openmdao.core.group import Group
from openmdao.utils.file_wrap import InputFileGenerator
from openmdao.core.system import System
import openmdao.api as om
from copy import deepcopy

from fastoad.openmdao.variables import VariableList
from fastoad.cmd.exceptions import FastFileExistsError
from fastoad.openmdao.problem import FASTOADProblem
from fastoad.io import DataFile, IVariableIOFormatter
from fastoad.io.xml import VariableXmlStandardFormatter
from fastoad.io import VariableIO
from fastoad.io.configuration.configuration import AutoUnitsDefaultGroup

from . import resources

_LOGGER = logging.getLogger(__name__)

SAMPLE_FILENAME = "fastga.yml"


def generate_configuration_file(configuration_file_path: str, overwrite: bool = False):
    """
    Generates a sample configuration file.

    :param configuration_file_path: the path of the file to be written
    :param overwrite: if True, the file will be written, even if it already exists
    :raise FastFileExistsError: if overwrite==False and configuration_file_path already exists
    """
    if not overwrite and pth.exists(configuration_file_path):
        raise FastFileExistsError(
            "Configuration file %s not written because it already exists. "
            "Use overwrite=True to bypass." % configuration_file_path,
            configuration_file_path,
        )

    if not pth.exists(pth.split(configuration_file_path)[0]):
        os.mkdir(pth.split(configuration_file_path)[0])
    parser = InputFileGenerator()
    root_folder = resources.__path__[0]
    for i in range(2):
        root_folder = pth.split(root_folder)[0]
    package_path = "module_folders: " + root_folder.replace('\\', '/')
    with path(resources, SAMPLE_FILENAME) as input_template_path:
        parser.set_template_file(str(input_template_path))
        # noinspection PyTypeChecker
        parser.set_generated_file(configuration_file_path)
        parser.reset_anchor()
        parser.mark_anchor("module_folders:")
        parser.transfer_var(package_path, 0, 1)
        parser.generate()

    _LOGGER.info("Sample configuration written in %s", configuration_file_path)


def overwrite_models_path(configuration_file_path: str, module: types.ModuleType):
    """
    Complete a .yaml file with provided path.

    :param configuration_file_path: the path of file to be written
    :param module: the module containing models and associated register.py file
    """
    configuration_file = open(configuration_file_path)
    content = configuration_file.readlines()
    configuration_file.close()
    new_content = copy.deepcopy(content)
    models_path = pth.split(pth.abspath(module.__file__))[0]
    for idx in range(len(content)):
        if "module_folders = [" in content[idx]:
            new_content[idx] = "module_folders: " + models_path.replace('\\', '/')
            break
    configuration_file = open(configuration_file_path, "w")
    for line in new_content:
        configuration_file.write(line)
    configuration_file.close()


def write_needed_inputs(
        problem: FASTOADProblem,
        xml_file_path: str,
        source_formatter: IVariableIOFormatter = None):
    variables = DataFile(xml_file_path)
    variables.update(
        VariableList.from_unconnected_inputs(problem, with_optional_inputs=True),
        add_variables=True,
    )
    if xml_file_path:
        ref_vars = DataFile(xml_file_path, source_formatter)
        variables.update(ref_vars)
        for var in variables:
            var.is_input = True
    variables.save()


def generate_block_analysis(
        system: Union[ExplicitComponent, ImplicitComponent, Group],
        var_inputs: List,
        xml_file_path: str,
        overwrite: bool = False,
):

    # Search what are the component/group outputs
    variables = list_variables(system)
    inputs_names = [var.name for var in variables if var.is_input]
    outputs_names = [var.name for var in variables if not var.is_input]

    # Check that variable inputs are in the group/component list
    if not(set(var_inputs) == set(inputs_names).intersection(set(var_inputs))):
        raise Exception('The input list contains name(s) out of component/group input list!')

    # Perform some tests on the .xml availability and completeness
    if not(os.path.exists(xml_file_path)) and not(set(var_inputs) == set(inputs_names)):
        # If no input file and some inputs are missing, generate it and return None
        if isinstance(system, Group):
            problem = FASTOADProblem(system)
        else:
            group = AutoUnitsDefaultGroup()
            group.add_subsystem('system', system, promotes=["*"])
            problem = FASTOADProblem(group)
        problem.setup()
        write_needed_inputs(problem, xml_file_path, VariableXmlStandardFormatter())
        raise Exception('Input .xml file not found, a default file has been created with default NaN values, '
                        'but no function is returned!\nConsider defining proper values before second execution!')

    elif os.path.exists(xml_file_path):

        reader = VariableIO(xml_file_path, VariableXmlStandardFormatter()).read(ignore=(var_inputs + outputs_names))
        xml_inputs = reader.names()
        if not(set(xml_inputs + var_inputs).intersection(set(inputs_names)) == set(inputs_names)):
            # If some inputs are missing write an error message and add them to the problem if authorized
            missing_inputs = list(
                set(inputs_names).difference(set(xml_inputs + var_inputs).intersection(set(inputs_names)))
            )
            message = 'The following inputs are missing in .xml file:'
            for item in missing_inputs:
                message += ' [' + item + '],'
            message = message[:-1] + '.\n'
            if overwrite:
                reader.path_separator = ":"
                ivc = reader.to_ivc()
                group = AutoUnitsDefaultGroup()
                group.add_subsystem('system', system, promotes=["*"])
                group.add_subsystem('ivc', ivc, promotes=["*"])
                problem = FASTOADProblem(group)
                problem.input_file_path = xml_file_path
                problem.output_file_path = xml_file_path
                problem.setup()
                problem.write_outputs()
                message += 'Default values have been added to {} file. ' \
                           'Consider modifying them for a second run!'.format(xml_file_path)
                raise Exception(message)
            else:
                raise Exception(message)
        else:
            # If all inputs addressed either by .xml or var_inputs, construct the function
            def patched_function(inputs_dict: dict) -> dict:
                """
                The patched function perform a run of an openmdao component or group applying FASTOAD formalism.

                @param inputs_dict: dictionary of input (values, units) saved with their key name,
                as an example: inputs_dict = {'in1': (3.0, "m")}.
                @return: dictionary of the component/group outputs saving names as keys and (value, units) as tuple.
                """


                # Read .xml file and construct Independent Variable Component excluding outputs
                reader.path_separator = ":"
                ivc_local = reader.to_ivc()
                for name, value in inputs_dict.items():
                    ivc_local.add_output(name, value[0], units=value[1])
                group_local = AutoUnitsDefaultGroup()
                group_local.add_subsystem('system', system, promotes=["*"])
                group_local.add_subsystem('ivc', ivc_local, promotes=["*"])
                problem_local = FASTOADProblem(group_local)
                problem_local.setup()
                problem_local.run_model()
                if overwrite:
                    problem_local.output_file_path = xml_file_path
                    problem_local.write_outputs()
                # Get output names from component/group and construct dictionary
                outputs_units = [var.units for var in variables if not var.is_input]
                outputs_dict = {}
                for idx in range(len(outputs_names)):
                    value = problem_local.get_val(outputs_names[idx], outputs_units[idx])
                    outputs_dict[outputs_names[idx]] = (value, outputs_units[idx])
                return outputs_dict
            return patched_function


class VariableListLocal(VariableList):

    @classmethod
    def from_system(cls, system: System) -> "VariableList":
        """
        Creates a VariableList instance containing inputs and outputs of a an OpenMDAO System.
        The inputs (is_input=True) correspond to the variables of IndepVarComp
        components and all the unconnected variables.

        Warning: setup() must NOT have been called.

        In the case of a group, if variables are promoted, the promoted name
        will be used. Otherwise, the absolute name will be used.

        :param system: OpenMDAO Component instance to inspect
        :return: VariableList instance
        """

        problem = om.Problem()
        if isinstance(system, om.Group):
            problem.model = deepcopy(system)
        else:
            # problem.model has to be a group
            problem.model.add_subsystem("comp", deepcopy(system), promotes=["*"])
        problem.setup()
        return VariableListLocal.from_problem(problem, use_initial_values=True)


def list_variables(component: Union[om.ExplicitComponent, om.Group]) -> list:
    """ Reads all variables from a component/problem and return as a list """
    if isinstance(component, om.Group):
        new_component = AutoUnitsDefaultGroup()
        new_component.add_subsystem("system", component, promotes=['*'])
        component = new_component
    variables = VariableListLocal.from_system(component)

    return variables