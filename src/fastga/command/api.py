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
import shutil
import inspect
import importlib
from typing import Union, List
from openmdao.core.explicitcomponent import ExplicitComponent
from openmdao.core.implicitcomponent import ImplicitComponent
from openmdao.core.indepvarcomp import IndepVarComp
from openmdao.core.group import Group
from openmdao.core.system import System
import openmdao.api as om
from copy import deepcopy
from itertools import product
from tempfile import TemporaryDirectory
from pathlib import Path
import tempfile
import glob

from fastoad.openmdao.variables import VariableList
from fastoad.cmd.exceptions import FastFileExistsError
from fastoad.openmdao.problem import FASTOADProblem
from fastoad.io import DataFile, IVariableIOFormatter
from fastoad.io.xml import VariableXmlStandardFormatter
from fastoad.io import VariableIO
from fastoad.io.configuration.configuration import AutoUnitsDefaultGroup

# noinspection PyProtectedMember
from fastoad.cmd.api import _get_simple_system_list

from . import resources

_LOGGER = logging.getLogger(__name__)

SAMPLE_FILENAME = "fastga.yml"
BOOLEAN_OPTIONS = [
    "use_openvsp",
    "compute_mach_interpolation",
    "compute_slipstream",
    "low_speed_aero",
]


def _create_tmp_directory() -> TemporaryDirectory:
    """Provide temporary directory."""

    for tmp_base_path in [None, pth.join(str(Path.home()), ".fast")]:
        if tmp_base_path is not None:
            os.makedirs(tmp_base_path, exist_ok=True)
        tmp_directory = tempfile.TemporaryDirectory(prefix="x", dir=tmp_base_path)
        break

    return tmp_directory


def file_temporary_transfer(file_path: str):
    # Put a copy of original python file into temporary directory and remove plugin registration from current file

    tmp_folder = _create_tmp_directory()
    file_name = pth.split(file_path)[-1]
    shutil.copy(file_path, pth.join(tmp_folder.name, file_name))
    file = open(file_path, "r")
    lines = file.read()
    lines = lines.split("\n")
    idx_to_remove = []
    for idx in range(len(lines)):
        if "@RegisterOpenMDAOSystem" in lines[idx]:
            idx_to_remove.append(idx)
    for idx in sorted(idx_to_remove, reverse=True):
        del lines[idx]
    file.close()
    file = open(file_path, "w")
    for line in lines:
        file.write(line + "\n")
    file.close()

    return tmp_folder


def retrieve_original_file(tmp_folder, file_path: str):
    # Retrieve the original file

    file_name = pth.split(file_path)[-1]
    shutil.copy(pth.join(tmp_folder.name, file_name), file_path)

    tmp_folder.cleanup()


def generate_variables_description(subpackage_path: str, overwrite: bool = False):
    """
    Generates/append the variable descriptions file for a given subpackage.

    To use it simply type:
    from fastga.command.api import generate_variables_description
    import my_package

    generate_variables_description(my_package.__path__[0], overwrite=True)

    :param subpackage_path: the path of the subpackage to explore
    :param overwrite: if True, the file will be written, even if it already exists
    :raise FastFileExistsError: if overwrite==False and subpackage_path already exists
    """

    if not overwrite and pth.exists(pth.join(subpackage_path, "variable_descriptions.txt")):
        # noinspection PyStringFormat
        raise FastFileExistsError(
            "Variable descriptions file is not written because it already exists. "
            "Use overwrite=True to bypass."
            % pth.join(subpackage_path, "variable_descriptions.txt"),
            pth.join(subpackage_path, "variable_descriptions.txt"),
        )

    if not pth.exists(subpackage_path):
        _LOGGER.info("Sub-package path %s not found!", subpackage_path)
    else:
        # Read file and construct dictionary of variables name index
        saved_dict = {}
        if pth.exists(pth.join(subpackage_path, "variable_descriptions.txt")):
            file = open(pth.join(subpackage_path, "variable_descriptions.txt"), "r")
            for line in file:
                if line[0] != "#" and len(line.split("||")) == 2:
                    variable_name, variable_description = line.split("||")
                    variable_name_length = len(variable_name)
                    variable_name = variable_name.replace(" ", "")
                    while variable_name_length != len(variable_name):
                        variable_name = variable_name.replace(" ", "")
                        variable_name_length = len(variable_name)
                    saved_dict[variable_name] = variable_description
            file.close()

        # If path point to ./models directory list output variables described in the different models
        if subpackage_path.split("\\")[-1] == "models":
            for root, dirs, files in os.walk(subpackage_path, topdown=False):
                for name in files:
                    if name == "variable_descriptions.txt":
                        file = open(pth.join(root, name), "r")
                        for line in file:
                            if line[0] != "#" and len(line.split("||")) == 2:
                                variable_name, variable_description = line.split("||")
                                variable_name_length = len(variable_name)
                                variable_name = variable_name.replace(" ", "")
                                while variable_name_length != len(variable_name):
                                    variable_name = variable_name.replace(" ", "")
                                    variable_name_length = len(variable_name)
                                if variable_name not in saved_dict.keys():
                                    saved_dict[variable_name] = variable_description
                                else:
                                    print(
                                        "\nWarning: file contains parameter "
                                        + variable_name
                                        + " already saved in "
                                        + pth.split(root)[-1]
                                        + " subpackage !"
                                    )
                        file.close()

        # Explore subpackage models and find the output variables and store them in a dictionary
        dict_to_be_saved = {}
        for root, dirs, files in os.walk(subpackage_path, topdown=False):
            for name in files:
                if name[-3:] == ".py":
                    spec = importlib.util.spec_from_file_location(
                        name.replace(".py", ""), pth.join(root, name)
                    )
                    module = importlib.util.module_from_spec(spec)
                    tmp_folder = None
                    # noinspection PyBroadException
                    try:
                        # if register decorator in module, temporary replace file removing decorators
                        # noinspection PyBroadException
                        try:
                            spec.loader.exec_module(module)
                        except:
                            pass
                        if "RegisterOpenMDAOSystem" in dir(module):
                            tmp_folder = file_temporary_transfer(pth.join(root, name))
                        spec.loader.exec_module(module)
                        class_list = [x for x in dir(module) if inspect.isclass(getattr(module, x))]
                        # noinspection PyUnboundLocalVariable
                        retrieve_original_file(tmp_folder, pth.join(root, name))
                        root_lib = ".".join(root.split("\\")[root.split("\\").index("fastga") :])
                        root_lib += "." + name.replace(".py", "")
                        for class_name in class_list:
                            # noinspection PyBroadException
                            try:
                                exec("from " + root_lib + " import " + class_name + " as my_class")
                                options = eval("my_class().options")
                                if "propulsion_id" in options:
                                    available_id_list = _get_simple_system_list()
                                    idx_to_remove = []
                                    for idx in range(len(available_id_list)):
                                        available_id_list[idx] = available_id_list[idx][0]
                                        if "PROPULSION" in available_id_list[idx]:
                                            idx_to_remove.extend(list(range(idx + 1)))
                                        if not ("fastga" in available_id_list[idx]):
                                            idx_to_remove.append(idx)
                                    idx_to_remove = list(dict.fromkeys(idx_to_remove))
                                    for idx in sorted(idx_to_remove, reverse=True):
                                        del available_id_list[idx]
                                    exec_base = (
                                        'my_class(propulsion_id="' + available_id_list[0] + '"'
                                    )
                                else:
                                    exec_base = "my_class("
                                exec("variables = list_variables(" + exec_base + "))")
                                local_options = []
                                for option_name in BOOLEAN_OPTIONS:
                                    # noinspection PyProtectedMember
                                    if option_name in options._dict.keys():
                                        local_options.append(option_name)
                                if len(local_options) == 0:
                                    if subpackage_path.split("\\")[-1] == "models":
                                        var_names = eval(
                                            "[var.name for var in variables if var.is_input]"
                                        )
                                    else:
                                        var_names = eval(
                                            "[var.name for var in variables if not var.is_input]"
                                        )
                                        if (
                                            len(eval("list_ivc_outputs_name(" + exec_base + "))"))
                                            != 0
                                        ):
                                            var_names.append(
                                                eval("list_ivc_outputs_name(" + exec_base + "))")
                                            )
                                    var_names = list(dict.fromkeys(var_names))
                                    for key in var_names:
                                        if (
                                            ("data:" in key)
                                            or ("settings:" in key)
                                            or ("tuning:" in key)
                                        ):
                                            if key not in dict_to_be_saved.keys():
                                                dict_to_be_saved[key] = ""
                                else:
                                    for options_tuple in list(
                                        product([True, False], repeat=len(local_options))
                                    ):
                                        if "=" in exec_base:
                                            exec_line = (
                                                "variables = list_variables(" + exec_base + ","
                                            )
                                        else:
                                            exec_line = "variables = list_variables(" + exec_base
                                        for idx in range(len(local_options)):
                                            exec_line += (
                                                local_options[idx]
                                                + "="
                                                + str(options_tuple[idx])
                                                + ","
                                            )
                                        exec_line = exec_line[0:-1] + "))"
                                        exec(exec_line)
                                        if subpackage_path.split("\\")[-1] == "models":
                                            var_names = eval(
                                                "[var.name for var in variables if var.is_input]"
                                            )
                                        else:
                                            var_names = eval(
                                                "[var.name for var in variables if not var.is_input]"
                                            )
                                            if (
                                                len(
                                                    eval(
                                                        exec_line.replace(
                                                            "variables = list_variables",
                                                            "list_ivc_outputs_name",
                                                        )
                                                    )
                                                )
                                                != 0
                                            ):
                                                var_names.append(
                                                    eval(
                                                        exec_line.replace(
                                                            "variables = list_variables",
                                                            "list_ivc_outputs_name",
                                                        )
                                                    )
                                                )
                                        var_names = list(dict.fromkeys(var_names))
                                        for key in var_names:
                                            if (
                                                ("data:" in key)
                                                or ("settings:" in key)
                                                or ("tuning:" in key)
                                            ):
                                                if key not in dict_to_be_saved.keys():
                                                    dict_to_be_saved[key] = ""
                            except:
                                pass
                    except:
                        if not (tmp_folder is None):
                            # noinspection PyUnboundLocalVariable
                            retrieve_original_file(tmp_folder, pth.join(root, name))

        # Complete the variable descriptions file with missing outputs
        if pth.exists(pth.join(subpackage_path, "variable_descriptions.txt")):
            file = open(pth.join(subpackage_path, "variable_descriptions.txt"), "a")
            if len(
                set(list(dict_to_be_saved.keys())).intersection(set(list(saved_dict.keys())))
            ) != len(dict_to_be_saved.keys()):
                file.write("\n")
        else:
            if len(dict_to_be_saved.keys()) != 0:
                file = open(pth.join(subpackage_path, "variable_descriptions.txt"), "w")
                file.write("# Documentation of variables used in FAST-GA models\n")
                file.write("# Each line should be like:\n")
                file.write(
                    "# my:variable||The description of my:variable, as long as needed, but on one line.\n"
                )
                file.write(
                    '# The separator "||" can be surrounded with spaces (that will be ignored)\n\n'
                )
        if len(dict_to_be_saved.keys()) != 0:
            sorted_keys = sorted(dict_to_be_saved.keys(), key=lambda x: x.lower())
            for key in sorted_keys:
                if not (key in saved_dict.keys()):
                    # noinspection PyUnboundLocalVariable
                    file.write(key + " || \n")
            file.close()


def generate_configuration_file(configuration_file_path: str, overwrite: bool = False):
    """
    Generates a sample configuration file.

    :param configuration_file_path: the path of the file to be written
    :param overwrite: if True, the file will be written, even if it already exists
    :raise FastFileExistsError: if overwrite==False and configuration_file_path already exists
    """
    if not overwrite and pth.exists(configuration_file_path):
        raise FastFileExistsError(
            "Configuration file is not written because it already exists. "
            "Use overwrite=True to bypass." % configuration_file_path,
            configuration_file_path,
        )

    if not pth.exists(pth.split(configuration_file_path)[0]):
        os.mkdir(pth.split(configuration_file_path)[0])
    shutil.copy(pth.join(resources.__path__[0], SAMPLE_FILENAME), configuration_file_path)

    _LOGGER.info("Sample configuration written in %s", configuration_file_path)


def write_needed_inputs(
    problem: FASTOADProblem, xml_file_path: str, source_formatter: IVariableIOFormatter = None
):
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


def list_ivc_outputs_name(system: Union[ExplicitComponent, ImplicitComponent, Group],):
    # List all "root" components in the systems, meaning the components that don't have any subcomponents
    group = AutoUnitsDefaultGroup()
    group.add_subsystem("system", system, promotes=["*"])
    problem = FASTOADProblem(group)
    problem.setup()
    model = problem.model
    dict_sub_system = {}
    dict_sub_system = list_all_subsystem(model, "model", dict_sub_system)
    ivc_outputs_names = []

    # Find the outputs of all of those systems that are IndepVarComp
    for sub_system_keys in dict_sub_system.keys():
        if dict_sub_system[sub_system_keys] == "IndepVarComp":
            actual_attribute_name = sub_system_keys.replace("model.system.", "")
            address_levels = actual_attribute_name.split(".")
            component = model.system
            for next_level in address_levels:
                component = getattr(component, next_level)
            component_output = component.list_outputs()
            for outputs in component_output:
                ivc_outputs_names.append(outputs[0])

    return ivc_outputs_names


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

    # Check the sub-systems of the system in question, and if there are ivc, list the outputs  of those ivc. We are
    # gonna assume that ivc are only use in a situation similar to the one for the ComputePropellerPerformance group,
    # meaning if there is an ivc, it will always start the group

    ivc_outputs_names = list_ivc_outputs_name(system)

    # Check that variable inputs are in the group/component list
    if not (set(var_inputs) == set(inputs_names).intersection(set(var_inputs))):
        raise Exception("The input list contains name(s) out of component/group input list!")

    # Perform some tests on the .xml availability and completeness
    if not (os.path.exists(xml_file_path)) and not (set(var_inputs) == set(inputs_names)):
        # If no input file and some inputs are missing, generate it and return None
        group = AutoUnitsDefaultGroup()
        group.add_subsystem("system", system, promotes=["*"])
        problem = FASTOADProblem(group)
        problem.setup()
        write_needed_inputs(problem, xml_file_path, VariableXmlStandardFormatter())
        raise Exception(
            "Input .xml file not found, a default file has been created with default NaN values, "
            "but no function is returned!\nConsider defining proper values before second execution!"
        )

    else:

        if os.path.exists(xml_file_path):
            reader = VariableIO(xml_file_path, VariableXmlStandardFormatter()).read(
                ignore=(var_inputs + outputs_names + ivc_outputs_names)
            )
            xml_inputs = reader.names()
        else:
            xml_inputs = []
        if not (
            set(xml_inputs + var_inputs + ivc_outputs_names).intersection(set(inputs_names))
            == set(inputs_names)
        ):
            # If some inputs are missing write an error message and add them to the problem if authorized
            missing_inputs = list(
                set(inputs_names).difference(
                    set(xml_inputs + var_inputs + ivc_outputs_names).intersection(set(inputs_names))
                )
            )
            message = "The following inputs are missing in .xml file:"
            for item in missing_inputs:
                message += " [" + item + "],"
            message = message[:-1] + ".\n"
            if overwrite:
                # noinspection PyUnboundLocalVariable
                reader.path_separator = ":"
                ivc = reader.to_ivc()
                group = AutoUnitsDefaultGroup()
                group.add_subsystem("system", system, promotes=["*"])
                group.add_subsystem("ivc", ivc, promotes=["*"])
                problem = FASTOADProblem(group)
                problem.input_file_path = xml_file_path
                problem.output_file_path = xml_file_path
                problem.setup()
                problem.write_outputs()
                message += (
                    "Default values have been added to {} file. "
                    "Consider modifying them for a second run!".format(xml_file_path)
                )
                raise Exception(message)
            else:
                raise Exception(message)
        else:
            # If all inputs addressed either by .xml or var_inputs or in an IVC, construct the function
            def patched_function(inputs_dict: dict) -> dict:
                """
                The patched function perform a run of an openmdao component or group applying FASTOAD formalism.

                @param inputs_dict: dictionary of input (values, units) saved with their key name,
                as an example: inputs_dict = {'in1': (3.0, "m")}.
                @return: dictionary of the component/group outputs saving names as keys and (value, units) as tuple.
                """

                # Read .xml file and construct Independent Variable Component excluding outputs
                if os.path.exists(xml_file_path):
                    reader.path_separator = ":"
                    ivc_local = reader.to_ivc()
                else:
                    ivc_local = IndepVarComp()
                for name, value in inputs_dict.items():
                    ivc_local.add_output(name, value[0], units=value[1])
                group_local = AutoUnitsDefaultGroup()
                group_local.add_subsystem("system", system, promotes=["*"])
                group_local.add_subsystem("ivc", ivc_local, promotes=["*"])
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


def list_all_subsystem(model, model_address, dict_subsystems):
    # noinspection PyBroadException
    try:
        # noinspection PyProtectedMember
        subsystem_list = model._proc_info.keys()
        for subsystem in subsystem_list:
            dict_subsystems = list_all_subsystem(
                getattr(model, subsystem), model_address + "." + subsystem, dict_subsystems
            )
    except:
        dict_subsystems[model_address] = get_type(model)

    return dict_subsystems


def get_type(model):
    raw_type = model.msginfo.split("<")[-1]
    type_alone = raw_type.split(" ")[-1]
    model_type = type_alone[:-1]

    return model_type


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
        new_component.add_subsystem("system", component, promotes=["*"])
        component = new_component
    variables = VariableListLocal.from_system(component)

    return variables
