"""API."""
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

import logging
import pathlib
import shutil
import tempfile
from copy import deepcopy
from tempfile import TemporaryDirectory

import fastoad.api as oad
import numpy as np
import openmdao.api as om
from deprecated import deprecated

# noinspection PyProtectedMember
from fastoad.io import IVariableIOFormatter, VariableIO
from fastoad.io.xml import VariableXmlStandardFormatter
from fastoad.openmdao.problem import AutoUnitsDefaultGroup
from openmdao.core.explicitcomponent import ExplicitComponent
from openmdao.core.group import Group
from openmdao.core.implicitcomponent import ImplicitComponent
from openmdao.core.indepvarcomp import IndepVarComp
from openmdao.core.system import System

_LOGGER = logging.getLogger(__name__)

SAMPLE_FILENAME = "fastga.yml"
SAMPLE_XML_NAME = "fastga.xml"
BOOLEAN_OPTIONS = [
    "use_openvsp",
    "compute_mach_interpolation",
    "compute_slipstream",
    "low_speed_aero",
]


def _create_tmp_directory() -> TemporaryDirectory:
    """Provide temporary directory."""
    for tmp_base_path in [None, pathlib.Path.home() / ".fast"]:
        if tmp_base_path is not None:
            tmp_base_path.mkdir(parents=True)
        tmp_directory = tempfile.TemporaryDirectory(prefix="x", dir=tmp_base_path)
        break

    return tmp_directory


def file_temporary_transfer(file_path: pathlib.Path):
    """
    Put a copy of original python file into temporary directory and remove plugin registration
    from current file.
    """
    tmp_folder = _create_tmp_directory()
    file_name = file_path.name
    shutil.copy(file_path, tmp_folder.name / file_name)

    with file_path.open(encoding="utf-8") as file:
        lines = file.read()
        lines = lines.split("\n")
        idx_to_remove = []
        for idx, _ in enumerate(lines):
            if "@oad.RegisterOpenMDAOSystem" in lines[idx]:
                idx_to_remove.append(idx)
        for idx in sorted(idx_to_remove, reverse=True):
            del lines[idx]

    with file_path.open(encoding="utf-8") as file:
        for line in lines:
            file.write(line + "\n")

    return tmp_folder


def retrieve_original_file(tmp_folder, file_path: pathlib.Path):
    """Retrieve the original file."""
    file_name = file_path.name
    shutil.copy(tmp_folder.name / file_name, file_path)

    tmp_folder.cleanup()


@deprecated(
    version="0.2.0",
    reason="Will be removed in version 1.0. Please use the generate_configuration_file from "
    "fast-oad-core api with the distribution_name='fast-oad-cs23' and sample_file_name="
    + SAMPLE_FILENAME
    + " instead",
)
def generate_configuration_file(configuration_file_path: pathlib.Path, *, overwrite: bool = False):
    """
    Generates a sample configuration file.

    :param configuration_file_path: the path of the file to be written
    :param overwrite: if True, the file will be written, even if it already exists
    :raise FastPathExistsError: if overwrite==False and configuration_file_path already exists
    """
    oad.generate_configuration_file(
        configuration_file_path, overwrite, "fast-oad-cs23", SAMPLE_FILENAME
    )


@deprecated(
    version="1.3.2",
    reason="Will be removed in version 1.0. Please use the generate_source_data_file from "
    "fast-oad-core api with the distribution_name='fast-oad-cs23' and sample_file_name="
    + SAMPLE_FILENAME
    + " instead",
)
def generate_xml_file(xml_file_path: pathlib.Path, *, overwrite: bool = False):
    """
    Generates a sample XML file.

    :param xml_file_path: the path of the file to be written
    :param overwrite: if True, the file will be written, even if it already exists
    :raise FastPathExistsError: if overwrite==False and configuration_file_path already exists
    """
    oad.generate_source_data_file(xml_file_path, overwrite, "fast-oad-cs23", SAMPLE_FILENAME)


def write_needed_inputs(
    problem: oad.FASTOADProblem,
    xml_file_path: pathlib.Path,
    source_formatter: IVariableIOFormatter = None,
):
    """
    Writes the input file of the problem with unconnected inputs of the configured problem.
    Written value of each variable will be taken:
        1. from input_data if it contains the variable
        2. from defined default values in component definitions
    :param problem: problem we want to write the inputs of
    :param xml_file_path: if provided, variable values will be read from it
    :param source_formatter: the class that defines format of input file. if
                             not provided, expected format will be the default one.
    """
    variables = oad.DataFile(xml_file_path)

    unconnected_inputs = oad.VariableList.from_problem(
        problem,
        use_initial_values=True,
        get_promoted_names=True,
        promoted_only=True,
        io_status="inputs",
    )

    variables.update(
        unconnected_inputs,
        add_variables=True,
    )
    if xml_file_path:
        ref_vars = oad.DataFile(xml_file_path, source_formatter)
        variables.update(ref_vars)
        for var in variables:
            var.is_input = True
    variables.save()


def list_ivc_outputs_name(local_system: ExplicitComponent | ImplicitComponent | Group):
    """
    List all "root" components in the systems, meaning the components that don't have any
    subcomponents.
    """
    group = AutoUnitsDefaultGroup()
    group.add_subsystem("system", local_system, promotes=["*"])
    problem = oad.FASTOADProblem()
    problem.model = group
    try:
        problem.setup()
    except RuntimeError:
        _LOGGER.info(
            "Some problem occurred while setting-up the problem without input file probably "
            "because shape_by_conn variables exist!"
        )
    model = problem.model
    dict_sub_system = {}
    dict_sub_system = list_all_subsystem(model, "model", dict_sub_system)
    ivc_outputs_names = []

    # Find the outputs of all of those systems that are IndepVarComp
    for sub_system_keys in dict_sub_system:
        if (
            dict_sub_system[sub_system_keys] == "IndepVarComp"
            and sub_system_keys.split(".")[-1] != "fastoad_shaper"
        ):
            actual_attribute_name = sub_system_keys.replace("model.system.", "")
            address_levels = actual_attribute_name.split(".")
            component = model.system
            for next_level in address_levels:
                component = getattr(component, next_level)
            component_output = component.list_outputs()
            for outputs in component_output:
                ivc_outputs_names.append(outputs[0])

    return ivc_outputs_names


def generate_block_analysis(  # noqa: PLR0915, function is inherently complex and should be reworked
    local_system: ExplicitComponent | ImplicitComponent | Group | str | pathlib.Path,
    var_inputs: list,
    xml_file_path: pathlib.Path,
    options: dict | None = None,
    *,
    overwrite: bool = False,
):
    """
    Generates a function based on set of models and a set of variables that we want this
    functions to take.

    :param local_system: the system the function is going to be based one, can be either an
    OpenMDAO component (Implicit, Explicit or a Group), a registered FAST-OAD id, or the absolute
    path to a configuration file
    :param var_inputs: a list of variables name that we want the
    patched function to have as an input, it will be the list of keys expected as an input of the
    function
    :param xml_file_path: the path of the XML that contains the values of the variables
    necessary for the models but that are not inputs
    :param options: the options of the group, required if an id is provided
    :param overwrite: boolean to set whether or not the input XML file will be overwritten once
    the function runs

    :return patched_function: the function constructed based on the provided system which takes
    var_inputs as inputs under the form of a dictionary {"var_name": (var_value, var_units)}
    """
    xml_file_path = pathlib.Path(xml_file_path)

    # If a valid ID or a path to a configuration file is provided, build a system based on that ID
    if isinstance(local_system, pathlib.Path):
        configurator = oad.FASTOADProblemConfigurator(local_system)
        dummy_problem = configurator.get_problem(read_inputs=False)
        local_system = dummy_problem.model
    elif isinstance(local_system, str):
        local_system = oad.RegisterOpenMDAOSystem.get_system(local_system, options=options)

    # Search what are the component/group outputs
    variables = list_variables(local_system)
    inputs_names = [var.name for var in variables if var.is_input]
    outputs_names = [var.name for var in variables if not var.is_input]

    # Check the sub-systems of the local_system in question, and if there are ivc, list the
    # outputs  of those ivc. We are gonna assume that ivc are only use in a situation similar to
    # the one for the ComputePropellerPerformance group, meaning if there is an ivc,
    # it will always start the group

    ivc_outputs_names = list_ivc_outputs_name(local_system)

    # Check that variable inputs are in the group/component list
    if not (set(var_inputs) == set(inputs_names).intersection(set(var_inputs))):
        # TODO: Shouldn't be raising bare exceptions
        raise Exception("The input list contains name(s) out of component/group input list!")

    # Perform some tests on the .xml availability and completeness
    if not xml_file_path.exists() and set(var_inputs) != set(inputs_names):
        # If no input file and some inputs are missing, generate it and return None
        group = AutoUnitsDefaultGroup()
        group.add_subsystem("system", local_system, promotes=["*"])
        problem = oad.FASTOADProblem()
        problem.model = group
        problem.setup()
        write_needed_inputs(problem, xml_file_path, VariableXmlStandardFormatter())
        # TODO: Shouldn't be raising bare exceptions
        raise Exception(
            "Input .xml file not found, a default file has been created with default NaN values, "
            "but no function is returned!\nConsider defining proper values before second execution!"
        )

    if xml_file_path.exists():
        reader = VariableIO(xml_file_path, VariableXmlStandardFormatter()).read(
            ignore=(var_inputs + outputs_names + ivc_outputs_names)
        )
        xml_inputs = reader.names()
    else:
        xml_inputs = []
    if set(xml_inputs + var_inputs + ivc_outputs_names).intersection(set(inputs_names)) != set(
        inputs_names
    ):
        # If some inputs are missing write an error message and add them to the problem if
        # authorized
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
            group.add_subsystem("system", local_system, promotes=["*"])
            group.add_subsystem("ivc", ivc, promotes=["*"])
            problem = oad.FASTOADProblem()
            problem.model = group
            problem.input_file_path = xml_file_path
            problem.output_file_path = xml_file_path
            problem.setup()
            problem.write_outputs()
            message += (
                f"Default values have been added to {xml_file_path} file. "
                "Consider modifying them for a second run!"
            )
            # TODO: Shouldn't be raising bare exceptions
            raise Exception(message)
        # TODO: Shouldn't be raising bare exceptions
        raise Exception(message)

    # If all inputs addressed either by .xml or var_inputs or in an IVC, construct the
    # function
    def patched_function(inputs_dict: dict) -> dict:
        """
        The patched function perform a run of an openmdao component or group applying
        FASTOAD formalism.

        @param inputs_dict: dictionary of input (values, units) saved with their key name,
        as an example: inputs_dict = {'in1': (3.0, "m")}.
        @return: dictionary of the component/group outputs saving names as keys and (value,
        units) as tuple.
        """

        # Read .xml file and construct Independent Variable Component excluding outputs
        if xml_file_path.exists():
            reader.path_separator = ":"
            ivc_local = reader.to_ivc()
        else:
            ivc_local = IndepVarComp()
        for name, value in inputs_dict.items():
            ivc_local.add_output(name, value[0], units=value[1])
        group_local = AutoUnitsDefaultGroup()
        group_local.add_subsystem("ivc", ivc_local, promotes=["*"])
        group_local.add_subsystem("system", local_system, promotes=["*"])
        problem_local = oad.FASTOADProblem()
        model_local = problem_local.model
        model_local.add_subsystem("local_system", group_local, promotes=["*"])
        problem_local.setup()
        problem_local.run_model()
        if overwrite:
            problem_local.output_file_path = xml_file_path
            problem_local.write_outputs()
        # Get output names from component/group and construct dictionary
        outputs_units = [var.units for var in variables if not var.is_input]
        outputs_dict = {}
        for idx, _ in enumerate(outputs_names):
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
    except AttributeError:
        dict_subsystems[model_address] = get_type(model)

    return dict_subsystems


def get_type(model):
    raw_type = model.msginfo.split("<")[-1]
    type_alone = raw_type.split(" ")[-1]
    return type_alone[:-1]


class VariableListLocal(oad.VariableList):
    @classmethod
    def from_system(cls, local_system: System) -> "oad.VariableList":
        """
        Creates a VariableList instance containing inputs and outputs of a an OpenMDAO System.
        The inputs (is_input=True) correspond to the variables of IndepVarComp
        components and all the unconnected variables.

        Warning: setup() must NOT have been called.

        In the case of a group, if variables are promoted, the promoted name
        will be used. Otherwise, the absolute name will be used.

        :param local_system: OpenMDAO Component instance to inspect
        :return: VariableList instance.
        """

        problem = oad.FASTOADProblem()
        if isinstance(local_system, om.Group):
            problem.model = deepcopy(local_system)
        else:
            # problem.model has to be a group
            problem.model.add_subsystem("comp", deepcopy(local_system), promotes=["*"])
        try:
            problem.setup()
        except RuntimeError:
            _LOGGER.info(
                "Some problem occurred while setting-up the problem without input file probably "
                "because shape_by_conn variables exist!"
            )
        return VariableListLocal.from_problem(problem, use_initial_values=True)


def list_variables(component: om.ExplicitComponent | om.Group) -> list:
    """Reads all variables from a component/problem and return as a list."""
    if isinstance(component, om.Group):
        new_component = AutoUnitsDefaultGroup()
        new_component.add_subsystem("system", component, promotes=["*"])
        component = new_component
    return VariableListLocal.from_system(component)


def list_inputs(component: om.ExplicitComponent | om.Group) -> list:
    """Reads all variables from a component/problem and returns inputs as a list."""
    variables = list_variables(component)
    return [var.name for var in variables if var.is_input]


def list_inputs_metadata(component: om.ExplicitComponent | om.Group) -> tuple:
    """
    Reads all variables from a component/problem and returns inputs name and metadata as a
    list.
    """

    prob = oad.FASTOADProblem()
    model = prob.model
    model.add_subsystem("component", component, promotes=["*"])

    prob_copy = deepcopy(prob)

    var_copy_shape_name_list = []
    var_copy_shape_list = []

    try:
        prob_copy.setup()
    except RuntimeError:
        # noinspection PyProtectedMember
        vars_metadata = oad.FASTOADProblem()._get_undetermined_dynamic_vars_metadata(prob_copy)
        if vars_metadata:
            # If vars_metadata is empty, it means the RuntimeError was not because
            # of dynamic shapes, and the incoming self.setup() will raise it.
            ivc = om.IndepVarComp()
            for name, meta in vars_metadata.items():
                # We use a (2,)-shaped array as value here. This way, it will be easier to
                # identify dynamic-shaped data in an input file generated from current problem.
                var_copy_shape_name_list.append(name)
                var_copy_shape_list.append(meta["copy_shape"])
                ivc.add_output(name, [np.nan, np.nan], units=meta["units"])
            prob.model.add_subsystem("temp_shaper", ivc, promotes=["*"])

    variables = prob_copy.model.get_io_metadata("input")

    var_inputs = []
    var_units = []
    var_shape = []
    var_shape_by_conn = []
    var_copy_shape = []

    for variable_name in variables:
        variable = variables[variable_name]
        var_prom_name = variable["prom_name"]

        # We check that it has not been added already
        if var_prom_name not in var_inputs:
            var_inputs.append(variable["prom_name"])
            var_units.append(variable["units"])
            var_shape.append(variable["shape"])
            var_shape_by_conn.append(variable["shape_by_conn"])
            var_copy_shape.append(variable["copy_shape"])

    return var_inputs, var_units, var_shape, var_shape_by_conn, var_copy_shape


def list_outputs(component: om.ExplicitComponent | om.Group) -> list:
    """Reads all variables from a component/problem and returns outputs as a list."""
    variables = list_variables(component)
    return [var.name for var in variables if not var.is_input]


def string_to_array(arr):
    """
    Convert a numpy array that was stored as a string back to a numpy array.

    "[-20.0, -19.5, -19.0, ..., 19.0, 19.5, 20.0]" --> [-20.0, -19.5, -19.0, ..., 19.0, 19.5, 20.0]
    """
    return np.array(arr.strip("[]").split(","), dtype=float)
