import os.path as pth

from fastoad.io.configuration.configuration import FASTOADProblemConfigurator

from fastga.models.geometry.geom_components.nacelle.compute_nacelle import ComputeNacelleGeometry

DATA_FOLDER_PATH = pth.join(pth.dirname(__file__), "data")

from tests.testing_utilities import run_system, get_indep_var_comp, list_inputs


def test_mission():

    configurator = FASTOADProblemConfigurator(pth.join(DATA_FOLDER_PATH, "mission_run.yml"))

    ref_inputs = pth.join(DATA_FOLDER_PATH, "source_input.xml")
    configurator.write_needed_inputs(ref_inputs)

    problem = configurator.get_problem(read_inputs=True)
    problem.setup()
    problem.run_model()
    problem.write_outputs()


def _test_engine_alone():
    ivc = get_indep_var_comp(
        list_inputs(
            ComputeNacelleGeometry(propulsion_id="fastga.wrapper.propulsion.basicIC_engine")
        ),
        __file__,
        pth.join(DATA_FOLDER_PATH, "source_input.xml"),
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        ComputeNacelleGeometry(propulsion_id="fastga.wrapper.propulsion.basicIC_engine"), ivc
    )
