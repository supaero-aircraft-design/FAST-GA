import os.path as pth

from fastoad.io.configuration.configuration import FASTOADProblemConfigurator

DATA_FOLDER_PATH = pth.join(pth.dirname(__file__), "data")


def test_mission():

    configurator = FASTOADProblemConfigurator(pth.join(DATA_FOLDER_PATH, "mission_run.yml"))

    ref_inputs = pth.join(DATA_FOLDER_PATH, "source_input.xml")
    configurator.write_needed_inputs(ref_inputs)

    problem = configurator.get_problem(read_inputs=True)
    problem.setup()
    problem.run_model()
    problem.write_outputs()
