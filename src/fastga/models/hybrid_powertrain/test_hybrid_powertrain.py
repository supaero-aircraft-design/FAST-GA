from tests.testing_utilities import run_system, get_indep_var_comp, list_inputs
import pytest
from .components import (
    ComputeBatteries,
    ComputeBoP,
    ComputeCompressor,
    ComputeElectricMotor,
    ComputeFuelCells,
    ComputeH2Storage,
    ComputeHex,
    ComputeIntakes,
    ComputeInverter
)

XML_FILE = "hybrid_aircraft.xml"


def test_compute_fuel_cells():
    """ Tests computation of the fuel cell stacks """

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(ComputeFuelCells()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeFuelCells(), ivc)
    nb_cells = problem.get_val("data:geometry:hybrid_powertrain:fuel_cell:number_cells", units=None)
    assert nb_cells == pytest.approx(315, abs=5)
    ox_flow = problem.get_val("data:propulsion:hybrid_powertrain:fuel_cell:ox_mass_flow", units='kg/s')
    assert ox_flow == pytest.approx(5.74, abs=1e-1)
    des_power = problem.get_val("data:propulsion:hybrid_powertrain:fuel_cell:design_power", units='W')
    assert des_power == pytest.approx(22000, abs=1000)
    vol = problem.get_val("data:geometry:hybrid_powertrain:fuel_cell:stack_volume", units='m**3')
    assert vol == pytest.approx(0.02363032, abs=1e-3)
    eff = problem.get_val("data:propulsion:hybrid_powertrain:fuel_cell:efficiency", units=None)
    assert eff == pytest.approx(0.5, abs=1e-1)
    P_cooling = problem.get_val("data:propulsion:hybrid_powertrain:fuel_cell:cooling_power", units='W')
    assert P_cooling == pytest.approx(21036, abs=1000)


def test_compute_compressor():
    """ Tests computation of the compressor """

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(ComputeCompressor()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeCompressor(), ivc)
    radius = problem.get_val("data:geometry:hybrid_powertrain:compressor:radius", units='m')
    assert radius == pytest.approx(0.029, abs=1e-3)
    power = problem.get_val("data:propulsion:hybrid_powertrain:compressor:power", units='kW')
    assert power == pytest.approx(2, abs=1e-1)


def test_compute_bop():
    """ Tests computation of the Balance of Plant """

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(ComputeBoP()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeBoP(), ivc)
    L = problem.get_val("data:geometry:hybrid_powertrain:bop:extra_length", units='m')
    assert L == pytest.approx(0.1764, abs=1e-3)
    l = problem.get_val("data:geometry:hybrid_powertrain:bop:extra_width", units='m')
    assert l == pytest.approx(0.3069, abs=1e-3)
    h = problem.get_val("data:geometry:hybrid_powertrain:bop:extra_height", units='m')
    assert h == pytest.approx(0.2336, abs=1e-3)


def test_compute_hex():
    """ Tests computation of the heat exchanger """

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(ComputeHex()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeHex(), ivc)
    area = problem.get_val("data:geometry:hybrid_powertrain:hex:area", units='m**2')
    assert area == pytest.approx(0.01832894, abs=1e-3)


def test_compute_intakes():
    """ Tests computation of the intakes """

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(ComputeIntakes()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeIntakes(), ivc)
    fc_intake_L = problem.get_val("data:geometry:hybrid_powertrain:fc_intake:length", units='m')
    assert fc_intake_L == pytest.approx(1.07, abs=1e-1)
    fc_intake_l = problem.get_val("data:geometry:hybrid_powertrain:fc_intake:width", units='m')
    assert fc_intake_l == pytest.approx(0.56, abs=1e-1)
    fc_intake_d = problem.get_val("data:geometry:hybrid_powertrain:fc_intake:depth", units='m')
    assert fc_intake_d == pytest.approx(0.141, abs=1e-1)
    cooling_intake_L = problem.get_val("data:geometry:hybrid_powertrain:cooling_intake:length", units='m')
    assert cooling_intake_L == pytest.approx(0.321, abs=1e-1)
    cooling_intake_l = problem.get_val("data:geometry:hybrid_powertrain:cooling_intake:width", units='m')
    assert cooling_intake_l == pytest.approx(0.168, abs=1e-1)
    cooling_intake_d = problem.get_val("data:geometry:hybrid_powertrain:cooling_intake:depth", units='m')
    assert cooling_intake_d == pytest.approx(0.042, abs=1e-1)
    cd0 = problem.get_val("data:aerodynamics:intakes:CD0")
    assert cd0 == pytest.approx(0.43279067, abs=1e-1)


def test_compute_battery():
    """ Tests computation of the batteries """

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(ComputeBatteries()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeBatteries(), ivc)
    N_ser = problem.get_val("data:geometry:hybrid_powertrain:battery:N_series", units=None)
    assert N_ser == pytest.approx(181, abs=3)
    N_par = problem.get_val("data:geometry:hybrid_powertrain:battery:N_parallel", units=None)
    assert N_par == pytest.approx(15, abs=3)
    vol = problem.get_val("data:geometry:hybrid_powertrain:battery:pack_volume", units='m**3')
    assert vol == pytest.approx(0.08252006, abs=1e-3)
    tot_vol = problem.get_val("data:geometry:hybrid_powertrain:battery:tot_volume", units='m**3')
    assert tot_vol == pytest.approx(0.16, abs=1e-2)


def test_compute_inverter():
    """ Tests computation of the inverter """

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(ComputeInverter()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeInverter(), ivc)
    power = problem.get_val("data:propulsion:hybrid_powertrain:inverter:output_power", units='kW')
    assert power == pytest.approx(42, abs=2)
    vol = problem.get_val("data:geometry:hybrid_powertrain:inverter:volume", units='m**3')
    assert vol == pytest.approx(0.00096657, abs=1e-3)


def test_compute_electric_motor():
    """ Tests computation of the electric motor """

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(ComputeElectricMotor()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeElectricMotor(), ivc)
    L = problem.get_val("data:geometry:hybrid_powertrain:motor:length", units='m')
    assert L == pytest.approx(0.091, abs=1e-3)
    d = problem.get_val("data:geometry:hybrid_powertrain:motor:diameter", units='m')
    assert d == pytest.approx(0.268, abs=1e-3)
    w = problem.get_val("data:propulsion:hybrid_powertrain:motor:speed", units='rpm')
    assert w == pytest.approx(2500, abs=100)
    m = problem.get_val("data:weight:hybrid_powertrain:motor:mass", units='kg')
    assert m == pytest.approx(22.7, abs=3)
    peak_torque = problem.get_val("data:propulsion:hybrid_powertrain:motor:peak_torque", units='N*m')
    assert peak_torque == pytest.approx(500, abs=10)


def test_h2_storage():
    """ Tests computation of the hydrogen storage """

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(ComputeH2Storage()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeH2Storage(), ivc)
    vol = problem.get_val("data:geometry:hybrid_powertrain:h2_storage:total_tanks_volume", units='m**3')
    assert vol == pytest.approx(0.53686679, abs=1e-3)
    R = problem.get_val("data:geometry:hybrid_powertrain:h2_storage:tank_internal_radius", units='m')
    assert R == pytest.approx(0.24199837, abs=1e-3)
    L = problem.get_val("data:geometry:hybrid_powertrain:h2_storage:tank_internal_length", units='m')
    assert L == pytest.approx(1.45199024, abs=1e-3)
    t = problem.get_val("data:geometry:hybrid_powertrain:h2_storage:wall_thickness", units='m')
    assert t == pytest.approx(0.00050151, abs=1e-6)
    mass = problem.get_val("data:weight:hybrid_powertrain:h2_storage:single_tank_mass", units='kg')
    assert mass == pytest.approx(1.979, abs=1e-3)
