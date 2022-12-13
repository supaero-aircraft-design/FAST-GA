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

from fastga.models.weight_fuel_cell.mass_breakdown.b_propulsion.b12_h2_storage_weight import (
    ComputeH2StorageWeightPhysical,
    ComputeH2StorageWeightLegacy
)

from fastga.models.geometry.geom_components.nacelle.compute_nacelle_dimension import ComputeNacelleDimension

XML_FILE = "hybrid_aircraft.xml"
ENGINE_WRAPPER = "fastga.wrapper.propulsion.basicHE_engine"


def test_compute_fuel_cells():
    """ Tests computation of the fuel cell stacks """

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(ComputeFuelCells()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeFuelCells(), ivc)
    nb_cells = problem.get_val("data:geometry:hybrid_powertrain:fuel_cell:number_cells", units=None)
    assert nb_cells == pytest.approx(329, abs=1)
    cell_area = problem.get_val("data:geometry:hybrid_powertrain:fuel_cell:stack_area", units='cm**2')
    assert cell_area == pytest.approx(759.5, abs=1e-1)
    ox_flow = problem.get_val("data:propulsion:hybrid_powertrain:fuel_cell:ox_mass_flow", units='kg/s')
    assert ox_flow == pytest.approx(0.006, abs=1e-5)
    hy_flow = problem.get_val("data:propulsion:hybrid_powertrain:fuel_cell:hyd_mass_flow", units='kg/s')
    assert hy_flow == pytest.approx(0.00075, abs=1e-5)
    stack_power = problem.get_val("data:propulsion:hybrid_powertrain:fuel_cell:stack_power", units='W')
    assert stack_power == pytest.approx(22000, abs=1)
    out_power = problem.get_val("data:propulsion:hybrid_powertrain:fuel_cell:output_power", units='W')
    assert out_power == pytest.approx(40000, abs=1)
    vol = problem.get_val("data:geometry:hybrid_powertrain:fuel_cell:stack_volume", units='m**3')
    assert vol == pytest.approx(0.0416, abs=1e-3)
    eff = problem.get_val("data:propulsion:hybrid_powertrain:fuel_cell:efficiency", units=None)
    assert eff == pytest.approx(0.474, abs=1e-3)
    P_cooling = problem.get_val("data:propulsion:hybrid_powertrain:fuel_cell:cooling_power", units='W')
    assert P_cooling == pytest.approx(22413, abs=1)


def test_compute_compressor():
    """ Tests computation of the compressor """

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(ComputeCompressor()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeCompressor(), ivc)
    radius = problem.get_val("data:geometry:hybrid_powertrain:compressor:radius", units='m')
    assert radius == pytest.approx(0.0508, abs=1e-3)
    power = problem.get_val("data:propulsion:hybrid_powertrain:compressor:power", units='W')
    assert power == pytest.approx(1702.8, abs=1)


def test_compute_bop():
    """ Tests computation of the Balance of Plant """

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(ComputeBoP()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeBoP(), ivc)
    vol = problem.get_val("data:geometry:hybrid_powertrain:bop:volume", units='m**3')
    assert vol == pytest.approx(0.120, abs=1e-3)
    l = problem.get_val("data:geometry:hybrid_powertrain:bop:length_in_nacelle", units='m')
    assert l == pytest.approx(0.3159, abs=1e-3)
    # h = problem.get_val("data:geometry:hybrid_powertrain:bop:extra_height", units='m')
    # assert h == pytest.approx(0.2336, abs=1e-3)


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
    inputs = list_inputs(ComputeBatteries())
    inputs.remove("data:propulsion:hybrid_powertrain:battery:sys_nom_voltage")
    ivc = get_indep_var_comp(inputs, __file__, XML_FILE)

    ivc.add_output("data:propulsion:hybrid_powertrain:battery:cell_current_limit", val=10, units='A')
    ivc.add_output('data:mission:sizing:end_of_mission:SOC', val=0.2)
    ivc.add_output('data:mission:sizing:total_battery_energy', val=24, units='kW*h')
    ivc.add_output('data:mission:sizing:battery_max_current', val=100)
    ivc.add_output("data:propulsion:hybrid_powertrain:battery:sys_nom_voltage", val = 540, units='V')

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeBatteries(), ivc)
    N_ser = problem.get_val("data:geometry:hybrid_powertrain:battery:N_series", units=None)
    assert N_ser == pytest.approx(181, abs=1.0)
    N_par = problem.get_val("data:geometry:hybrid_powertrain:battery:N_parallel", units=None)
    assert N_par == pytest.approx(16.0, abs=1.0)
    vol = problem.get_val("data:geometry:hybrid_powertrain:battery:pack_volume", units='m**3')
    assert vol == pytest.approx(0.044, abs=1e-3)
    tot_vol = problem.get_val("data:geometry:hybrid_powertrain:battery:tot_volume", units='m**3')
    assert tot_vol == pytest.approx(0.088, abs=1e-3)


def test_compute_inverter():
    """ Tests computation of the inverter """

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(ComputeInverter()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeInverter(), ivc)
    power = problem.get_val("data:propulsion:hybrid_powertrain:inverter:output_power", units='kW')
    assert power == pytest.approx(45.1425, abs=1e-3)
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
    ivc = get_indep_var_comp(list_inputs(ComputeH2Storage(H2_storage_model='physical')), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeH2Storage(H2_storage_model='physical'), ivc)
    vol = problem.get_val("data:geometry:hybrid_powertrain:h2_storage:total_tanks_volume", units='m**3')
    assert vol == pytest.approx(0.373, abs=1e-3)
    R = problem.get_val("data:geometry:hybrid_powertrain:h2_storage:tank_internal_radius", units='m')
    assert R == pytest.approx(0.2725, abs=1e-3)
    L = problem.get_val("data:geometry:hybrid_powertrain:h2_storage:tank_internal_length", units='m')
    assert L == pytest.approx(0.8177, abs=1e-3)
    t = problem.get_val("data:geometry:hybrid_powertrain:h2_storage:wall_thickness", units='m')
    assert t == pytest.approx(0.02018, abs=1e-4)
    volume = problem.get_val("data:geometry:hybrid_powertrain:h2_storage:single_tank_liner_volume", units='m**3')
    assert volume == pytest.approx(0.00711, abs=1e-4)

def test_h2_storage_legacy():
    """ Tests computation of the hydrogen storage """

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(ComputeH2Storage(H2_storage_model='legacy')), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeH2Storage(H2_storage_model='legacy'), ivc)
    vol = problem.get_val("data:geometry:hybrid_powertrain:h2_storage:total_tanks_volume", units='m**3')
    assert vol == pytest.approx(0.456, abs=1e-3)
    R = problem.get_val("data:geometry:hybrid_powertrain:h2_storage:tank_internal_volume", units='m**3')
    assert R == pytest.approx(0.148, abs=1e-3)
    L = problem.get_val("data:geometry:hybrid_powertrain:h2_storage:tank_ext_length", units='m')
    assert L == pytest.approx(0.943, abs=1e-3)
    t = problem.get_val("data:geometry:hybrid_powertrain:h2_storage:tank_ext_diameter", units='m')
    assert t == pytest.approx(0.629, abs=1e-4)

def test_h2_storage_weight_legacy():
    """ Tests computation of the hydrogen storage """

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(ComputeH2StorageWeightLegacy()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeH2StorageWeightLegacy(), ivc)
    vol = problem.get_val("data:weight:hybrid_powertrain:h2_storage:mass", units='kg')
    assert vol == pytest.approx(238, abs=1e-3)

def test_h2_storage_weight_physical():
    """ Tests computation of the hydrogen storage """

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(ComputeH2StorageWeightPhysical()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeH2StorageWeightPhysical(), ivc)
    vol = problem.get_val("data:weight:hybrid_powertrain:h2_storage:mass", units='kg')
    assert vol == pytest.approx(199.05, abs=1e-1)

def test_dep_nacelle_dimensions():
    """ Tests computation of the hydrogen storage """
    XML_FILE = "fc_aircraft_dep.xml"
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(ComputeNacelleDimension(propulsion_id = ENGINE_WRAPPER)), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeNacelleDimension(propulsion_id = ENGINE_WRAPPER), ivc)
    length = problem.get_val("data:geometry:propulsion:nacelle:length", units='m')
    cross_section = problem.get_val("data:geometry:propulsion:nacelle:master_cross_section", units='m**2')
    wet_area = problem.get_val("data:geometry:propulsion:nacelle:wet_area", units='m**2')
    assert length == pytest.approx(4.7, abs=1e-1)
    assert cross_section == pytest.approx(0.38, abs=1e-2)
    assert wet_area == pytest.approx(11.6, abs=1e-1)
