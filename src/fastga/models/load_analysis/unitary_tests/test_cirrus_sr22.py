"""
Test load_analysis module
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

import pandas as pd
import numpy as np
from openmdao.core.component import Component
import pytest
from typing import Union

from fastoad.module_management.service_registry import RegisterPropulsion
from fastoad.model_base import FlightPoint, Atmosphere
from fastoad.model_base.propulsion import IOMPropulsionWrapper

from ..aerostructural_loads import AerostructuralLoad
from ..structural_loads import StructuralLoads
from ..aerodynamic_loads import AerodynamicLoads

from tests.testing_utilities import run_system, get_indep_var_comp, list_inputs

from fastga.models.propulsion.fuel_propulsion.base import AbstractFuelPropulsion
from fastga.models.propulsion.propulsion import IPropulsion

XML_FILE = "cirrus_sr22.xml"
ENGINE_WRAPPER = "test.wrapper.load_analysis.cirrus.dummy_engine"


class DummyEngine(AbstractFuelPropulsion):

    def __init__(self):
        """
        Dummy engine model returning thrust in particular conditions defined for htp/vtp areas.

        """
        super().__init__()
        self.max_power = 231000.0
        self.max_thrust = 5417.0

    def compute_flight_points(self, flight_points: Union[FlightPoint, pd.DataFrame]):

        altitude = float(Atmosphere(np.array(flight_points.altitude)).get_altitude(altitude_in_feet=True))
        mach = np.array(flight_points.mach)
        thrust = np.array(flight_points.thrust)
        sigma = Atmosphere(altitude).density / Atmosphere(0.0).density
        max_power = self.max_power * (sigma - (1 - sigma) / 7.55)
        max_thrust = min(
            self.max_thrust * sigma ** (1. / 3.),
            max_power * 0.8 / np.maximum(mach * Atmosphere(altitude).speed_of_sound, 1e-20)
        )
        if flight_points.thrust_rate is None:
            flight_points.thrust = min(max_thrust, float(thrust))
            flight_points.thrust_rate = float(thrust) / max_thrust
        else:
            flight_points.thrust = max_thrust * np.array(flight_points.thrust_rate)
        sfc_pmax = 8.5080e-08  # fixed whatever the thrust ratio, sfc for ONE 130kW engine !
        sfc = sfc_pmax * flight_points.thrust_rate * mach * Atmosphere(altitude).speed_of_sound
        flight_points.sfc = sfc

    def compute_weight(self) -> float:
        return 0.0

    def compute_dimensions(self) -> (float, float, float, float):
        return [0.0, 0.0, 0.0, 0.0]

    def compute_drag(self, mach, unit_reynolds, l0_wing):
        return 0.0

    # noinspection PyMethodMayBeStatic
    def compute_sl_thrust(self) -> float:
        return 5417.0


class DummyEngineWrapper(IOMPropulsionWrapper):
    def setup(self, component: Component):
        component.add_input("data:TLAR:v_cruise", np.nan, units="m/s")
        component.add_input("data:mission:sizing:main_route:cruise:altitude", np.nan, units="m")

    @staticmethod
    def get_model(inputs) -> IPropulsion:
        return DummyEngine()


RegisterPropulsion(ENGINE_WRAPPER)(DummyEngineWrapper)


def test_compute_shear_stress():

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(AerostructuralLoad()), __file__, XML_FILE)
    cl_vector_only_prop = [-0.266, 0.1052, 0.1068, 0.1022, 0.0935, 0.0791, 0.0615,
                           0.0087, 0.0071, 0.0057, 0.0041, 0.0033, 0.003, 0.0027,
                           0.0024, 0.0023, 0.002, 0.0018, 0.0014, 0.0012, 0.0009,
                           0.0008, 0.0007, 0.0007, 0.0007, 0.0007, 0.0007, 0.0007,
                           0.0007, 0.0007, 0.0007, 0.0007, 0.0006, 0.0006, 0.0006,
                           0.0006, 0.0006, 0.0006, 0.0014, 0., 0., 0.,
                           0., 0., 0., 0., 0., 0., 0.,
                           0.]
    y_vector = [0.045, 0.136, 0.227, 0.318, 0.408, 0.499, 0.59, 0.716, 0.88,
                1.045, 1.21, 1.376, 1.543, 1.71, 1.878, 2.046, 2.214, 2.383,
                2.551, 2.719, 2.887, 3.055, 3.222, 3.389, 3.556, 3.721, 3.886,
                4.05, 4.212, 4.374, 4.535, 4.694, 4.852, 5.008, 5.163, 5.317,
                5.468, 5.618, 5.766, 0., 0., 0., 0., 0., 0.,
                0., 0., 0., 0., 0.]
    ivc.add_output("data:aerodynamics:slipstream:wing:cruise:only_prop:CL_vector", cl_vector_only_prop)
    ivc.add_output("data:aerodynamics:slipstream:wing:cruise:prop_on:Y_vector", y_vector, units="m")
    ivc.add_output("data:aerodynamics:slipstream:wing:cruise:prop_on:velocity", 82.311, units="m/s")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(AerostructuralLoad(), ivc)
    shear_max_mass_condition = problem.get_val("data:loads:max_shear:mass", units="kg")
    assert shear_max_mass_condition == pytest.approx(1457.251, abs=1e-1)
    shear_max_lf_condition = problem.get_val("data:loads:max_shear:load_factor")
    assert shear_max_lf_condition == pytest.approx(4.170, abs=1e-2)
    lift_shear_diagram = problem.get_val("data:loads:max_shear:lift_shear", units="N")
    lift_root_shear = lift_shear_diagram[0]
    assert lift_root_shear == pytest.approx(46968.880, abs=1)
    weight_shear_diagram = problem.get_val("data:loads:max_shear:weight_shear", units="N")
    weight_root_shear = weight_shear_diagram[0]
    assert weight_root_shear == pytest.approx(-6342.346, abs=1)


def test_compute_root_bending_moment():

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(AerostructuralLoad()), __file__, XML_FILE)
    cl_vector_only_prop = [-0.266, 0.1052, 0.1068, 0.1022, 0.0935, 0.0791, 0.0615,
                           0.0087, 0.0071, 0.0057, 0.0041, 0.0033, 0.003, 0.0027,
                           0.0024, 0.0023, 0.002, 0.0018, 0.0014, 0.0012, 0.0009,
                           0.0008, 0.0007, 0.0007, 0.0007, 0.0007, 0.0007, 0.0007,
                           0.0007, 0.0007, 0.0007, 0.0007, 0.0006, 0.0006, 0.0006,
                           0.0006, 0.0006, 0.0006, 0.0014, 0., 0., 0.,
                           0., 0., 0., 0., 0., 0., 0.,
                           0.]
    y_vector = [0.045, 0.136, 0.227, 0.318, 0.408, 0.499, 0.59, 0.716, 0.88,
                1.045, 1.21, 1.376, 1.543, 1.71, 1.878, 2.046, 2.214, 2.383,
                2.551, 2.719, 2.887, 3.055, 3.222, 3.389, 3.556, 3.721, 3.886,
                4.05, 4.212, 4.374, 4.535, 4.694, 4.852, 5.008, 5.163, 5.317,
                5.468, 5.618, 5.766, 0., 0., 0., 0., 0., 0.,
                0., 0., 0., 0., 0.]
    ivc.add_output("data:aerodynamics:slipstream:wing:cruise:only_prop:CL_vector", cl_vector_only_prop)
    ivc.add_output("data:aerodynamics:slipstream:wing:cruise:prop_on:Y_vector", y_vector, units="m")
    ivc.add_output("data:aerodynamics:slipstream:wing:cruise:prop_on:velocity", 82.311, units="m/s")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(AerostructuralLoad(), ivc)
    max_rbm_mass_condition = problem.get_val("data:loads:max_rbm:mass", units="kg")
    assert max_rbm_mass_condition == pytest.approx(1457.251, abs=1e-1)
    max_rbm_lf_condition = problem.get_val("data:loads:max_rbm:load_factor")
    assert max_rbm_lf_condition == pytest.approx(4.170, abs=1e-2)
    lift_rbm_diagram = problem.get_val("data:loads:max_rbm:lift_rbm", units="N*m")
    lift_rbm = lift_rbm_diagram[0]
    assert lift_rbm == pytest.approx(119173.146, abs=1)
    weight_rbm_diagram = problem.get_val("data:loads:max_rbm:weight_rbm", units="N*m")
    weight_rbm = weight_rbm_diagram[0]
    assert weight_rbm == pytest.approx(-14215.587, abs=1)


def test_compute_mass_distribution():

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(StructuralLoads()), __file__, XML_FILE)
    load_factor_shear = 4.0
    ivc.add_output("data:loads:max_shear:load_factor", load_factor_shear)
    load_factor_rbm = 4.0
    ivc.add_output("data:loads:max_rbm:load_factor", load_factor_rbm)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(StructuralLoads(), ivc)
    point_mass_array = problem.get_val("data:loads:structure:ultimate:force_distribution:point_mass", units="N/m")
    point_mass_result = np.array([-0., -0., -0., -0., -0., -0., -0., -0., -0., -0., -16953.9, -16953.9, -16953.9,
                                  -16953.9, -16953.9, -16953.9, -0., -0., -0., -0., -0., -0., -0., -0., -0., -0., -0.,
                                  -0., -0., -0., -0., -0., -0., -0., -0., -0., -0., -0., -0., -0., -0., -0., -0., -0.,
                                  0., -0., -0., -0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                  0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    assert np.max(np.abs(point_mass_array - point_mass_result)) <= 1e-1
    wing_mass_array = problem.get_val("data:loads:structure:ultimate:force_distribution:wing", units="N/m")
    wing_mass_result = np.array([-718.5, -714.7, -714.7, -714.7, -714.7, -714.7, -714.7, -714.7,
                                 -709.1, -698.8, -698.7, -697.9, -697.9, -697.1, -696.3, -695.5,
                                 -695.5, -686.6, -675.2, -663.8, -652.3, -640.9, -629.4, -617.8,
                                 -606.3, -594.7, -583.1, -571.6, -560.1, -548.5, -537.1, -525.6,
                                 -514.2, -502.8, -491.5, -480.3, -469.1, -458., -447., -436.,
                                 -425.2, -414.5, -403.8, -393.3, -382.9, -372.6, -362.4, -359.3,
                                 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    assert np.max(np.abs(wing_mass_result - wing_mass_array)) <= 1e-1
    fuel_mass_array = problem.get_val("data:loads:structure:ultimate:force_distribution:fuel", units="N/m")
    fuel_mass_result = np.array([-1154.3, -1148.2, -1148.2, -1148.2, -1148.2, -1148.2, -1148.2,
                                 -1148.2, -1139.2, -1122.6, -1122.5, -1121.2, -1121.1, -1119.9,
                                 -1118.7, -1117.4, -1117.3, -1103., -1084.7, -1066.4, -1048.,
                                 -1029.6, -1011.1, -992.5, -974., -955.4, -936.8, -918.3,
                                 -899.7, -881.2, -862.8, -844.4, -826., -807.8, -789.6,
                                 -771.6, -753.6, -735.8, -718.1, -700.5, -683.1, -665.8,
                                 -648.8, -631.8, -615.1, -598.6, -582.2, -577.1, 0., 0., 0., 0., 0., 0., 0., 0.,
                                 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])

    assert np.max(np.abs(fuel_mass_result - fuel_mass_array)) <= 1e-1


def test_compute_structure_shear():

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(StructuralLoads()), __file__, XML_FILE)
    load_factor_shear = 4.0
    ivc.add_output("data:loads:max_shear:load_factor", load_factor_shear)
    load_factor_rbm = 4.0
    ivc.add_output("data:loads:max_rbm:load_factor", load_factor_rbm)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(StructuralLoads(), ivc)
    point_mass_array = problem.get_val("data:loads:structure:ultimate:shear:point_mass", units="N")
    point_mass_result = np.array([-808.4, -808.4, -808.4, -808.4, -808.4, -808.4, -808.4, -808.4,
                                  -808.4, -808.4, -800., -602.1, -585., -404.2, -206.3, -8.5,
                                  0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                  0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                  0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                  0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    assert np.max(np.abs(point_mass_array - point_mass_result)) <= 1e-1
    wing_mass_array = problem.get_val("data:loads:structure:ultimate:shear:wing", units="N")
    wing_mass_result = np.array([-3241.5, -3209., -3144., -3079.1, -3014.2, -2949.2, -2884.3,
                                 -2819.4, -2729.2, -2623.7, -2623., -2614.9, -2614.2, -2606.7,
                                 -2598.6, -2590.5, -2589.8, -2500.4, -2387.9, -2276.8, -2167.2,
                                 -2059.1, -1952.7, -1848., -1745.2, -1644.2, -1545.2, -1448.1,
                                 -1353.1, -1260.2, -1169.5, -1081., -994.6, -910.6, -828.8,
                                 -749.3, -672.1, -597.2, -524.7, -454.4, -386.5, -320.9,
                                 -257.6, -196.5, -137.7, -81.1, -26.7, 0., 0.,
                                 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    assert np.max(np.abs(wing_mass_result - wing_mass_array)) <= 1e-1
    fuel_mass_array = problem.get_val("data:loads:structure:ultimate:shear:fuel", units="N")
    fuel_mass_result = np.array([-5207.6, -5155.3, -5051., -4946.7, -4842.3, -4738., -4633.7,
                                 -4529.4, -4384.5, -4215.1, -4213.9, -4200.8, -4199.7, -4187.8,
                                 -4174.7, -4161.6, -4160.5, -4016.9, -3836.2, -3657.7, -3481.6,
                                 -3308., -3137.1, -2968.9, -2803.7, -2641.4, -2482.3, -2326.4,
                                 -2173.8, -2024.6, -1878.8, -1736.6, -1597.9, -1462.8, -1331.4,
                                 -1203.7, -1079.7, -959.4, -842.9, -730.1, -620.9, -515.5,
                                 -413.8, -315.7, -221.2, -130.3, -42.9, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    assert np.max(np.abs(fuel_mass_result - fuel_mass_array)) <= 1e-1


def test_compute_structure_bending():

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(StructuralLoads()), __file__, XML_FILE)
    load_factor_shear = 4.0
    ivc.add_output("data:loads:max_shear:load_factor", load_factor_shear)
    load_factor_rbm = 4.0
    ivc.add_output("data:loads:max_rbm:load_factor", load_factor_rbm)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(StructuralLoads(), ivc)
    point_mass_array = problem.get_val("data:loads:structure:ultimate:root_bending:point_mass", units="N*m")
    point_mass_result = np.array([-720.7, -683.9, -610.5, -537., -463.6, -390.1, -316.7, -243.2,
                                  -140.8, -19.7, -18.9, -10.7, -10.1, -4.8, -1.3, 0.,
                                  0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                  0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                  0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                  0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    assert np.max(np.abs(point_mass_array - point_mass_result)) <= 1e-1
    wing_mass_array = problem.get_val("data:loads:structure:ultimate:root_bending:wing", units="N*m")
    wing_mass_result = np.array([-8356.3, -8209.7, -7921.1, -7638.4, -7361.6, -7090.7, -6825.7,
                                 -6566.6, -6215.1, -5814.2, -5811.6, -5781., -5778.4, -5750.6,
                                 -5720.2, -5689.9, -5687.3, -5358., -4954.3, -4567.3, -4197.3,
                                 -3844.2, -3508.3, -3189.4, -2887.5, -2602.6, -2334.5, -2083.,
                                 -1848., -1629.1, -1426.1, -1238.7, -1066.4, -909., -766.,
                                 -637., -521.5, -419.1, -329.2, -251.4, -185.1, -129.9,
                                 -85.2, -50.5, -25.2, -8.9, -1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])

    assert np.max(np.abs(wing_mass_result - wing_mass_array)) <= 1e-1
    fuel_mass_array = problem.get_val("data:loads:structure:ultimate:root_bending:fuel", units="N*m")
    fuel_mass_result = np.array([-13424.5, -13189.1, -12725.4, -12271.3, -11826.6, -11391.3,
                                 -10965.6, -10549.3, -9984.6, -9340.6, -9336.4, -9287.3,
                                 -9283.1, -9238.4, -9189.6, -9140.9, -9136.7, -8607.7,
                                 -7959.1, -7337.5, -6743., -6175.9, -5636.1, -5123.8,
                                 -4638.9, -4181.2, -3750.4, -3346.4, -2968.8, -2617.2,
                                 -2291.1, -1989.9, -1713.2, -1460.3, -1230.6, -1023.3,
                                 -837.8, -673.2, -528.9, -403.9, -297.4, -208.7,
                                 -136.9, -81.1, -40.5, -14.3, -1.6, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])

    assert np.max(np.abs(fuel_mass_result - fuel_mass_array)) <= 1e-1


def test_compute_lift_distribution():

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(AerodynamicLoads()), __file__, XML_FILE)
    load_factor_shear = 4.0
    ivc.add_output("data:loads:max_shear:load_factor", load_factor_shear)
    ivc.add_output("data:loads:max_shear:mass", 1638.94, units="kg")
    load_factor_rbm = 4.0
    ivc.add_output("data:loads:max_rbm:load_factor", load_factor_rbm)
    ivc.add_output("data:loads:max_rbm:mass", 1638.94, units="kg")
    cl_vector_only_prop = [0.0161, 0.0017, 0.0021, 0.0026, 0.0034, 0.0042, 0.0052, 0.0063, 0.0082, 0.0117, 0.0426,
                           0.0739, 0.1035, 0.1072, 0.0931, 0.0327, 0.0136, 0.001, -0.0069, -0.0062, -0.0259, -0.0192,
                           -0.0147, -0.0117, -0.0097, -0.0082, -0.0071, -0.0065, -0.006, -0.0055, -0.0048, -0.0042,
                           -0.0033, -0.0027, -0.0021, -0.0017, -0.001, -0.0006, 0.0004, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                           0.0, 0.0, 0.0, 0.0, 0.0]
    y_vector = [0.04278, 0.12834, 0.21389, 0.29945, 0.38501, 0.47056, 0.55612, 0.67982, 0.84215, 1.00539, 1.16945,
                1.33423, 1.49963, 1.66555, 1.8319, 1.99856, 2.16543, 2.33242, 2.49942, 2.66632, 2.83301, 2.99941,
                3.16539, 3.33087, 3.49574, 3.6599, 3.82325, 3.98571, 4.14717, 4.30755, 4.46676, 4.6247, 4.78131,
                4.9365, 5.0902, 5.24233, 5.39282, 5.5416, 5.68862, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0]
    ivc.add_output("data:aerodynamics:slipstream:wing:cruise:only_prop:CL_vector", cl_vector_only_prop)
    ivc.add_output("data:aerodynamics:slipstream:wing:cruise:prop_on:Y_vector", y_vector, units="m")
    ivc.add_output("data:aerodynamics:slipstream:wing:cruise:prop_on:velocity", 84.368, units="m/s")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(AerodynamicLoads(), ivc)
    lift_array = problem.get_val("data:loads:aerodynamic:ultimate:force_distribution", units="N/m")
    lift_result = np.array([7274.92, 7234.12, 7167.82, 7165.37, 7161.37, 7160.4, 7154.52,
                            7140.69, 7082.78, 7018.11, 7017.72, 7013.1, 7012.7, 7011.42,
                            7010.01, 7008.58, 7008.46, 7023.82, 7104.16, 7173.47, 7171.57,
                            7068.4, 6858.86, 6548.22, 6376.11, 6233.73, 6103.32, 5977.09,
                            5807.24, 5724.69, 5625.88, 5524.06, 5404.5, 5289.51, 5156.1,
                            5029.07, 4890.7, 4768.26, 4631.49, 4503.69, 4360.36, 4199.45,
                            3996.99, 3761.7, 3459.69, 2979.53, 2282.57, 0., 0.,
                            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])

    assert np.max(np.abs(lift_array - lift_result)) <= 1e-1
