"""
Test module for OpenMDAO versions of basicICEngine
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

import openmdao.api as om
from fastoad.constants import EngineSetting

from ..openmdao import OMBasicTPEngineMappedComponent

from .data.dummy_maps import *

from tests.testing_utilities import run_system


def test_OMBasicTPEngineMappedComponent():
    """Tests ManualBasicTPEngine component"""
    # Same test as in test_basicIC_engine.test_compute_flight_points
    engine = OMBasicTPEngineMappedComponent(flight_point_count=(2, 5))

    machs = [0.06, 0.12, 0.18, 0.22, 0.375]
    altitudes = [0, 0, 0, 1000, 2400]
    thrust_rates = [0.8, 0.5, 0.5, 0.4, 0.7]
    thrusts = [8723.73643444, 4352.43513423, 3670.56367899, 2378.52672608, 2754.44695866]
    engine_settings = [
        EngineSetting.TAKEOFF,
        EngineSetting.TAKEOFF,
        EngineSetting.CLIMB,
        EngineSetting.IDLE,
        EngineSetting.CRUISE,
    ]  # mix EngineSetting with integers
    expected_sfc = [7.11080994e-06, 1.14707988e-05, 1.40801854e-05, 1.75188296e-05, 1.82845376e-05]

    ivc = om.IndepVarComp()
    ivc.add_output("data:propulsion:turboprop:design_point:power", 1342.285, units="kW")
    ivc.add_output(
        "data:propulsion:turboprop:design_point:turbine_entry_temperature", 1400, units="K"
    )
    ivc.add_output("data:propulsion:turboprop:design_point:OPR", 12.0)
    ivc.add_output("data:propulsion:turboprop:design_point:altitude", 0.0, units="ft")
    ivc.add_output("data:propulsion:turboprop:design_point:mach", 0.0)
    ivc.add_output("data:propulsion:turboprop:off_design:bleed_usage", val=1.0)
    ivc.add_output("data:propulsion:turboprop:off_design:itt_limit", val=1125.0, units="K")
    ivc.add_output("data:propulsion:turboprop:off_design:power_limit", val=634.0, units="kW")
    ivc.add_output("data:propulsion:turboprop:off_design:opr_limit", val=12.0)
    ivc.add_output("data:TLAR:v_cruise", 164.622, units="m/s")
    ivc.add_output("data:aerodynamics:propeller:cruise_level:altitude", 6096.0, units="m")
    ivc.add_output("data:geometry:propulsion:engine:layout", val=1.0)
    ivc.add_output("data:geometry:propulsion:engine:count", 1.0)
    ivc.add_output("data:aerodynamics:propeller:sea_level:speed", SPEED, units="m/s")
    ivc.add_output("data:aerodynamics:propeller:sea_level:thrust", THRUST_SL, units="N")
    ivc.add_output("data:aerodynamics:propeller:sea_level:thrust_limit", THRUST_SL_LIMIT, units="N")
    ivc.add_output("data:aerodynamics:propeller:sea_level:efficiency", EFFICIENCY_SL)
    ivc.add_output("data:aerodynamics:propeller:cruise_level:speed", SPEED, units="m/s")
    ivc.add_output("data:aerodynamics:propeller:cruise_level:thrust", THRUST_CL, units="N")
    ivc.add_output(
        "data:aerodynamics:propeller:cruise_level:thrust_limit", THRUST_CL_LIMIT, units="N"
    )
    ivc.add_output("data:aerodynamics:propeller:cruise_level:efficiency", EFFICIENCY_CL)
    ivc.add_output(
        "data:propulsion:turboprop:sea_level:mach",
        val=MACH_ARRAY,
    )
    ivc.add_output("data:propulsion:turboprop:sea_level:thrust", units="N", val=THRUST_ARRAY_SL)
    ivc.add_output(
        "data:propulsion:turboprop:sea_level:thrust_limit",
        val=THRUST_MAX_ARRAY_SL,
        units="N",
    )
    ivc.add_output(
        "data:propulsion:turboprop:sea_level:sfc",
        units="kg/s/N",
        val=SFC_SL,
    )

    ivc.add_output(
        "data:propulsion:turboprop:cruise_level:mach",
        val=MACH_ARRAY,
    )
    ivc.add_output(
        "data:propulsion:turboprop:cruise_level:thrust",
        val=THRUST_ARRAY_CL,
        units="N",
    )
    ivc.add_output(
        "data:propulsion:turboprop:cruise_level:thrust_limit",
        val=THRUST_MAX_ARRAY_CL,
        units="N",
    )
    ivc.add_output(
        "data:propulsion:turboprop:cruise_level:sfc",
        units="kg/s/N",
        val=SFC_CL,
    )

    ivc.add_output("data:propulsion:turboprop:intermediate_level:altitude", val=3048, units="m")
    ivc.add_output(
        "data:propulsion:turboprop:intermediate_level:mach",
        val=MACH_ARRAY,
    )
    ivc.add_output(
        "data:propulsion:turboprop:intermediate_level:thrust",
        val=THRUST_ARRAY_IL,
        units="N",
    )
    ivc.add_output(
        "data:propulsion:turboprop:intermediate_level:thrust_limit",
        val=THRUST_MAX_ARRAY_IL,
        units="N",
    )
    ivc.add_output(
        "data:propulsion:turboprop:intermediate_level:sfc",
        val=SFC_IL,
        units="kg/s/N",
    )

    ivc.add_output("data:propulsion:mach", [machs, machs])
    ivc.add_output("data:propulsion:altitude", [altitudes, altitudes], units="m")
    ivc.add_output("data:propulsion:engine_setting", [engine_settings, engine_settings])
    ivc.add_output("data:propulsion:use_thrust_rate", [[True] * 5, [False] * 5])
    ivc.add_output("data:propulsion:required_thrust_rate", [thrust_rates, [0] * 5])
    ivc.add_output("data:propulsion:required_thrust", [[0] * 5, thrusts], units="N")

    problem = run_system(engine, ivc)

    np.testing.assert_allclose(problem["data:propulsion:thrust"], [thrusts, thrusts], rtol=1e-2)
    np.testing.assert_allclose(
        problem["data:propulsion:thrust_rate"], [thrust_rates, thrust_rates], rtol=1e-2
    )
    np.testing.assert_allclose(
        problem["data:propulsion:SFC"], [expected_sfc, expected_sfc], rtol=1e-2
    )
