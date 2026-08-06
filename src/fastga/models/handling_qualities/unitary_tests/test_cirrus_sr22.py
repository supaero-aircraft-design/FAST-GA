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

import pytest

from tests.testing_utilities import (
    get_indep_var_comp,
    run_system,
    setup_and_run_system,
)

from .dummy_engines import ENGINE_WRAPPER_SR22 as ENGINE_WRAPPER
from ..compute_static_margin import ComputeStaticMargin
from ..tail_sizing.compute_balked_landing_limit import ComputeBalkedLandingLimit
from ..tail_sizing.compute_to_rotation_limit import ComputeTORotationLimitGroup
from ..tail_sizing.update_ht_area import UpdateHTArea, UpdateHTAreaVolumeCoefficient
from ..tail_sizing.update_vt_area import UpdateVTArea, UpdateVTAreaVolumeCoefficient

XML_FILE = "cirrus_sr22.xml"


def test_update_vt_area():
    """Tests computation of the vertical tail area"""

    # Run problem and check obtained value(s) is/(are) correct
    problem = setup_and_run_system(UpdateVTArea(propulsion_id=ENGINE_WRAPPER), __file__, XML_FILE)
    vt_area = problem.get_val("data:geometry:vertical_tail:area", units="m**2")
    assert vt_area == pytest.approx(1.41, abs=1e-2)

    vt_area_constraints_cruise = problem.get_val(
        "data:constraints:vertical_tail:target_cruise_stability", units="m**2"
    )
    assert vt_area_constraints_cruise == pytest.approx(0.0, abs=1e-2)
    vt_area_constraints_crosswind = problem.get_val(
        "data:constraints:vertical_tail:crosswind_landing", units="m**2"
    )
    assert vt_area_constraints_crosswind == pytest.approx(0.0, abs=1e-2)
    vt_area_constraints_eo_climb = problem.get_val(
        "data:constraints:vertical_tail:engine_out_climb", units="m**2"
    )
    assert vt_area_constraints_eo_climb == pytest.approx(1.41, abs=1e-2)
    vt_area_constraints_eo_takeoff = problem.get_val(
        "data:constraints:vertical_tail:engine_out_takeoff", units="m**2"
    )
    assert vt_area_constraints_eo_takeoff == pytest.approx(1.41, abs=1e-2)
    vt_area_constraints_eo_landing = problem.get_val(
        "data:constraints:vertical_tail:engine_out_landing", units="m**2"
    )
    assert vt_area_constraints_eo_landing == pytest.approx(
        1.41, abs=1e-2
    )  # Should be equal to vtp_area but since
    # the dummy engine is not recognized as an ICE engine the last constraints applies even though
    # it shouldn't


def test_update_ht_area():
    """Tests computation of the horizontal tail area"""

    # Research independent input value in .xml file
    problem = setup_and_run_system(UpdateHTArea(propulsion_id=ENGINE_WRAPPER), __file__, XML_FILE)
    ht_area = problem.get_val("data:geometry:horizontal_tail:area", units="m**2")
    assert ht_area == pytest.approx(3.95, abs=1e-2)


def test_update_tail_area_volume():
    """
    Tests computation of the horizontal tail area and vertical tail area with volume coefficient
    """
    # Research independent input value in .xml file
    inputs_list = [
        "data:geometry:wing:area",
        "data:geometry:wing:span",
        "data:geometry:vertical_tail:volume_coefficient",
        "data:geometry:vertical_tail:MAC:at25percent:x:from_wingMAC25",
    ]

    ivc = get_indep_var_comp(inputs_list, __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(UpdateVTAreaVolumeCoefficient(propulsion_id=ENGINE_WRAPPER), ivc)

    vt_area = problem.get_val("data:geometry:vertical_tail:area", units="m**2")
    assert vt_area == pytest.approx(1.41, abs=1e-2)

    problem.check_partials(compact_print=True)

    # Research independent input value in .xml file
    inputs_list = [
        "data:geometry:wing:area",
        "data:geometry:wing:MAC:length",
        "data:geometry:horizontal_tail:volume_coefficient",
        "data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25",
    ]

    ivc = get_indep_var_comp(inputs_list, __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(UpdateHTAreaVolumeCoefficient(propulsion_id=ENGINE_WRAPPER), ivc)

    ht_area = problem.get_val("data:geometry:horizontal_tail:area", units="m**2")
    assert ht_area == pytest.approx(3.95, abs=1e-2)

    problem.check_partials(compact_print=True)


def test_compute_static_margin():
    """Tests computation of static margin"""

    # Research independent input value in .xml file
    problem = setup_and_run_system(ComputeStaticMargin(), __file__, XML_FILE)
    stick_fixed_static_margin = problem["data:handling_qualities:stick_fixed_static_margin"]
    assert stick_fixed_static_margin == pytest.approx(0.34, abs=1e-2)
    free_elevator_factor = problem["data:aerodynamics:cruise:neutral_point:free_elevator_factor"]
    assert free_elevator_factor == pytest.approx(0.74, abs=1e-2)
    stick_free_static_margin = problem["data:handling_qualities:stick_free_static_margin"]
    assert stick_free_static_margin == pytest.approx(0.27, abs=1e-2)


def test_compute_to_rotation_limit():
    """Tests computation of static margin"""

    problem = setup_and_run_system(
        ComputeTORotationLimitGroup(propulsion_id=ENGINE_WRAPPER), __file__, XML_FILE
    )
    x_cg_rotation_limit = problem["data:handling_qualities:to_rotation_limit:x"]
    assert x_cg_rotation_limit == pytest.approx(1.98, abs=1e-2)
    x_cg_ratio_rotation_limit = problem["data:handling_qualities:to_rotation_limit:MAC_position"]
    assert x_cg_ratio_rotation_limit == pytest.approx(-0.47, abs=1e-2)


def test_compute_balked_landing():
    """Tests computation of static margin"""

    # Research independent input value in .xml file
    problem = setup_and_run_system(
        ComputeBalkedLandingLimit(propulsion_id=ENGINE_WRAPPER), __file__, XML_FILE
    )
    x_cg_balked_landing_limit = problem["data:handling_qualities:balked_landing_limit:x"]
    assert x_cg_balked_landing_limit == pytest.approx(2.67, abs=1e-2)
    x_cg_ratio_balked_landing_limit = problem[
        "data:handling_qualities:balked_landing_limit:MAC_position"
    ]
    assert x_cg_ratio_balked_landing_limit == pytest.approx(0.11, abs=1e-2)
