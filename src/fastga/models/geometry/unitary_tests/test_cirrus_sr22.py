"""
Test module for geometry functions of cg components.
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
import pytest

from openmdao.utils.assert_utils import assert_check_partials

from ..geom_components.fuselage.components import (
    ComputeFuselageGeometryBasic,
    ComputeFuselageDepth,
    ComputeFuselageVolume,
    ComputeFuselageWetArea,
    ComputeFuselageWetAreaFLOPS,
    ComputeMasterCrossSection,
    ComputeFuselageCabinLength,
    ComputeFuselageLengthFD,
    ComputeFuselageLengthFL,
    ComputeFuselageLuggageLength,
    ComputeFuselageMaxHeight,
    ComputeFuselageMaxWidth,
    ComputeFuselageNoseLengthFD,
    ComputeFuselageNoseLengthFL,
    ComputeFuselageNPAX,
    ComputeFuselagePAXLength,
    ComputeFuselageRearLength,
    ComputePlaneLength,
)
from ..geom_components.wing.components import (
    ComputeWingB50,
    ComputeWingL1,
    ComputeWingL2,
    ComputeWingL3,
    ComputeWingL4,
    ComputeWingMacLength,
    ComputeWingMacX,
    ComputeWingMacY,
    ComputeWingSweep0,
    ComputeWingSweep50,
    ComputeWingSweep100Inner,
    ComputeWingSweep100Outer,
    ComputeWingTocRoot,
    ComputeWingTocKink,
    ComputeWingTocTip,
    ComputeWingWetArea,
    ComputeWingOuterArea,
    ComputeWingXKink,
    ComputeWingXTip,
    ComputeWingY,
    ComputeWingZRoot,
    ComputeWingZTip,
    ComputeWingXAbsoluteMac,
    ComputeWingXAbsoluteTip,
)
from ..geom_components.ht.components import (
    ComputeHTMacLength,
    ComputeHTMacY,
    ComputeHTMacX25,
    ComputeHTMacX25Wing,
    ComputeHTWetArea,
    ComputeHTDistance,
    ComputeHTVolumeCoefficient,
    ComputeHTSpan,
    ComputeHTRootChord,
    ComputeHTTipChord,
    ComputeHTSweep0,
    ComputeHTSweep50,
    ComputeHTSweep100,
    ComputeHTEfficiency,
)
from ..geom_components.vt.components import (
    ComputeVTRootChord,
    ComputeVTTipChord,
    ComputeVTSpan,
    ComputeVTMacLength,
    ComputeVTMacX25Local,
    ComputeVTMacZ,
    ComputeVTMacPositionFD,
    ComputeVTMacPositionFL,
    ComputeVTSweep0,
    ComputeVTSweep50,
    ComputeVTSweep100,
    ComputeVTWetArea,
    ComputeVTXTip,
)
from ..geom_components.nacelle.components import (
    ComputeNacelleDimension,
    ComputeNacelleXPosition,
    ComputeNacelleYPosition,
)
from ..geom_components.propeller.components import (
    ComputePropellerPosition,
    ComputePropellerInstallationEffect,
)
from ..geom_components.landing_gears.components import ComputeLGHeight, ComputeLGPosition
from ..geom_components.wing_tank import ComputeMFWSimple, ComputeMFWAdvanced
from ..geom_components import ComputeTotalArea
from ..geometry import GeometryFixedFuselage, GeometryFixedTailDistance

from tests.testing_utilities import run_system, get_indep_var_comp, list_inputs

from .dummy_engines import ENGINE_WRAPPER_SR22 as ENGINE_WRAPPER

XML_FILE = "cirrus_sr22.xml"


def test_vt_root_chord():
    """Tests computation of the vertical tail chord: root"""

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(ComputeVTRootChord()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeVTRootChord(), ivc)
    root_chord = problem.get_val("data:geometry:vertical_tail:root:chord", units="m")
    assert root_chord == pytest.approx(1.118, abs=1e-3)

    data = problem.check_partials(compact_print=True)
    assert_check_partials(data, atol=1.0e-3, rtol=1.0e-3)


def test_vt_tip_chord():
    """Tests computation of the vertical tail chord: tip"""

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(ComputeVTTipChord()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeVTTipChord(), ivc)
    tip_chord = problem.get_val("data:geometry:vertical_tail:tip:chord", units="m")
    assert tip_chord == pytest.approx(0.561, abs=1e-3)

    data = problem.check_partials(compact_print=True)
    assert_check_partials(data, atol=1.0e-3, rtol=1.0e-3)


def test_vt_tip_x():
    """Tests computation of the vertical tail tip x position"""

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(ComputeVTXTip()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeVTXTip(), ivc)
    tip_x = problem.get_val("data:geometry:vertical_tail:tip:x", units="m")
    assert tip_x == pytest.approx(6.941, abs=1e-3)

    data = problem.check_partials(compact_print=True)
    assert_check_partials(data, atol=1.0e-3, rtol=1.0e-3)


def test_vt_span():
    """Tests computation of the vertical tail span"""

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(ComputeVTSpan()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeVTSpan(), ivc)
    span = problem.get_val("data:geometry:vertical_tail:span", units="m")
    assert span == pytest.approx(1.680, abs=1e-3)

    data = problem.check_partials(compact_print=True)
    assert_check_partials(data, atol=1.0e-3, rtol=1.0e-3)


def test_vt_mac_length():
    """Tests computation of the vertical tail mac length"""

    # Research independent input value in .xml file and add values calculated from other modules
    ivc = get_indep_var_comp(list_inputs(ComputeVTMacLength()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeVTMacLength(), ivc)
    length = problem.get_val("data:geometry:vertical_tail:MAC:length", units="m")
    assert length == pytest.approx(0.871, abs=1e-3)

    data = problem.check_partials(compact_print=True)
    assert_check_partials(data, atol=1.0e-3, rtol=1.0e-3)


def test_vt_mac_x25_local():
    """Tests computation of the vertical tail 25% mac x position (local)"""

    # Research independent input value in .xml file and add values calculated from other modules
    ivc = get_indep_var_comp(list_inputs(ComputeVTMacX25Local()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeVTMacX25Local(), ivc)
    vt_x0 = problem.get_val("data:geometry:vertical_tail:MAC:at25percent:x:local", units="m")
    assert vt_x0 == pytest.approx(0.193, abs=1e-3)

    data = problem.check_partials(compact_print=True)
    assert_check_partials(data, atol=1.0e-3, rtol=1.0e-3)


def test_vt_mac_z():
    """Tests computation of the vertical tail mac z position"""

    # Research independent input value in .xml file and add values calculated from other modules
    ivc = get_indep_var_comp(list_inputs(ComputeVTMacZ()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeVTMacZ(), ivc)
    vt_z0 = problem.get_val("data:geometry:vertical_tail:MAC:z", units="m")
    assert vt_z0 == pytest.approx(0.747, abs=1e-3)

    data = problem.check_partials(compact_print=True)
    assert_check_partials(data, atol=1.0e-3, rtol=1.0e-3)


def test_vt_mac_position():
    """Tests computation of the vertical tail mac position from wing"""

    # Research independent input value in .xml file and add values calculated from other modules
    ivc = get_indep_var_comp(list_inputs(ComputeVTMacPositionFD()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeVTMacPositionFD(), ivc)
    lp_vt = problem.get_val(
        "data:geometry:vertical_tail:MAC:at25percent:x:from_wingMAC25", units="m"
    )
    assert lp_vt == pytest.approx(4.0, abs=1e-3)

    data = problem.check_partials(compact_print=True)
    assert_check_partials(data, atol=1.0e-3, rtol=1.0e-3)


def test_vt_mac_position_fl():
    """Tests computation of the vertical tail mac position from wing"""

    # Research independent input value in .xml file and add values calculated from other modules
    ivc = get_indep_var_comp(list_inputs(ComputeVTMacPositionFL()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeVTMacPositionFL(), ivc)
    lp_vt = problem.get_val(
        "data:geometry:vertical_tail:MAC:at25percent:x:from_wingMAC25", units="m"
    )
    assert lp_vt == pytest.approx(4.255, abs=1e-3)

    data = problem.check_partials(compact_print=True)
    assert_check_partials(data, atol=1.0e-3, rtol=1.0e-3)


def test_vt_sweep_0():
    """Tests computation of the vertical tail sweep at l/c=0%"""

    # Research independent input value in .xml file and add values calculated from other modules
    ivc = get_indep_var_comp(list_inputs(ComputeVTSweep0()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeVTSweep0(), ivc)
    sweep_0 = problem.get_val("data:geometry:vertical_tail:sweep_0", units="deg")
    assert sweep_0 == pytest.approx(14.532, abs=1e-1)

    data = problem.check_partials(compact_print=True)
    assert_check_partials(data, atol=1.0e-3, rtol=1.0e-3)


def test_vt_sweep_50():
    """Tests computation of the vertical tail sweep at l/c=50%"""

    # Research independent input value in .xml file and add values calculated from other modules
    ivc = get_indep_var_comp(list_inputs(ComputeVTSweep50()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeVTSweep50(), ivc)
    sweep_50 = problem.get_val("data:geometry:vertical_tail:sweep_50", units="deg")
    assert sweep_50 == pytest.approx(-4.13, abs=1e-1)

    data = problem.check_partials(compact_print=True)
    assert_check_partials(data, atol=1.0e-3, rtol=1.0e-3)


def test_vt_sweep_100():
    """Tests computation of the vertical tail sweep at l/c=100%"""

    # Research independent input value in .xml file and add values calculated from other modules
    ivc = get_indep_var_comp(list_inputs(ComputeVTSweep100()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeVTSweep100(), ivc)
    sweep_100 = problem.get_val("data:geometry:vertical_tail:sweep_100", units="deg")
    assert sweep_100 == pytest.approx(-4.13, abs=1e-1)

    data = problem.check_partials(compact_print=True)
    assert_check_partials(data, atol=1.0e-3, rtol=1.0e-3)


def test_vt_wet_area():
    """Tests computation of the vertical wet area"""

    # Research independent input value in .xml file and add values calculated from other modules
    ivc = get_indep_var_comp(list_inputs(ComputeVTWetArea()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeVTWetArea(), ivc)
    wet_area = problem.get_val("data:geometry:vertical_tail:wet_area", units="m**2")
    assert wet_area == pytest.approx(2.965, abs=1e-3)

    data = problem.check_partials(compact_print=True)
    assert_check_partials(data, atol=1.0e-3, rtol=1.0e-3)


def test_ht_distance():
    """Tests computation of the horizontal tail distance"""

    # Research independent input value in .xml file and add values calculated from other modules
    ivc = get_indep_var_comp(list_inputs(ComputeHTDistance()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeHTDistance(), ivc)
    lp_vt = problem.get_val("data:geometry:horizontal_tail:z:from_wingMAC25", units="m")
    assert lp_vt == pytest.approx(0.0, abs=1e-3)

    data = problem.check_partials(compact_print=True)
    assert_check_partials(data, atol=1.0e-3, rtol=1.0e-3)


def test_ht_span():
    """Tests computation of the horizontal tail span"""

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(ComputeHTSpan()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeHTSpan(), ivc)
    span = problem.get_val("data:geometry:horizontal_tail:span", units="m")
    assert span == pytest.approx(3.824, abs=1e-3)

    data = problem.check_partials(compact_print=True)
    assert_check_partials(data, atol=1.0e-3, rtol=1.0e-3)


def test_ht_chord_root():
    """Tests computation of the horizontal tail root chord"""

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(ComputeHTRootChord()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeHTRootChord(), ivc)
    root_chord = problem.get_val("data:geometry:horizontal_tail:root:chord", units="m")
    assert root_chord == pytest.approx(0.866, abs=1e-3)

    data = problem.check_partials(compact_print=True)
    assert_check_partials(data, atol=1.0e-3, rtol=1.0e-3)


def test_ht_chord_tip():
    """Tests computation of the horizontal tail tip chord"""

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(ComputeHTTipChord()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeHTTipChord(), ivc)
    tip_chord = problem.get_val("data:geometry:horizontal_tail:tip:chord", units="m")
    assert tip_chord == pytest.approx(0.531, abs=1e-3)

    data = problem.check_partials(compact_print=True)
    assert_check_partials(data, atol=1.0e-3, rtol=1.0e-3)


def test_compute_ht_aspect_ratio():
    """Tests computation of the horizontal tail aspect ratio"""

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(ComputeHTSpan()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeHTSpan(), ivc)
    aspect_ratio = problem.get_val("data:geometry:horizontal_tail:aspect_ratio")
    assert aspect_ratio == pytest.approx(5.47, abs=1e-3)


def test_ht_mac_length():
    """Tests computation of the horizontal tail mac length"""

    # Research independent input value in .xml file and add values calculated from other modules
    ivc = get_indep_var_comp(list_inputs(ComputeHTMacLength()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeHTMacLength(), ivc)
    length = problem.get_val("data:geometry:horizontal_tail:MAC:length", units="m")
    assert length == pytest.approx(0.712, abs=1e-3)

    data = problem.check_partials(compact_print=True)
    assert_check_partials(data, atol=1.0e-3, rtol=1.0e-3)


def test_ht_mac_x0():
    """Tests computation of the horizontal tail mac x local"""

    # Research independent input value in .xml file and add values calculated from other modules
    ivc = get_indep_var_comp(list_inputs(ComputeHTMacX25()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeHTMacX25(), ivc)
    ht_x0 = problem.get_val("data:geometry:horizontal_tail:MAC:at25percent:x:local", units="m")
    assert ht_x0 == pytest.approx(0.100, abs=1e-3)

    data = problem.check_partials(compact_print=True)
    assert_check_partials(data, atol=1.0e-3, rtol=1.0e-3)


def test_ht_mac_x0_from_wing():
    """Tests computation of the horizontal tail mac x from 25% wing mac"""

    # Research independent input value in .xml file and add values calculated from other modules
    ivc = get_indep_var_comp(list_inputs(ComputeHTMacX25Wing()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeHTMacX25Wing(), ivc)
    lp_ht = problem.get_val(
        "data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25", units="m"
    )
    assert lp_ht == pytest.approx(3.887, abs=1e-3)

    data = problem.check_partials(compact_print=True)
    del data["component"][
        "data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25", "data:geometry:has_T_tail"
    ]
    assert_check_partials(data, atol=1.0e-3, rtol=1.0e-3)


def test_ht_mac_y0():
    """Tests computation of the horizontal tail mac y"""

    # Research independent input value in .xml file and add values calculated from other modules
    ivc = get_indep_var_comp(list_inputs(ComputeHTMacY()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeHTMacY(), ivc)
    ht_y0 = problem.get_val("data:geometry:horizontal_tail:MAC:y", units="m")
    assert ht_y0 == pytest.approx(0.880, abs=1e-3)

    data = problem.check_partials(compact_print=True)
    assert_check_partials(data, atol=1.0e-3, rtol=1.0e-3)


def test_ht_sweep_0():
    """Tests computation of the horizontal tail sweep at l/c=0%"""

    # Research independent input value in .xml file and add values calculated from other modules
    ivc = get_indep_var_comp(list_inputs(ComputeHTSweep0()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeHTSweep0(), ivc)
    sweep_0 = problem.get_val("data:geometry:horizontal_tail:sweep_0", units="deg")
    assert sweep_0 == pytest.approx(6.491, abs=1e-1)

    data = problem.check_partials(compact_print=True)
    assert_check_partials(data, atol=1.0e-3, rtol=1.0e-3)


def test_ht_sweep_50():
    """Tests computation of the horizontal tail sweep at l/c=50%"""

    # Research independent input value in .xml file and add values calculated from other modules
    ivc = get_indep_var_comp(list_inputs(ComputeHTSweep50()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeHTSweep50(), ivc)
    sweep_50 = problem.get_val("data:geometry:horizontal_tail:sweep_50", units="deg")
    assert sweep_50 == pytest.approx(1.4930608, abs=1e-1)

    data = problem.check_partials(compact_print=True)
    assert_check_partials(data, atol=1.0e-3, rtol=1.0e-3)


def test_ht_sweep_100():
    """Tests computation of the horizontal tail sweep at l/c=100%"""

    # Research independent input value in .xml file and add values calculated from other modules
    ivc = get_indep_var_comp(list_inputs(ComputeHTSweep100()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeHTSweep100(), ivc)
    sweep_100 = problem.get_val("data:geometry:horizontal_tail:sweep_100", units="deg")
    assert sweep_100 == pytest.approx(176.471, abs=1e-1)

    data = problem.check_partials(compact_print=True)
    assert_check_partials(data, atol=1.0e-3, rtol=1.0e-3)


def test_ht_efficiency():
    """Tests computation of the horizontal tail efficiency"""

    # Research independent input value in .xml file and add values calculated from other modules
    ivc = get_indep_var_comp(list_inputs(ComputeHTEfficiency()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeHTEfficiency(), ivc)
    sweep_100 = problem.get_val("data:aerodynamics:horizontal_tail:efficiency")
    assert sweep_100 == pytest.approx(0.9, abs=1e-3)

    data = problem.check_partials(compact_print=True)
    assert_check_partials(data, atol=1.0e-3, rtol=1.0e-3)


def test_ht_wet_area():
    """Tests computation of the horizontal tail wet area"""

    # Research independent input value in .xml file and add values calculated from other modules
    ivc = get_indep_var_comp(list_inputs(ComputeHTWetArea()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeHTWetArea(), ivc)
    wet_area = problem.get_val("data:geometry:horizontal_tail:wet_area", units="m**2")
    assert wet_area == pytest.approx(5.615, abs=1e-2)

    data = problem.check_partials(compact_print=True)
    assert_check_partials(data, atol=1.0e-3, rtol=1.0e-3)


def test_ht_volume_coefficient():
    """Tests computation of the horizontal tail volume coefficient"""

    # Research independent input value in .xml file and add values calculated from other modules
    ivc = get_indep_var_comp(list_inputs(ComputeHTVolumeCoefficient()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeHTVolumeCoefficient(), ivc)
    vol_coeff = problem.get_val("data:geometry:horizontal_tail:volume_coefficient")
    assert vol_coeff == pytest.approx(0.674, rel=1e-2)

    data = problem.check_partials(compact_print=True)
    assert_check_partials(data, atol=1.0e-3, rtol=1.0e-3)


def test_fuselage_basic():
    """Tests computation of the fuselage with no cabin sizing"""

    # Define the independent input values that should be filled if basic function is chosen
    ivc = om.IndepVarComp()
    ivc.add_output("data:geometry:fuselage:length", 8.888, units="m")
    ivc.add_output("data:geometry:fuselage:maximum_width", 1.198, units="m")
    ivc.add_output("data:geometry:fuselage:maximum_height", 1.338, units="m")
    ivc.add_output("data:geometry:fuselage:front_length", 2.274, units="m")
    ivc.add_output("data:geometry:fuselage:rear_length", 2.852, units="m")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeFuselageGeometryBasic(), ivc)
    fuselage_lcabin = problem.get_val("data:geometry:cabin:length", units="m")
    assert fuselage_lcabin == pytest.approx(3.762, abs=1e-3)

    data = problem.check_partials(compact_print=True)
    assert_check_partials(data, atol=1.0e-3, rtol=1.0e-3)


def test_fuselage_npax():
    """Tests computation of the fuselage npax"""

    # Research independent input value in .xml file and add values calculated from other modules
    ivc = get_indep_var_comp(list_inputs(ComputeFuselageNPAX()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeFuselageNPAX(), ivc)
    npax = problem.get_val("data:geometry:cabin:NPAX")
    assert npax == pytest.approx(2.0, abs=1)


def test_fuselage_pax_length():
    """Tests computation of the fuselage pax length"""

    # Research independent input value in .xml file and add values calculated from other modules
    ivc = get_indep_var_comp(list_inputs(ComputeFuselagePAXLength()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeFuselagePAXLength(), ivc)
    fuselage_lpax = problem.get_val("data:geometry:fuselage:PAX_length", units="m")
    assert fuselage_lpax == pytest.approx(1.75, abs=1e-3)

    data = problem.check_partials(compact_print=True)
    assert_check_partials(data, atol=1.0e-3, rtol=1.0e-3)


def test_fuselage_max_width():
    """Tests computation of the fuselage maximum width"""

    # Research independent input value in .xml file and add values calculated from other modules
    ivc = get_indep_var_comp(list_inputs(ComputeFuselageMaxWidth()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeFuselageMaxWidth(), ivc)
    fuselage_width_max = problem.get_val("data:geometry:fuselage:maximum_width", units="m")
    assert fuselage_width_max == pytest.approx(1.272, abs=1e-3)


def test_fuselage_max_height():
    """Tests computation of the fuselage maximum height"""

    # Research independent input value in .xml file and add values calculated from other modules
    ivc = get_indep_var_comp(list_inputs(ComputeFuselageMaxHeight()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeFuselageMaxHeight(), ivc)
    fuselage_height_max = problem.get_val("data:geometry:fuselage:maximum_height", units="m")
    assert fuselage_height_max == pytest.approx(1.412, abs=1e-3)

    data = problem.check_partials(compact_print=True)
    assert_check_partials(data, atol=1.0e-3, rtol=1.0e-3)


def test_fuselage_luggage_length():
    """Tests computation of the fuselage luggage length"""

    # Research independent input value in .xml file and add values calculated from other modules
    ivc = get_indep_var_comp(list_inputs(ComputeFuselageLuggageLength()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeFuselageLuggageLength(), ivc)
    luggage_length = problem.get_val("data:geometry:fuselage:luggage_length", units="m")
    assert luggage_length == pytest.approx(0.411, abs=1e-3)


def test_fuselage_cabin_length():
    """Tests computation of the fuselage cabin length"""

    # Research independent input value in .xml file and add values calculated from other modules
    ivc = get_indep_var_comp(list_inputs(ComputeFuselageCabinLength()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeFuselageCabinLength(), ivc)
    fuselage_lcabin = problem.get_val("data:geometry:cabin:length", units="m")
    assert fuselage_lcabin == pytest.approx(2.861, abs=1e-3)

    data = problem.check_partials(compact_print=True)
    assert_check_partials(data, atol=1.0e-3, rtol=1.0e-3)


def test_fuselage_nose_length_fd():
    """Tests computation of the fuselage nose length FD"""

    # Research independent input value in .xml file and add values calculated from other modules
    ivc = get_indep_var_comp(list_inputs(ComputeFuselageNoseLengthFD()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeFuselageNoseLengthFD(), ivc)
    fuselage_lav = problem.get_val("data:geometry:fuselage:front_length", units="m")
    assert fuselage_lav == pytest.approx(1.448, abs=1e-3)

    data = problem.check_partials(compact_print=True)
    del data["component"][
        "data:geometry:fuselage:front_length", "data:geometry:propulsion:engine:layout"
    ]
    assert_check_partials(data, atol=1.0e-3, rtol=1.0e-3)


def test_fuselage_nose_length_fl():
    """Tests computation of the fuselage nose length FL"""

    # Research independent input value in .xml file and add values calculated from other modules
    ivc = get_indep_var_comp(list_inputs(ComputeFuselageNoseLengthFL()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeFuselageNoseLengthFL(), ivc)
    fuselage_lav = problem.get_val("data:geometry:fuselage:front_length", units="m")
    assert fuselage_lav == pytest.approx(1.148, abs=1e-3)

    data = problem.check_partials(compact_print=True)
    del data["component"][
        "data:geometry:fuselage:front_length", "data:geometry:propulsion:engine:layout"
    ]
    assert_check_partials(data, atol=1.0e-3, rtol=1.0e-3)


def test_fuselage_length_fd():
    """Tests computation of the fuselage length FD"""

    # Research independent input value in .xml file and add values calculated from other modules
    ivc = get_indep_var_comp(list_inputs(ComputeFuselageLengthFD()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeFuselageLengthFD(), ivc)
    fuselage_length = problem.get_val("data:geometry:fuselage:length", units="m")
    assert fuselage_length == pytest.approx(7.491, abs=1e-3)

    data = problem.check_partials(compact_print=True)
    assert_check_partials(data, atol=1.0e-3, rtol=1.0e-3)


def test_fuselage_length_fl():
    """Tests computation of the fuselage length FL"""

    # Research independent input value in .xml file and add values calculated from other modules
    ivc = get_indep_var_comp(list_inputs(ComputeFuselageLengthFL()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeFuselageLengthFL(), ivc)
    fuselage_length = problem.get_val("data:geometry:fuselage:length", units="m")
    assert fuselage_length == pytest.approx(7.492, abs=1e-3)


def test_fuselage_rear_length():
    """Tests computation of the fuselage rear length"""

    # Research independent input value in .xml file and add values calculated from other modules
    ivc = get_indep_var_comp(list_inputs(ComputeFuselageRearLength()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeFuselageRearLength(), ivc)
    fuselage_lar = problem.get_val("data:geometry:fuselage:rear_length", units="m")
    assert fuselage_lar == pytest.approx(3.181, abs=1e-3)

    data = problem.check_partials(compact_print=True)
    assert_check_partials(data, atol=1.0e-3, rtol=1.0e-3)


def test_plane_length():
    """Tests computation of the plane length"""

    # Research independent input value in .xml file and add values calculated from other modules
    ivc = get_indep_var_comp(list_inputs(ComputePlaneLength()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputePlaneLength(), ivc)
    fuselage_lar = problem.get_val("data:geometry:aircraft:length", units="m")
    assert fuselage_lar == pytest.approx(7.788, abs=1e-3)

    data = problem.check_partials(compact_print=True)
    assert_check_partials(data, atol=1.0e-3, rtol=1.0e-3)


def test_fuselage_wet_area():

    ivc = get_indep_var_comp(
        list_inputs(ComputeFuselageWetArea()),
        __file__,
        XML_FILE,
    )

    problem = run_system(ComputeFuselageWetArea(), ivc)
    fuselage_wet_area = problem["data:geometry:fuselage:wet_area"]
    assert fuselage_wet_area == pytest.approx(26.613, abs=1e-3)

    data = problem.check_partials(compact_print=True)
    assert_check_partials(data, atol=1.0e-3, rtol=1.0e-3)


def test_fuselage_wet_area_flops():

    ivc = get_indep_var_comp(
        list_inputs(ComputeFuselageWetAreaFLOPS()),
        __file__,
        XML_FILE,
    )

    problem = run_system(ComputeFuselageWetAreaFLOPS(), ivc)
    fuselage_wet_area = problem["data:geometry:fuselage:wet_area"]
    assert fuselage_wet_area == pytest.approx(21.952, abs=1e-3)

    data = problem.check_partials(compact_print=True)
    assert_check_partials(data, atol=1.0e-3, rtol=1.0e-3)


def test_fuselage_master_cross_section():

    ivc = get_indep_var_comp(
        list_inputs(ComputeMasterCrossSection()),
        __file__,
        XML_FILE,
    )

    problem = run_system(ComputeMasterCrossSection(), ivc)
    fuselage_master_cross_section = problem["data:geometry:fuselage:master_cross_section"]
    assert fuselage_master_cross_section == pytest.approx(1.410, abs=1e-3)

    data = problem.check_partials(compact_print=True)
    assert_check_partials(data, atol=1.0e-3, rtol=1.0e-3)


def test_fuselage_depth():

    ivc = get_indep_var_comp(
        list_inputs(ComputeFuselageDepth()),
        __file__,
        XML_FILE,
    )

    problem = run_system(ComputeFuselageDepth(), ivc)
    avg_fuselage_depth = problem.get_val("data:geometry:fuselage:average_depth", units="m")
    assert avg_fuselage_depth == pytest.approx(0.235, rel=1e-2)

    data = problem.check_partials(compact_print=True)
    assert_check_partials(data, atol=1.0e-3, rtol=1.0e-3)


def test_fuselage_volume():

    ivc = get_indep_var_comp(
        list_inputs(ComputeFuselageVolume()),
        __file__,
        XML_FILE,
    )

    problem = run_system(ComputeFuselageVolume(), ivc)
    avg_fuselage_depth = problem.get_val("data:geometry:fuselage:volume", units="m**3")
    assert avg_fuselage_depth == pytest.approx(7.711, rel=1e-2)

    data = problem.check_partials(compact_print=True)
    assert_check_partials(data, atol=1.0e-3, rtol=1.0e-3)


def test_wing_toc_root():
    """Tests computation of the wing root ToC (Thickness of Chord)"""

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(ComputeWingTocRoot()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeWingTocRoot(), ivc)
    toc_root = problem["data:geometry:wing:root:thickness_ratio"]
    assert toc_root == pytest.approx(0.149, abs=1e-3)

    data = problem.check_partials(compact_print=True)
    assert_check_partials(data, atol=1.0e-3, rtol=1.0e-3)


def test_wing_toc_kink():
    """Tests computation of the wing kink ToC (Thickness of Chord)"""

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(ComputeWingTocKink()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeWingTocKink(), ivc)
    toc_kink = problem["data:geometry:wing:kink:thickness_ratio"]
    assert toc_kink == pytest.approx(0.113, abs=1e-3)

    data = problem.check_partials(compact_print=True)
    assert_check_partials(data, atol=1.0e-3, rtol=1.0e-3)


def test_wing_toc_tip():
    """Tests computation of the wing tip ToC (Thickness of Chord)"""

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(ComputeWingTocTip()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeWingTocTip(), ivc)
    toc_tip = problem["data:geometry:wing:tip:thickness_ratio"]
    assert toc_tip == pytest.approx(0.103, abs=1e-3)

    data = problem.check_partials(compact_print=True)
    assert_check_partials(data, atol=1.0e-3, rtol=1.0e-3)


def test_geometry_wing_y():
    """Tests computation of the wing Ys"""

    # Research independent input value in .xml file and add values calculated from other modules
    ivc = get_indep_var_comp(list_inputs(ComputeWingY()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeWingY(), ivc)
    span = problem.get_val("data:geometry:wing:span", units="m")
    assert span == pytest.approx(11.599, abs=1e-3)
    wing_y2 = problem.get_val("data:geometry:wing:root:y", units="m")
    assert wing_y2 == pytest.approx(0.636, abs=1e-3)
    wing_y3 = problem.get_val("data:geometry:wing:kink:y", units="m")
    assert wing_y3 == pytest.approx(0.0, abs=1e-3)  # point 3 is virtual central point
    wing_y4 = problem.get_val("data:geometry:wing:tip:y", units="m")
    assert wing_y4 == pytest.approx(5.799, abs=1e-3)


def test_wing_z_root():
    """Tests computation of the wing root Z"""

    # Research independent input value in .xml file and add values calculated from other modules
    ivc = get_indep_var_comp(list_inputs(ComputeWingZRoot()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeWingZRoot(), ivc)
    wing_z2 = problem.get_val("data:geometry:wing:root:z", units="m")
    assert wing_z2 == pytest.approx(0.596, rel=1e-2)

    data = problem.check_partials(compact_print=True)
    del data["component"]["data:geometry:wing:root:z", "data:geometry:wing_configuration"]
    assert_check_partials(data, atol=1.0e-3, rtol=1.0e-3)


def test_wing_z_tip():
    """Tests computation of the wing tip Z"""

    # Research independent input value in .xml file and add values calculated from other modules
    ivc = get_indep_var_comp(list_inputs(ComputeWingZTip()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeWingZTip(), ivc)
    wing_z4 = problem.get_val("data:geometry:wing:tip:z", units="m")
    assert wing_z4 == pytest.approx(0.216, rel=1e-2)

    data = problem.check_partials(compact_print=True)
    del data["component"]["data:geometry:wing:tip:z", "data:geometry:wing_configuration"]
    assert_check_partials(data, atol=1.0e-3, rtol=1.0e-3)


def test_wing_l1():
    """Tests computation of the wing chords (l1)"""

    # Research independent input value in .xml file and add values calculated from other modules
    ivc = get_indep_var_comp(list_inputs(ComputeWingL1()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeWingL1(), ivc)
    wing_l1 = problem.get_val("data:geometry:wing:root:virtual_chord", units="m")
    assert wing_l1 == pytest.approx(1.4742, abs=1e-3)

    data = problem.check_partials(compact_print=True)
    assert_check_partials(data, atol=1.0e-3, rtol=1.0e-3)


def test_wing_l2():
    """Tests computation of the wing chords (l2)"""

    # Research independent input value in .xml file and add values calculated from other modules
    ivc = get_indep_var_comp(list_inputs(ComputeWingL2()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeWingL2(), ivc)
    wing_l2 = problem.get_val("data:geometry:wing:root:chord", units="m")
    assert wing_l2 == pytest.approx(1.474, abs=1e-2)

    data = problem.check_partials(compact_print=True)
    assert_check_partials(data, atol=1.0e-3, rtol=1.0e-3)


def test_wing_l3():
    """Tests computation of the wing chords (l3)"""

    # Research independent input value in .xml file and add values calculated from other modules
    ivc = get_indep_var_comp(list_inputs(ComputeWingL3()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeWingL3(), ivc)
    wing_l3 = problem.get_val("data:geometry:wing:kink:chord", units="m")
    assert wing_l3 == pytest.approx(
        1.474, abs=1e-2
    )  # point 3 and 2 equal (previous version ignored)

    data = problem.check_partials(compact_print=True)
    assert_check_partials(data, atol=1.0e-3, rtol=1.0e-3)


def test_wing_l4():
    """Tests computation of the wing chords (l4)"""

    # Research independent input value in .xml file and add values calculated from other modules
    ivc = get_indep_var_comp(list_inputs(ComputeWingL4()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeWingL4(), ivc)
    wing_l4 = problem.get_val("data:geometry:wing:tip:chord", units="m")
    assert wing_l4 == pytest.approx(0.737, abs=1e-3)

    data = problem.check_partials(compact_print=True)
    assert_check_partials(data, atol=1.0e-3, rtol=1.0e-3)


def test_wing_x_kink():
    """Tests computation of the wing kink X local"""

    # Research independent input value in .xml file and add values calculated from other modules
    ivc = get_indep_var_comp(list_inputs(ComputeWingXKink()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeWingXKink(), ivc)
    wing_x3 = problem.get_val("data:geometry:wing:kink:leading_edge:x:local", units="m")
    assert wing_x3 == pytest.approx(0.0, abs=1e-3)

    data = problem.check_partials(compact_print=True)
    assert_check_partials(data, atol=1.0e-3, rtol=1.0e-3)


def test_wing_x_tip():
    """Tests computation of the wing tip X local"""

    # Research independent input value in .xml file and add values calculated from other modules
    ivc = get_indep_var_comp(list_inputs(ComputeWingXTip()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeWingXTip(), ivc)
    wing_x4 = problem.get_val("data:geometry:wing:tip:leading_edge:x:local", units="m")
    assert wing_x4 == pytest.approx(0.184, abs=1e-3)

    data = problem.check_partials(compact_print=True)
    assert_check_partials(data, atol=1.0e-3, rtol=1.0e-3)


def test_wing_x_absolute_mac():
    """Tests computation of the wing MAC absolute X"""

    # Research independent input value in .xml file and add values calculated from other modules
    ivc = get_indep_var_comp(list_inputs(ComputeWingXAbsoluteMac()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeWingXAbsoluteMac(), ivc)
    wing_x0_abs = problem.get_val("data:geometry:wing:MAC:leading_edge:x:absolute", units="m")
    assert wing_x0_abs == pytest.approx(2.539, abs=1e-3)

    data = problem.check_partials(compact_print=True)
    assert_check_partials(data, atol=1.0e-3, rtol=1.0e-3)


def test_wing_x_absolute_tip():
    """Tests computation of the wing tip absolute X"""

    # Research independent input value in .xml file and add values calculated from other modules
    ivc = get_indep_var_comp(list_inputs(ComputeWingXAbsoluteTip()), __file__, XML_FILE)

    # Define input value calculated from other modules
    ivc.add_output("data:geometry:wing:MAC:leading_edge:x:absolute", 2.539, units="m")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeWingXAbsoluteTip(), ivc)
    wing_x4_abs = problem.get_val("data:geometry:wing:tip:leading_edge:x:absolute", units="m")
    assert wing_x4_abs == pytest.approx(2.653, abs=1e-3)

    data = problem.check_partials(compact_print=True)
    assert_check_partials(data, atol=1.0e-3, rtol=1.0e-3)


def test_wing_b50():
    """Tests computation of the wing B50"""

    # Define input values calculated from other modules
    ivc = om.IndepVarComp()
    ivc.add_output("data:geometry:wing:span", 12.363, units="m")
    ivc.add_output("data:geometry:wing:root:y", 0.6, units="m")
    ivc.add_output("data:geometry:wing:tip:y", 6.181, units="m")
    ivc.add_output("data:geometry:wing:root:virtual_chord", 1.549, units="m")
    ivc.add_output("data:geometry:wing:tip:chord", 1.549, units="m")
    ivc.add_output("data:geometry:wing:tip:leading_edge:x:local", 0.0, units="m")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeWingB50(), ivc)
    wing_b_50 = problem.get_val("data:geometry:wing:b_50", units="m")
    assert wing_b_50 == pytest.approx(12.363, abs=1e-3)

    data = problem.check_partials(compact_print=True)
    assert_check_partials(data, atol=1.0e-3, rtol=1.0e-3)


def test_wing_mac_length():
    """Tests computation of the wing mean aerodynamic chord length"""

    # Research independent input value in .xml file and add values calculated from other modules
    ivc = get_indep_var_comp(list_inputs(ComputeWingMacLength()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeWingMacLength(), ivc)
    wing_l0 = problem.get_val("data:geometry:wing:MAC:length", units="m")
    assert wing_l0 == pytest.approx(1.193, abs=1e-3)

    data = problem.check_partials(compact_print=True)
    assert_check_partials(data, atol=1.0e-3, rtol=1.0e-3)


def test_wing_mac_x_pos():
    """Tests computation of the wing mean aerodynamic chord x local"""

    # Research independent input value in .xml file and add values calculated from other modules
    ivc = get_indep_var_comp(list_inputs(ComputeWingMacX()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeWingMacX(), ivc)
    wing_x0 = problem.get_val("data:geometry:wing:MAC:leading_edge:x:local", units="m")
    assert wing_x0 == pytest.approx(0.070, abs=1e-3)

    data = problem.check_partials(compact_print=True)
    assert_check_partials(data, atol=1.0e-3, rtol=1.0e-3)


def test_wing_mac_y_pos():
    """Tests computation of the wing mean aerodynamic chord y position"""

    # Research independent input value in .xml file and add values calculated from other modules
    ivc = get_indep_var_comp(list_inputs(ComputeWingMacY()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeWingMacY(), ivc)
    wing_y0 = problem.get_val("data:geometry:wing:MAC:y", units="m")
    assert wing_y0 == pytest.approx(2.562, abs=1e-3)

    data = problem.check_partials(compact_print=True)
    assert_check_partials(data, atol=1.0e-3, rtol=1.0e-3)


def test_wing_sweep_0():
    """Test computation of the wing sweep at l/c=0%"""

    # Define input values calculated from other modules
    ivc = get_indep_var_comp(list_inputs(ComputeWingSweep0()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeWingSweep0(), ivc)
    sweep_0 = problem.get_val("data:geometry:wing:sweep_0", units="deg")
    assert sweep_0 == pytest.approx(2.044, abs=1e-1)

    data = problem.check_partials(compact_print=True)
    assert_check_partials(data, atol=1.0e-3, rtol=1.0e-3)


def test_wing_sweep_50():
    """Test computation of the wing sweep at l/c=50%"""

    # Define input values calculated from other modules
    ivc = get_indep_var_comp(list_inputs(ComputeWingSweep50()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeWingSweep50(), ivc)
    sweep_50 = problem.get_val("data:geometry:wing:sweep_50", units="deg")
    assert sweep_50 == pytest.approx(-1.73, abs=1e-1)

    data = problem.check_partials(compact_print=True)
    assert_check_partials(data, atol=1.0e-3, rtol=1.0e-3)


def test_wing_sweep_100_inner():
    """Test computation of the wing sweep at l/c=100% (inner)"""

    # Define input values calculated from other modules
    ivc = get_indep_var_comp(list_inputs(ComputeWingSweep100Inner()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeWingSweep100Inner(), ivc)
    sweep_100_inner = problem.get_val("data:geometry:wing:sweep_100_inner", units="deg")
    assert sweep_100_inner == pytest.approx(-6.11, abs=1e-1)

    data = problem.check_partials(compact_print=True)
    assert_check_partials(data, atol=1.0e-3, rtol=1.0e-3)


def test_wing_sweep_100_outer():
    """Test computation of the wing sweep at l/c=100% (outer)"""

    # Define input values calculated from other modules
    ivc = get_indep_var_comp(list_inputs(ComputeWingSweep100Outer()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeWingSweep100Outer(), ivc)
    sweep_100_outer = problem.get_val("data:geometry:wing:sweep_100_outer", units="deg")
    assert sweep_100_outer == pytest.approx(-6.11, abs=1e-1)

    data = problem.check_partials(compact_print=True)
    assert_check_partials(data, atol=1.0e-3, rtol=1.0e-3)


def test_wing_wet_area():
    """Tests computation of the wing wet area"""

    # Research independent input value in .xml file and add values calculated from other modules
    ivc = get_indep_var_comp(list_inputs(ComputeWingWetArea()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeWingWetArea(), ivc)
    wet_area = problem.get_val("data:geometry:wing:wet_area", units="m**2")
    assert wet_area == pytest.approx(24.436, abs=1e-3)

    data = problem.check_partials(compact_print=True)
    assert_check_partials(data, atol=1.0e-3, rtol=1.0e-3)


def test_wing_outer_area():
    """Tests computation of the wing outer area"""

    # Research independent input value in .xml file and add values calculated from other modules
    ivc = get_indep_var_comp(list_inputs(ComputeWingOuterArea()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeWingOuterArea(), ivc)
    area_pf = problem.get_val("data:geometry:wing:outer_area", units="m**2")
    assert area_pf == pytest.approx(11.418, abs=1e-1)

    data = problem.check_partials(compact_print=True)
    assert_check_partials(data, atol=1.0e-3, rtol=1.0e-3)


def test_wing_mfw_simple():
    """Tests computation of the wing max fuel weight"""

    # Research independent input value in .xml file and add values calculated from other modules
    ivc = get_indep_var_comp(list_inputs(ComputeMFWSimple()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeMFWSimple(), ivc)
    mfw = problem.get_val("data:weight:aircraft:MFW", units="kg")
    assert mfw == pytest.approx(296.517, abs=1e-2)

    data = problem.check_partials(compact_print=True)
    del data["component"]["data:weight:aircraft:MFW", "data:propulsion:fuel_type"]
    assert_check_partials(data, atol=1.0e-3, rtol=1.0e-3)


def test_geometry_wing_mfw_advanced():
    """Tests computation of the wing max fuel weight"""

    # Research independent input value in .xml file and add values calculated from other modules
    ivc = get_indep_var_comp(list_inputs(ComputeMFWAdvanced()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeMFWAdvanced(), ivc)
    mfw = problem.get_val("data:weight:aircraft:MFW", units="kg")
    assert mfw == pytest.approx(231.26, abs=1e-2)


def test_dimension_nacelle():
    """Tests computation of the nacelle and pylons component"""

    # Research independent input value in .xml file and add values calculated from other modules
    ivc = get_indep_var_comp(
        list_inputs(ComputeNacelleDimension(propulsion_id=ENGINE_WRAPPER)), __file__, XML_FILE
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeNacelleDimension(propulsion_id=ENGINE_WRAPPER), ivc)
    nacelle_length = problem.get_val("data:geometry:propulsion:nacelle:length", units="m")
    assert nacelle_length == pytest.approx(1.1488, abs=1e-3)
    nacelle_height = problem.get_val("data:geometry:propulsion:nacelle:height", units="m")
    assert nacelle_height == pytest.approx(0.754, abs=1e-3)
    nacelle_width = problem.get_val("data:geometry:propulsion:nacelle:width", units="m")
    assert nacelle_width == pytest.approx(1.125, abs=1e-3)
    nacelle_wet_area = problem.get_val("data:geometry:propulsion:nacelle:wet_area", units="m**2")
    assert nacelle_wet_area == pytest.approx(4.319, abs=1e-3)
    nacelle_master_cross_section = problem.get_val(
        "data:geometry:propulsion:nacelle:master_cross_section", units="m**2"
    )
    assert nacelle_master_cross_section == pytest.approx(0.849, abs=1e-3)


def test_x_position_nacelle():
    """Tests computation of the nacelle and pylons component x position"""

    # Research independent input value in .xml file and add values calculated from other modules
    ivc = get_indep_var_comp(list_inputs(ComputeNacelleXPosition()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeNacelleXPosition(), ivc)
    x_nacelle = problem.get_val("data:geometry:propulsion:nacelle:x", units="m")
    x_nacelle_result = 1.148
    assert abs(x_nacelle - x_nacelle_result) < 1e-3

    data = problem.check_partials(compact_print=True)
    del data["component"][
        "data:geometry:propulsion:nacelle:x", "data:geometry:propulsion:engine:layout"
    ]
    assert_check_partials(data, atol=1.0e-3, rtol=1.0e-3)


def test_y_position_nacelle():
    """Tests computation of the nacelle and pylons component y position"""

    # Research independent input value in .xml file and add values calculated from other modules
    ivc = get_indep_var_comp(list_inputs(ComputeNacelleYPosition()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeNacelleYPosition(), ivc)
    y_nacelle = problem.get_val("data:geometry:propulsion:nacelle:y", units="m")
    y_nacelle_result = 0.0
    assert abs(y_nacelle - y_nacelle_result) < 1e-3

    data = problem.check_partials(compact_print=True)
    del data["component"][
        "data:geometry:propulsion:nacelle:y", "data:geometry:propulsion:engine:layout"
    ]
    assert_check_partials(data, atol=1.0e-3, rtol=1.0e-3)


def test_position_propeller():
    """Tests computation of the nacelle and pylons component"""

    # Research independent input value in .xml file and add values calculated from other modules
    ivc = get_indep_var_comp(list_inputs(ComputePropellerPosition()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputePropellerPosition(), ivc)
    x_prop_from_le = problem.get_val("data:geometry:propulsion:nacelle:from_LE", units="m")
    x_prop_from_le_result = 2.5397
    assert abs(x_prop_from_le - x_prop_from_le_result) < 1e-3

    data = problem.check_partials(compact_print=True)
    del data["component"][
        "data:geometry:propulsion:nacelle:from_LE", "data:geometry:propulsion:engine:layout"
    ]
    assert_check_partials(data, atol=1.0e-3, rtol=1.0e-3)


def test_installation_effect_propeller():
    """Tests computation propeller effective advance ratio factor computation"""

    # Research independent input value in .xml file and add values calculated from other modules
    ivc = get_indep_var_comp(list_inputs(ComputePropellerInstallationEffect()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputePropellerInstallationEffect(), ivc)
    prop_installation_effect = problem.get_val(
        "data:aerodynamics:propeller:installation_effect" ":effective_advance_ratio"
    )
    assert prop_installation_effect == pytest.approx(0.883, abs=1e-3)

    data = problem.check_partials(compact_print=True)
    del data["component"][
        "data:aerodynamics:propeller:installation_effect:effective_advance_ratio",
        "data:geometry:propulsion:engine:layout",
    ]
    assert_check_partials(data, atol=1.0e-3, rtol=1.0e-3)


def test_landing_gear_height():

    # Research independent input value in .xml file and add values calculated from other modules
    ivc = get_indep_var_comp(list_inputs(ComputeLGHeight()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeLGHeight(), ivc)

    lg_height = problem.get_val("data:geometry:landing_gear:height", units="m")
    assert lg_height == pytest.approx(0.811, abs=1e-3)

    data = problem.check_partials(compact_print=True)
    assert_check_partials(data, atol=1.0e-3, rtol=1.0e-3)


def test_landing_gear_position():

    # Research independent input value in .xml file and add values calculated from other modules
    ivc = get_indep_var_comp(list_inputs(ComputeLGPosition()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeLGPosition(), ivc)

    lg_position = problem.get_val("data:geometry:landing_gear:y", units="m")
    assert lg_position == pytest.approx(1.610, abs=1e-3)

    data = problem.check_partials(compact_print=True)
    assert_check_partials(data, atol=1.0e-3, rtol=1.0e-3)


def test_total_area():
    """Tests computation of the total area"""

    # Research independent input value in .xml file and add values calculated from other modules
    ivc = get_indep_var_comp(list_inputs(ComputeTotalArea()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeTotalArea(), ivc)
    total_surface = problem.get_val("data:geometry:aircraft:wet_area", units="m**2")
    assert total_surface == pytest.approx(59.300, abs=1e-3)

    data = problem.check_partials(compact_print=True)
    assert_check_partials(data, atol=1.0e-3, rtol=1.0e-3)


def test_complete_geometry_FD():
    """Run computation of all models for fixed distance hypothesis"""

    # Research independent input value in .xml file and add values calculated from other modules
    # noinspection PyTypeChecker
    ivc = get_indep_var_comp(
        list_inputs(GeometryFixedTailDistance(propulsion_id=ENGINE_WRAPPER)), __file__, XML_FILE
    )

    # Run problem and check obtained value(s) is/(are) correct
    # noinspection PyTypeChecker
    run_system(GeometryFixedTailDistance(propulsion_id=ENGINE_WRAPPER), ivc)


def test_complete_geometry_FL():
    """Run computation of all models for fixed length hypothesis"""

    # Research independent input value in .xml file and add values calculated from other modules
    # noinspection PyTypeChecker
    ivc = get_indep_var_comp(
        list_inputs(GeometryFixedFuselage(propulsion_id=ENGINE_WRAPPER)), __file__, XML_FILE
    )

    # Run problem and check obtained value(s) is/(are) correct
    # noinspection PyTypeChecker
    problem = run_system(GeometryFixedFuselage(propulsion_id=ENGINE_WRAPPER), ivc)
    total_surface = problem.get_val("data:geometry:aircraft:wet_area", units="m**2")
    assert total_surface == pytest.approx(62.965, abs=1e-3)
