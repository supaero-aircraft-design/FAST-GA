"""
Test module for geometry functions of the different components.
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

from ..geom_components.fuselage.components import (
    ComputeFuselageGeometryBasic,
    ComputeFuselageGeometryCabinSizingFD,
    ComputeFuselageGeometryCabinSizingFL,
    ComputeFuselageDepth,
    ComputeFuselageVolume,
    ComputeFuselageWetArea,
    ComputeFuselageWetAreaFLOPS,
)
from ..geom_components.wing.components import (
    ComputeWingB50,
    ComputeWingL1AndL4,
    ComputeWingL2AndL3,
    ComputeWingMAC,
    ComputeWingSweep,
    ComputeWingToc,
    ComputeWingWetArea,
    ComputeWingX,
    ComputeWingY,
    ComputeWingZ,
    ComputeWingXAbsolute,
)
from ..geom_components.ht.components import (
    ComputeHTChord,
    ComputeHTMacFD,
    ComputeHTMacFL,
    ComputeHTSweep,
    ComputeHTWetArea,
    ComputeHTDistance,
    ComputeHTVolumeCoefficient,
)
from ..geom_components.vt.components import (
    ComputeVTChords,
    ComputeVTMacFD,
    ComputeVTMacFL,
    ComputeVTMacPositionFD,
    ComputeVTMacPositionFL,
    ComputeVTSweep,
    ComputeVTWetArea,
)
from ..geom_components.nacelle import ComputeNacellePosition, ComputeNacelleDimension
from ..geom_components.propeller.components import (
    ComputePropellerPosition,
    ComputePropellerInstallationEffect,
)
from ..geom_components.landing_gears.compute_lg import ComputeLGGeometry
from ..geom_components.wing_tank import ComputeMFWSimple, ComputeMFWAdvanced
from ..geom_components import ComputeTotalArea
from ..geometry import GeometryFixedFuselage, GeometryFixedTailDistance

from tests.testing_utilities import run_system, get_indep_var_comp, list_inputs

from .dummy_engines import ENGINE_WRAPPER_BE76 as ENGINE_WRAPPER

XML_FILE = "beechcraft_76.xml"


def test_compute_vt_chords():
    """Tests computation of the vertical tail chords"""

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(ComputeVTChords()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeVTChords(), ivc)
    span = problem.get_val("data:geometry:vertical_tail:span", units="m")
    assert span == pytest.approx(1.459, abs=1e-3)
    root_chord = problem.get_val("data:geometry:vertical_tail:root:chord", units="m")
    assert root_chord == pytest.approx(1.501, abs=1e-3)
    tip_chord = problem.get_val("data:geometry:vertical_tail:tip:chord", units="m")
    assert tip_chord == pytest.approx(0.930, abs=1e-3)


def test_compute_vt_mac():
    """Tests computation of the vertical tail mac"""

    # Research independent input value in .xml file and add values calculated from other modules
    ivc = get_indep_var_comp(list_inputs(ComputeVTMacFD()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeVTMacFD(), ivc)
    length = problem.get_val("data:geometry:vertical_tail:MAC:length", units="m")
    assert length == pytest.approx(1.237, abs=1e-3)
    vt_x0 = problem.get_val("data:geometry:vertical_tail:MAC:at25percent:x:local", units="m")
    assert vt_x0 == pytest.approx(0.453, abs=1e-3)
    vt_z0 = problem.get_val("data:geometry:vertical_tail:MAC:z", units="m")
    assert vt_z0 == pytest.approx(0.672, abs=1e-3)


def test_compute_vt_mac_fl():
    """Tests computation of the horizontal tail mac"""

    # Research independent input value in .xml file and add values calculated from other modules
    ivc = get_indep_var_comp(list_inputs(ComputeVTMacFL()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeVTMacFL(), ivc)
    length = problem.get_val("data:geometry:vertical_tail:MAC:length", units="m")
    assert length == pytest.approx(1.237, abs=1e-3)
    vt_z0 = problem.get_val("data:geometry:vertical_tail:MAC:z", units="m")
    assert vt_z0 == pytest.approx(0.672, abs=1e-3)


def test_compute_vt_mac_position():
    """Tests computation of the vertical tail mac position"""

    # Research independent input value in .xml file and add values calculated from other modules
    ivc = get_indep_var_comp(list_inputs(ComputeVTMacPositionFD()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeVTMacPositionFD(), ivc)
    lp_vt = problem.get_val(
        "data:geometry:vertical_tail:MAC:at25percent:x:from_wingMAC25", units="m"
    )
    assert lp_vt == pytest.approx(4.294, abs=1e-3)


def test_compute_vt_mac_position_fl():
    """Tests computation of the vertical tail mac position"""

    # Research independent input value in .xml file and add values calculated from other modules
    ivc = get_indep_var_comp(list_inputs(ComputeVTMacPositionFL()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeVTMacPositionFL(), ivc)
    lp_vt = problem.get_val(
        "data:geometry:vertical_tail:MAC:at25percent:x:from_wingMAC25", units="m"
    )
    assert lp_vt == pytest.approx(4.808, abs=1e-3)
    vt_x0 = problem.get_val("data:geometry:vertical_tail:MAC:at25percent:x:local", units="m")
    assert vt_x0 == pytest.approx(0.453, abs=1e-3)


def test_compute_vt_sweep():
    """Tests computation of the vertical tail sweep"""

    # Research independent input value in .xml file and add values calculated from other modules
    ivc = get_indep_var_comp(list_inputs(ComputeVTSweep()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeVTSweep(), ivc)
    sweep_0 = problem.get_val("data:geometry:vertical_tail:sweep_0", units="deg")
    assert sweep_0 == pytest.approx(34.03, abs=1e-1)
    sweep_50 = problem.get_val("data:geometry:vertical_tail:sweep_50", units="deg")
    assert sweep_50 == pytest.approx(15.86, abs=1e-1)
    sweep_100 = problem.get_val("data:geometry:vertical_tail:sweep_100", units="deg")
    assert sweep_100 == pytest.approx(15.83, abs=1e-1)


def test_compute_vt_wet_area():
    """Tests computation of the vertical wet area"""

    # Research independent input value in .xml file and add values calculated from other modules
    ivc = get_indep_var_comp(list_inputs(ComputeVTWetArea()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeVTWetArea(), ivc)
    wet_area = problem.get_val("data:geometry:vertical_tail:wet_area", units="m**2")
    assert wet_area == pytest.approx(3.727, abs=1e-3)


def test_compute_ht_distance():
    """Tests computation of the horizontal tail distance"""

    # Research independent input value in .xml file and add values calculated from other modules
    ivc = get_indep_var_comp(list_inputs(ComputeHTDistance()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeHTDistance(), ivc)
    lp_vt = problem.get_val("data:geometry:horizontal_tail:z:from_wingMAC25", units="m")
    assert lp_vt == pytest.approx(1.458, abs=1e-3)


def test_compute_ht_chord():
    """Tests computation of the horizontal tail chords"""

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(ComputeHTChord()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeHTChord(), ivc)
    span = problem.get_val("data:geometry:horizontal_tail:span", units="m")
    assert span == pytest.approx(3.776, abs=1e-3)
    root_chord = problem.get_val("data:geometry:horizontal_tail:root:chord", units="m")
    assert root_chord == pytest.approx(0.983, abs=1e-3)
    tip_chord = problem.get_val("data:geometry:horizontal_tail:tip:chord", units="m")
    assert tip_chord == pytest.approx(0.983, abs=1e-3)


def test_compute_ht_mac():
    """Tests computation of the horizontal tail mac"""

    # Research independent input value in .xml file and add values calculated from other modules
    ivc = get_indep_var_comp(list_inputs(ComputeHTMacFD()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeHTMacFD(), ivc)
    length = problem.get_val("data:geometry:horizontal_tail:MAC:length", units="m")
    assert length == pytest.approx(0.983, abs=1e-3)
    ht_x0 = problem.get_val("data:geometry:horizontal_tail:MAC:at25percent:x:local", units="m")
    assert ht_x0 == pytest.approx(0.065, abs=1e-3)
    ht_y0 = problem.get_val("data:geometry:horizontal_tail:MAC:y", units="m")
    assert ht_y0 == pytest.approx(0.943, abs=1e-3)


def test_compute_ht_mac_fl():
    """Tests computation of the horizontal tail mac"""

    # Research independent input value in .xml file and add values calculated from other modules
    ivc = get_indep_var_comp(list_inputs(ComputeHTMacFL()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeHTMacFL(), ivc)
    length = problem.get_val("data:geometry:horizontal_tail:MAC:length", units="m")
    assert length == pytest.approx(0.983, abs=1e-3)
    ht_x0 = problem.get_val("data:geometry:horizontal_tail:MAC:at25percent:x:local", units="m")
    assert ht_x0 == pytest.approx(0.065, abs=1e-3)
    ht_y0 = problem.get_val("data:geometry:horizontal_tail:MAC:y", units="m")
    assert ht_y0 == pytest.approx(0.943, abs=1e-3)
    lp_ht = problem.get_val(
        "data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25", units="m"
    )
    assert lp_ht == pytest.approx(4.93, abs=1e-3)


def test_compute_ht_sweep():
    """Tests computation of the horizontal tail sweep"""

    # Research independent input value in .xml file and add values calculated from other modules
    ivc = get_indep_var_comp(list_inputs(ComputeHTSweep()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeHTSweep(), ivc)
    sweep_0 = problem.get_val("data:geometry:horizontal_tail:sweep_0", units="deg")
    assert sweep_0 == pytest.approx(4.0, abs=1e-1)
    sweep_50 = problem.get_val("data:geometry:horizontal_tail:sweep_50", units="deg")
    assert sweep_50 == pytest.approx(4.0, abs=1e-1)
    sweep_100 = problem.get_val("data:geometry:horizontal_tail:sweep_100", units="deg")
    assert sweep_100 == pytest.approx(4.0, abs=1e-1)


def test_compute_ht_wet_area():
    """Tests computation of the horizontal tail wet area"""

    # Research independent input value in .xml file and add values calculated from other modules
    ivc = get_indep_var_comp(list_inputs(ComputeHTWetArea()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeHTWetArea(), ivc)
    wet_area = problem.get_val("data:geometry:horizontal_tail:wet_area", units="m**2")
    assert wet_area == pytest.approx(6.239, abs=1e-2)


def test_compute_ht_volume_coefficient():
    """Tests computation of the horizontal tail volume coefficient"""

    # Research independent input value in .xml file and add values calculated from other modules
    ivc = get_indep_var_comp(list_inputs(ComputeHTVolumeCoefficient()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeHTVolumeCoefficient(), ivc)
    vol_coeff = problem.get_val("data:geometry:horizontal_tail:volume_coefficient")
    assert vol_coeff == pytest.approx(0.726, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_compute_fuselage_cabin_sizing_fd():
    """Tests computation of the fuselage with cabin sizing"""

    # Research independent input value in .xml file and add values calculated from other modules
    ivc = get_indep_var_comp(
        list_inputs(ComputeFuselageGeometryCabinSizingFD(propulsion_id=ENGINE_WRAPPER)),
        __file__,
        XML_FILE,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeFuselageGeometryCabinSizingFD(propulsion_id=ENGINE_WRAPPER), ivc)
    npax = problem.get_val("data:geometry:cabin:NPAX")
    assert npax == pytest.approx(2.0, abs=1)
    fuselage_length = problem.get_val("data:geometry:fuselage:length", units="m")
    assert fuselage_length == pytest.approx(8.992, abs=1e-3)
    fuselage_width_max = problem.get_val("data:geometry:fuselage:maximum_width", units="m")
    assert fuselage_width_max == pytest.approx(1.198, abs=1e-3)
    fuselage_height_max = problem.get_val("data:geometry:fuselage:maximum_height", units="m")
    assert fuselage_height_max == pytest.approx(1.338, abs=1e-3)
    fuselage_lav = problem.get_val("data:geometry:fuselage:front_length", units="m")
    assert fuselage_lav == pytest.approx(1.873, abs=1e-3)
    fuselage_lar = problem.get_val("data:geometry:fuselage:rear_length", units="m")
    assert fuselage_lar == pytest.approx(4.222, abs=1e-3)
    fuselage_lpax = problem.get_val("data:geometry:fuselage:PAX_length", units="m")
    assert fuselage_lpax == pytest.approx(1.5, abs=1e-3)
    fuselage_lcabin = problem.get_val("data:geometry:cabin:length", units="m")
    assert fuselage_lcabin == pytest.approx(2.896, abs=1e-3)
    luggage_length = problem.get_val("data:geometry:fuselage:luggage_length", units="m")
    assert luggage_length == pytest.approx(0.696, abs=1e-3)


def test_compute_fuselage_cabin_sizing_fl():
    """Tests computation of the fuselage with cabin sizing"""

    # Research independent input value in .xml file and add values calculated from other modules
    ivc = get_indep_var_comp(
        list_inputs(ComputeFuselageGeometryCabinSizingFL(propulsion_id=ENGINE_WRAPPER)),
        __file__,
        XML_FILE,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeFuselageGeometryCabinSizingFL(propulsion_id=ENGINE_WRAPPER), ivc)
    npax = problem.get_val("data:geometry:cabin:NPAX")
    assert npax == pytest.approx(2.0, abs=1)
    fuselage_length = problem.get_val("data:geometry:fuselage:length", units="m")
    assert fuselage_length == pytest.approx(9.396, abs=1e-3)
    fuselage_width_max = problem.get_val("data:geometry:fuselage:maximum_width", units="m")
    assert fuselage_width_max == pytest.approx(1.198, abs=1e-3)
    fuselage_height_max = problem.get_val("data:geometry:fuselage:maximum_height", units="m")
    assert fuselage_height_max == pytest.approx(1.338, abs=1e-3)
    fuselage_lav = problem.get_val("data:geometry:fuselage:front_length", units="m")
    assert fuselage_lav == pytest.approx(2.274, abs=1e-3)
    fuselage_lpax = problem.get_val("data:geometry:fuselage:PAX_length", units="m")
    assert fuselage_lpax == pytest.approx(1.5, abs=1e-3)
    fuselage_lcabin = problem.get_val("data:geometry:cabin:length", units="m")
    assert fuselage_lcabin == pytest.approx(2.896, abs=1e-3)
    luggage_length = problem.get_val("data:geometry:fuselage:luggage_length", units="m")
    assert luggage_length == pytest.approx(0.696, abs=1e-3)


def test_compute_fuselage_basic():
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


def test_fuselage_wet_area():

    ivc = get_indep_var_comp(
        list_inputs(ComputeFuselageWetArea()),
        __file__,
        XML_FILE,
    )

    problem = run_system(ComputeFuselageWetArea(), ivc)
    fuselage_wet_area = problem["data:geometry:fuselage:wet_area"]
    assert fuselage_wet_area == pytest.approx(29.630, abs=1e-3)
    fuselage_master_cross_section = problem["data:geometry:fuselage:master_cross_section"]
    assert fuselage_master_cross_section == pytest.approx(1.258, abs=1e-3)


def test_fuselage_wet_area_flops():

    ivc = get_indep_var_comp(
        list_inputs(ComputeFuselageWetAreaFLOPS()),
        __file__,
        XML_FILE,
    )

    problem = run_system(ComputeFuselageWetAreaFLOPS(), ivc)
    fuselage_wet_area = problem["data:geometry:fuselage:wet_area"]
    assert fuselage_wet_area == pytest.approx(27.213, abs=1e-3)
    fuselage_master_cross_section = problem["data:geometry:fuselage:master_cross_section"]
    assert fuselage_master_cross_section == pytest.approx(1.258, abs=1e-3)


def test_fuselage_depth():

    ivc = get_indep_var_comp(
        list_inputs(ComputeFuselageDepth()),
        __file__,
        XML_FILE,
    )

    problem = run_system(ComputeFuselageDepth(), ivc)
    avg_fuselage_depth = problem.get_val("data:geometry:fuselage:average_depth", units="m")
    assert avg_fuselage_depth == pytest.approx(0.225, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_fuselage_volume():

    ivc = get_indep_var_comp(
        list_inputs(ComputeFuselageVolume()),
        __file__,
        XML_FILE,
    )

    problem = run_system(ComputeFuselageVolume(), ivc)
    avg_fuselage_depth = problem.get_val("data:geometry:fuselage:volume", units="m**3")
    assert avg_fuselage_depth == pytest.approx(7.95, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_geometry_wing_toc():
    """Tests computation of the wing ToC (Thickness of Chord)"""

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(ComputeWingToc()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeWingToc(), ivc)
    toc_root = problem["data:geometry:wing:root:thickness_ratio"]
    assert toc_root == pytest.approx(0.186, abs=1e-3)
    toc_kink = problem["data:geometry:wing:kink:thickness_ratio"]
    assert toc_kink == pytest.approx(0.141, abs=1e-3)
    toc_tip = problem["data:geometry:wing:tip:thickness_ratio"]
    assert toc_tip == pytest.approx(0.129, abs=1e-3)


def test_geometry_wing_y():
    """Tests computation of the wing Ys"""

    # Research independent input value in .xml file and add values calculated from other modules
    ivc = get_indep_var_comp(list_inputs(ComputeWingY()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeWingY(), ivc)
    span = problem.get_val("data:geometry:wing:span", units="m")
    assert span == pytest.approx(11.609, abs=1e-3)
    wing_y2 = problem.get_val("data:geometry:wing:root:y", units="m")
    assert wing_y2 == pytest.approx(0.599, abs=1e-3)
    wing_y3 = problem.get_val("data:geometry:wing:kink:y", units="m")
    assert wing_y3 == pytest.approx(0.0, abs=1e-3)  # point 3 is virtual central point
    wing_y4 = problem.get_val("data:geometry:wing:tip:y", units="m")
    assert wing_y4 == pytest.approx(5.804, abs=1e-3)


def test_geometry_wing_z():
    """Tests computation of the wing Zs"""

    # Research independent input value in .xml file and add values calculated from other modules
    ivc = get_indep_var_comp(list_inputs(ComputeWingZ()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeWingZ(), ivc)
    wing_z2 = problem.get_val("data:geometry:wing:root:z", units="m")
    assert wing_z2 == pytest.approx(0.533, rel=1e-2)
    wing_z4 = problem.get_val("data:geometry:wing:tip:z", units="m")
    assert wing_z4 == pytest.approx(0.028, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_geometry_wing_l1_l4():
    """Tests computation of the wing chords (l1 and l4)"""

    # Research independent input value in .xml file and add values calculated from other modules
    ivc = get_indep_var_comp(list_inputs(ComputeWingL1AndL4()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeWingL1AndL4(), ivc)
    wing_l1 = problem.get_val("data:geometry:wing:root:virtual_chord", units="m")
    assert wing_l1 == pytest.approx(1.455, abs=1e-3)
    wing_l4 = problem.get_val("data:geometry:wing:tip:chord", units="m")
    assert wing_l4 == pytest.approx(1.455, abs=1e-3)


def test_geometry_wing_l2_l3():
    """Tests computation of the wing chords (l2 and l3)"""

    # Research independent input value in .xml file and add values calculated from other modules
    ivc = get_indep_var_comp(list_inputs(ComputeWingL2AndL3()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeWingL2AndL3(), ivc)
    wing_l2 = problem.get_val("data:geometry:wing:root:chord", units="m")
    assert wing_l2 == pytest.approx(1.455, abs=1e-2)
    wing_l3 = problem.get_val("data:geometry:wing:kink:chord", units="m")
    assert wing_l3 == pytest.approx(
        1.455, abs=1e-2
    )  # point 3 and 2 equal (previous version ignored)


def test_geometry_wing_x():
    """Tests computation of the wing Xs"""

    # Research independent input value in .xml file and add values calculated from other modules
    ivc = get_indep_var_comp(list_inputs(ComputeWingX()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeWingX(), ivc)
    wing_x3 = problem.get_val("data:geometry:wing:kink:leading_edge:x:local", units="m")
    assert wing_x3 == pytest.approx(0.0, abs=1e-3)
    wing_x4 = problem.get_val("data:geometry:wing:tip:leading_edge:x:local", units="m")
    assert wing_x4 == pytest.approx(0.0, abs=1e-3)


def test_geometry_wing_x_absolute():
    """Tests computation of the wing absolute Xs"""

    # Research independent input value in .xml file and add values calculated from other modules
    ivc = get_indep_var_comp(list_inputs(ComputeWingXAbsolute()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeWingXAbsolute(), ivc)
    wing_x0_abs = problem.get_val("data:geometry:wing:MAC:leading_edge:x:absolute", units="m")
    assert wing_x0_abs == pytest.approx(3.091, abs=1e-3)
    wing_x4_abs = problem.get_val("data:geometry:wing:tip:leading_edge:x:absolute", units="m")
    assert wing_x4_abs == pytest.approx(3.091, abs=1e-3)

    problem.check_partials(compact_print=True)


def test_geometry_wing_b50():
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


def test_geometry_wing_mac():
    """Tests computation of the wing mean aerodynamic chord"""

    # Research independent input value in .xml file and add values calculated from other modules
    ivc = get_indep_var_comp(list_inputs(ComputeWingMAC()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeWingMAC(), ivc)
    wing_l0 = problem.get_val("data:geometry:wing:MAC:length", units="m")
    assert wing_l0 == pytest.approx(1.453, abs=1e-3)
    wing_x0 = problem.get_val("data:geometry:wing:MAC:leading_edge:x:local", units="m")
    assert wing_x0 == pytest.approx(0.0, abs=1e-3)
    wing_y0 = problem.get_val("data:geometry:wing:MAC:y", units="m")
    assert wing_y0 == pytest.approx(2.899, abs=1e-3)


def test_geometry_wing_sweep():
    """Tests computation of the wing sweeps"""

    # Define input values calculated from other modules
    ivc = get_indep_var_comp(list_inputs(ComputeWingSweep()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeWingSweep(), ivc)
    sweep_0 = problem.get_val("data:geometry:wing:sweep_0", units="deg")
    assert sweep_0 == pytest.approx(0.0, abs=1e-1)
    sweep_50 = problem.get_val("data:geometry:wing:sweep_50", units="deg")
    assert sweep_50 == pytest.approx(0.0, abs=1e-1)
    sweep_100_inner = problem.get_val("data:geometry:wing:sweep_100_inner", units="deg")
    assert sweep_100_inner == pytest.approx(0.0, abs=1e-1)
    sweep_100_outer = problem.get_val("data:geometry:wing:sweep_100_outer", units="deg")
    assert sweep_100_outer == pytest.approx(0.0, abs=1e-1)


def test_geometry_wing_wet_area():
    """Tests computation of the wing wet area"""

    # Research independent input value in .xml file and add values calculated from other modules
    ivc = get_indep_var_comp(list_inputs(ComputeWingWetArea()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeWingWetArea(), ivc)
    area_pf = problem.get_val("data:geometry:wing:outer_area", units="m**2")
    assert area_pf == pytest.approx(15.145, abs=1e-1)
    wet_area = problem.get_val("data:geometry:wing:wet_area", units="m**2")
    assert wet_area == pytest.approx(32.411, abs=1e-3)


def test_geometry_wing_mfw_simple():
    """Tests computation of the wing max fuel weight"""

    # Research independent input value in .xml file and add values calculated from other modules
    ivc = get_indep_var_comp(list_inputs(ComputeMFWSimple()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeMFWSimple(), ivc)
    mfw = problem.get_val("data:weight:aircraft:MFW", units="kg")
    assert mfw == pytest.approx(583.897, abs=1e-2)


def test_geometry_wing_mfw_advanced():
    """Tests computation of the wing max fuel weight"""

    # Research independent input value in .xml file and add values calculated from other modules
    ivc = get_indep_var_comp(list_inputs(ComputeMFWAdvanced()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeMFWAdvanced(), ivc)
    mfw = problem.get_val("data:weight:aircraft:MFW", units="kg")
    assert mfw == pytest.approx(304.73, abs=1e-2)


def test_geometry_nacelle():
    """Tests computation of the nacelle and pylons component"""

    # Research independent input value in .xml file and add values calculated from other modules
    ivc = get_indep_var_comp(
        list_inputs(ComputeNacelleDimension(propulsion_id=ENGINE_WRAPPER)), __file__, XML_FILE
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeNacelleDimension(propulsion_id=ENGINE_WRAPPER), ivc)
    nacelle_length = problem.get_val("data:geometry:propulsion:nacelle:length", units="m")
    assert nacelle_length == pytest.approx(1.237, abs=1e-3)
    nacelle_height = problem.get_val("data:geometry:propulsion:nacelle:height", units="m")
    assert nacelle_height == pytest.approx(0.623, abs=1e-3)
    nacelle_width = problem.get_val("data:geometry:propulsion:nacelle:width", units="m")
    assert nacelle_width == pytest.approx(0.929, abs=1e-3)
    nacelle_wet_area = problem.get_val("data:geometry:propulsion:nacelle:wet_area", units="m**2")
    assert nacelle_wet_area == pytest.approx(3.841, abs=1e-3)
    nacelle_master_cross_section = problem.get_val(
        "data:geometry:propulsion:nacelle:master_cross_section", units="m**2"
    )
    assert nacelle_master_cross_section == pytest.approx(0.57890572, abs=1e-3)


def test_position_nacelle():
    """Tests computation of the nacelle and pylons component"""

    # Research independent input value in .xml file and add values calculated from other modules
    ivc = get_indep_var_comp(list_inputs(ComputeNacellePosition()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeNacellePosition(), ivc)
    y_nacelle = problem.get_val("data:geometry:propulsion:nacelle:y", units="m")
    y_nacelle_result = 1.972
    assert abs(y_nacelle - y_nacelle_result) < 1e-3
    x_nacelle = problem.get_val("data:geometry:propulsion:nacelle:x", units="m")
    x_nacelle_result = 3.092
    assert abs(x_nacelle - x_nacelle_result) < 1e-3


def test_position_propeller():
    """Tests computation of the nacelle and pylons component"""

    # Research independent input value in .xml file and add values calculated from other modules
    ivc = get_indep_var_comp(list_inputs(ComputePropellerPosition()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputePropellerPosition(), ivc)
    x_prop_from_le = problem.get_val("data:geometry:propulsion:nacelle:from_LE", units="m")
    x_prop_from_le_result = 0.1954
    assert abs(x_prop_from_le - x_prop_from_le_result) < 1e-3


def test_installation_effect_propeller():
    """Tests computation propeller effective advance ratio factor computation"""

    # Research independent input value in .xml file and add values calculated from other modules
    ivc = get_indep_var_comp(list_inputs(ComputePropellerInstallationEffect()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputePropellerInstallationEffect(), ivc)
    prop_installation_effect = problem.get_val(
        "data:aerodynamics:propeller:installation_effect:effective_advance_ratio"
    )
    assert prop_installation_effect == pytest.approx(0.949, abs=1e-3)


def test_landing_gear_geometry():

    # Research independent input value in .xml file and add values calculated from other modules
    ivc = get_indep_var_comp(list_inputs(ComputeLGGeometry()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeLGGeometry(), ivc)

    lg_height = problem.get_val("data:geometry:landing_gear:height", units="m")
    assert lg_height == pytest.approx(0.791, abs=1e-3)
    lg_position = problem.get_val("data:geometry:landing_gear:y", units="m")
    assert lg_position == pytest.approx(1.548, abs=1e-3)


def test_geometry_total_area():
    """Tests computation of the total area"""

    # Research independent input value in .xml file and add values calculated from other modules
    ivc = get_indep_var_comp(list_inputs(ComputeTotalArea()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeTotalArea(), ivc)
    total_surface = problem.get_val("data:geometry:aircraft:wet_area", units="m**2")
    assert total_surface == pytest.approx(82.216, abs=1e-3)


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
    assert total_surface == pytest.approx(80.932, abs=1e-3)
