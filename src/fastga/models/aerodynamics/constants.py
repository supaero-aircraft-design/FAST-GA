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

POLAR_POINT_COUNT = 150
SPAN_MESH_POINT = 50
MACH_NB_PTS = 5
ENGINE_COUNT = 10
FIRST_INVALID_COEFF = 100.0

SUBMODEL_CD0 = "submodel.aerodynamics.aircraft.cd0"
SUBMODEL_CD0_WING = "submodel.aerodynamics.wing.cd0"
SUBMODEL_CD0_FUSELAGE = "submodel.aerodynamics.fuselage.cd0"
SUBMODEL_CD0_HT = "submodel.aerodynamics.horizontal_tail.cd0"
SUBMODEL_CD0_VT = "submodel.aerodynamics.vertical_tail.cd0"
SUBMODEL_CD0_NACELLE = "submodel.aerodynamics.nacelle.cd0"
SUBMODEL_CD0_LANDING_GEAR = "submodel.aerodynamics.landing_gear.cd0"
SUBMODEL_CD0_OTHER = "submodel.aerodynamics.other.cd0"
SUBMODEL_CD0_SUM = "submodel.aerodynamics.sum.cd0"
SUBMODEL_AIRFOIL_LIFT_SLOPE = "submodel.aerodynamics.airfoil.all.lift_curve_slope"
SUBMODEL_DELTA_HIGH_LIFT = "submodel.aerodynamics.high_lift.delta"
SUBMODEL_DELTA_ELEVATOR = "submodel.aerodynamics.elevator.delta"
SUBMODEL_CL_EXTREME = "submodel.aerodynamics.aircraft.extreme_lift_coefficient"
SUBMODEL_CL_EXTREME_CLEAN_WING = "submodel.aerodynamics.wing.extreme_lift_coefficient.clean"
SUBMODEL_CL_EXTREME_CLEAN_HT = (
    "submodel.aerodynamics.horizontal_tail.extreme_lift_coefficient.clean"
)
SUBMODEL_CL_ALPHA_VT = "submodel.aerodynamics.vertical_tail.lift_curve_slope"
SUBMODEL_CY_RUDDER = "submodel.aerodynamics.rudder.yawing_moment"
SUBMODEL_EFFECTIVE_EFFICIENCY_PROPELLER = "submodel.aerodynamics.propeller.effective_efficiency"
SUBMODEL_HINGE_MOMENTS_TAIL_2D = "submodel.aerodynamics.tail.hinge_moments.2d"
SUBMODEL_HINGE_MOMENTS_TAIL_3D = "submodel.aerodynamics.tail.hinge_moments.3d"
SUBMODEL_HINGE_MOMENTS_TAIL = "submodel.aerodynamics.tail.hinge_moments"
SUBMODEL_MAX_L_D = "submodel.aerodynamics.aircraft.l_d_max"
SUBMODEL_CN_BETA_FUSELAGE = "submodel.aerodynamics.fuselage.yawing_moment"
SUBMODEL_CM_ALPHA_FUSELAGE = "submodel.aerodynamics.fuselage.pitching_moment"
SUBMODEL_VH = "submodel.aerodynamics.aircraft.max_level_speed"
SUBMODEL_THRUST_POWER_SLIPSTREAM = "submodel.aerodynamics.wing.slipstream.thrust_power_computation"
