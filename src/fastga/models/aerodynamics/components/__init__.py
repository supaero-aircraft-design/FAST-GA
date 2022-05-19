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

from .airfoil_lift_curve_slope import ComputeAirfoilLiftCurveSlope
from .clalpha_vt import ComputeClAlphaVT
from .compute_L_D_max import ComputeLDMax
from .compute_cl_extreme import ComputeAircraftMaxCl
from .compute_cl_extreme_htp import ComputeExtremeCLHtp
from .compute_cl_extreme_wing import ComputeExtremeCLWing
from .compute_cm_alpha_fus import ComputeFuselagePitchingMoment
from .compute_cnbeta_fuselage import ComputeCnBetaFuselage
from .compute_cy_rudder import ComputeCyDeltaRudder
from .compute_effective_efficiency_prop import ComputeEffectiveEfficiencyPropeller
from .compute_equilibrated_polar import ComputeEquilibratedPolar
from .compute_non_equilibrated_polar import ComputeNonEquilibratedPolar
from .compute_reynolds import ComputeUnitReynolds
from .compute_vn import ComputeVNAndVH, ComputeVN
from .high_lift_aero import ComputeDeltaHighLift
from .elevator_aero import ComputeDeltaElevator
from .hinge_moments_elevator import (
    Compute2DHingeMomentsTail,
    Compute3DHingeMomentsTail,
    ComputeHingeMomentsTail,
)
from .mach_interpolation import ComputeMachInterpolation
