"""
Main components for mass breakdown
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

import openmdao.api as om

from fastga.models.weight.mass_breakdown.a_airframe import (
    ComputeWingWeight,
    ComputeFuselageWeight,
    ComputeFuselageWeightRaymer,
    ComputeTailWeight,
    ComputeFlightControlsWeight,
    ComputeLandingGearWeight,
)
from fastga.models.weight.mass_breakdown.b_propulsion import (
    ComputeEngineWeight,
    ComputeFuelLinesWeight,
    ComputeUnusableFuelWeight,
)
from fastga.models.weight.mass_breakdown.c_systems import (
    ComputePowerSystemsWeight,
    ComputeLifeSupportSystemsWeight,
    ComputeNavigationSystemsWeight,
)
from fastga.models.weight.mass_breakdown.d_furniture import ComputePassengerSeatsWeight
from fastga.models.weight.mass_breakdown.payload import ComputePayload
from fastga.models.weight.mass_breakdown.update_mlw_and_mzfw import UpdateMLWandMZFW

from fastga.models.options import PAYLOAD_FROM_NPAX


class MassBreakdown(om.Group):
    """
    Computes analytically the mass of each part of the aircraft, and the resulting sum,
    the Overall Weight Empty (OWE).

    Some models depend on MZFW (Max Zero Fuel Weight) and MTOW (Max TakeOff Weight),
    which depend on OWE.

    This model cycles for having consistent OWE, MZFW and MTOW based on MFW.

    Options:
    - payload_from_npax: If True (default), payload masses will be computed from NPAX, if False
                         design payload mass and maximum payload mass must be provided.
    """

    def initialize(self):
        self.options.declare(PAYLOAD_FROM_NPAX, types=bool, default=True)
        self.options.declare("propulsion_id", default="", types=str)

    def setup(self):
        if self.options[PAYLOAD_FROM_NPAX]:
            self.add_subsystem("payload", ComputePayload(), promotes=["*"])
        self.add_subsystem(
            "owe",
            ComputeOperatingWeightEmpty(propulsion_id=self.options["propulsion_id"]),
            promotes=["*"],
        )
        self.add_subsystem("update_mzfw_and_mlw", UpdateMLWandMZFW(), promotes=["*"])

        # Solvers setup
        self.nonlinear_solver = om.NonlinearBlockGS()
        self.nonlinear_solver.options["debug_print"] = True
        self.nonlinear_solver.options["err_on_non_converge"] = True
        self.nonlinear_solver.options["iprint"] = 0
        self.nonlinear_solver.options["maxiter"] = 50
        # self.nonlinear_solver.options["reraise_child_analysiserror"] = True
        # self.nonlinear_solver.options["rtol"] = 1e-3

        self.linear_solver = om.LinearBlockGS()
        self.linear_solver.options["err_on_non_converge"] = True
        self.linear_solver.options["iprint"] = 0
        self.linear_solver.options["maxiter"] = 10
        # self.linear_solver.options["rtol"] = 1e-3


class ComputeOperatingWeightEmpty(om.Group):
    """ Operating Empty Weight (OEW) estimation

    This group aggregates weight from all components of the aircraft.
    """

    def initialize(self):
        self.options.declare("propulsion_id", default="", types=str)

    def setup(self):
        # Airframe
        self.add_subsystem("wing_weight", ComputeWingWeight(), promotes=["*"])
        self.add_subsystem("fuselage_weight", ComputeFuselageWeight(), promotes=["*"])
        self.add_subsystem("fuselage_weight_raymer", ComputeFuselageWeightRaymer(), promotes=["*"])
        self.add_subsystem("empennage_weight", ComputeTailWeight(), promotes=["*"])
        self.add_subsystem("flight_controls_weight", ComputeFlightControlsWeight(), promotes=["*"])
        self.add_subsystem("landing_gear_weight", ComputeLandingGearWeight(), promotes=["*"])
        self.add_subsystem(
            "engine_weight",
            ComputeEngineWeight(propulsion_id=self.options["propulsion_id"]),
            promotes=["*"],
        )
        self.add_subsystem(
            "unusable_fuel",
            ComputeUnusableFuelWeight(propulsion_id=self.options["propulsion_id"]),
            promotes=["*"],
        )
        self.add_subsystem("fuel_lines_weight", ComputeFuelLinesWeight(), promotes=["*"])
        self.add_subsystem(
            "navigation_systems_weight", ComputeNavigationSystemsWeight(), promotes=["*"]
        )
        self.add_subsystem("power_systems_weight", ComputePowerSystemsWeight(), promotes=["*"])
        self.add_subsystem(
            "life_support_systems_weight", ComputeLifeSupportSystemsWeight(), promotes=["*"]
        )
        self.add_subsystem("passenger_seats_weight", ComputePassengerSeatsWeight(), promotes=["*"])

        # Make additions
        airframe_sum = om.AddSubtractComp()
        airframe_sum.add_equation(
            "data:weight:airframe:mass",
            [
                "data:weight:airframe:wing:mass",
                "data:weight:airframe:fuselage:mass",
                "data:weight:airframe:horizontal_tail:mass",
                "data:weight:airframe:vertical_tail:mass",
                "data:weight:airframe:flight_controls:mass",
                "data:weight:airframe:landing_gear:main:mass",
                "data:weight:airframe:landing_gear:front:mass",
            ],
            units="kg",
            desc="Mass of airframe",
        )
        self.add_subsystem(
            "airframe_weight_sum", airframe_sum, promotes=["*"],
        )

        propulsion_sum = om.AddSubtractComp()
        propulsion_sum.add_equation(
            "data:weight:propulsion:mass",
            ["data:weight:propulsion:engine:mass", "data:weight:propulsion:fuel_lines:mass",],
            units="kg",
            desc="Mass of the propulsion system",
        )
        self.add_subsystem(
            "propulsion_weight_sum", propulsion_sum, promotes=["*"],
        )

        systems_sum = om.AddSubtractComp()
        systems_sum.add_equation(
            "data:weight:systems:mass",
            [
                "data:weight:systems:power:electric_systems:mass",
                "data:weight:systems:power:hydraulic_systems:mass",
                "data:weight:systems:life_support:air_conditioning:mass",
                "data:weight:systems:life_support:insulation:mass",
                "data:weight:systems:life_support:de_icing:mass",
                "data:weight:systems:life_support:internal_lighting:mass",
                "data:weight:systems:life_support:seat_installation:mass",
                "data:weight:systems:life_support:fixed_oxygen:mass",
                "data:weight:systems:life_support:security_kits:mass",
                "data:weight:systems:navigation:mass",
            ],
            units="kg",
            desc="Mass of aircraft systems",
        )
        self.add_subsystem(
            "systems_weight_sum", systems_sum, promotes=["*"],
        )

        furniture_sum = om.AddSubtractComp()
        furniture_sum.add_equation(
            "data:weight:furniture:mass",
            [
                "data:weight:furniture:passenger_seats:mass",
                "data:weight:furniture:passenger_seats:mass",
            ],
            scaling_factors=[0.5, 0.5],
            units="kg",
            desc="Mass of aircraft furniture",
        )
        self.add_subsystem(
            "furniture_weight_sum", furniture_sum, promotes=["*"],
        )

        owe_sum = om.AddSubtractComp()
        owe_sum.add_equation(
            "data:weight:aircraft:OWE",
            [
                "data:weight:airframe:mass",
                "data:weight:propulsion:mass",
                "data:weight:systems:mass",
                "data:weight:furniture:mass",
            ],
            units="kg",
            desc="Mass of aircraft",
        )
        self.add_subsystem(
            "OWE_sum", owe_sum, promotes=["*"],
        )
