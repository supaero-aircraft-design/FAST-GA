"""Computation of the furniture mass."""
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

import fastoad.api as oad

from .constants import SUBMODEL_SEATS_MASS

from ..constants import SUBMODEL_FURNITURE_MASS


@oad.RegisterSubmodel(SUBMODEL_FURNITURE_MASS, "fastga.submodel.weight.mass.furniture.legacy")
class FurnitureWeight(om.Group):
    """
    Computes mass of furniture.
    """

    def setup(self):
        self.add_subsystem(
            "seats_weight",
            oad.RegisterSubmodel.get_submodel(SUBMODEL_SEATS_MASS),
            promotes=["*"],
        )

        weight_sum = om.AddSubtractComp()
        weight_sum.add_equation(
            "data:weight:furniture:mass",
            [
                "data:weight:furniture:passenger_seats:mass",
                "data:weight:furniture:passenger_seats:mass",
            ],
            scaling_factors=[0.5, 0.5],
            units="kg",
            desc="Mass of aircraft furniture",
        )

        self.add_subsystem("furniture_weight_sum", weight_sum, promotes=["*"])
