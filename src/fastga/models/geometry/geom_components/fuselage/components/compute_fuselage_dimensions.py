"""
Python module for fuselage dimension calculation, part of the fuselage geometry.
"""

#  This file is part of FAST-OAD_CS23 : A framework for rapid Overall Aircraft Design
#  Copyright (C) 2025  ONERA & ISAE-SUPAERO
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

import numpy as np
import openmdao.api as om

from .compute_aircraft_length import ComputeAircraftLength
from .compute_fuselage_cabin_length import ComputeFuselageCabinLength
from .compute_fuselage_luggage_length import ComputeFuselageLuggageLength
from .compute_fuselage_max_height import ComputeFuselageMaxHeight
from .compute_fuselage_max_width import ComputeFuselageMaxWidth
from .compute_fuselage_npax import ComputeFuselageNPAX
from .compute_fuselage_pax_length import ComputeFuselagePAXLength
from .compute_fuselage_length_fd import ComputeFuselageLengthFD
from .compute_fuselage_length_fl import ComputeFuselageLengthFL
from .compute_fuselage_nose_length import ComputeFuselageNoseLength
from .compute_fuselage_rear_length import ComputeFuselageRearLength


class ComputeFuselageGeometryBasic(om.ExplicitComponent):
    """
    Geometry of fuselage - Cabin length defined with total fuselage length input (no sizing).
    """

    # pylint: disable=missing-function-docstring
    # Overriding OpenMDAO setup
    def setup(self):
        self.add_input("data:geometry:fuselage:length", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:front_length", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:rear_length", val=np.nan, units="m")

        self.add_output("data:geometry:cabin:length", units="m")

        self.declare_partials("*", "data:geometry:fuselage:length", val=1.0)
        self.declare_partials(
            "*",
            ["data:geometry:fuselage:front_length", "data:geometry:fuselage:rear_length"],
            val=-1.0,
        )

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute, not all arguments are used
    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        fus_length = inputs["data:geometry:fuselage:length"]
        lav = inputs["data:geometry:fuselage:front_length"]
        lar = inputs["data:geometry:fuselage:rear_length"]

        # Cabin total length
        cabin_length = fus_length - (lav + lar)

        outputs["data:geometry:cabin:length"] = cabin_length


# pylint: disable=too-few-public-methods
class ComputeFuselageGeometryCabinSizingFD(om.Group):
    # TODO: Document equations. Cite sources
    """
    Geometry of fuselage - Cabin is sized based on layout (seats, aisle...) and HTP/VTP position
    (Fixed tail Distance).
    """

    # pylint: disable=missing-function-docstring
    # Overriding OpenMDAO setup
    def setup(self):
        self.add_subsystem(
            "number_of_passenger",
            ComputeFuselageNPAX(),
            promotes=["*"],
        )
        self.add_subsystem(
            "passenger_area_length",
            ComputeFuselagePAXLength(),
            promotes=["*"],
        )
        self.add_subsystem(
            "maximum_width",
            ComputeFuselageMaxWidth(),
            promotes=["*"],
        )
        self.add_subsystem(
            "maximum_height",
            ComputeFuselageMaxHeight(),
            promotes=["*"],
        )
        self.add_subsystem(
            "luggage_area_length",
            ComputeFuselageLuggageLength(),
            promotes=["*"],
        )
        self.add_subsystem(
            "cabin_length",
            ComputeFuselageCabinLength(),
            promotes=["*"],
        )
        self.add_subsystem(
            "nose_length",
            ComputeFuselageNoseLength(),
            promotes=["*"],
        )
        self.add_subsystem(
            "fuselage_length",
            ComputeFuselageLengthFD(),
            promotes=["*"],
        )
        self.add_subsystem(
            "aircraft_length",
            ComputeAircraftLength(),
            promotes=["*"],
        )
        self.add_subsystem(
            "fuselage_rear_length",
            ComputeFuselageRearLength(),
            promotes=["*"],
        )


# pylint: disable=too-few-public-methods
class ComputeFuselageGeometryCabinSizingFL(om.Group):
    # TODO: Document equations. Cite sources
    """
    Geometry of fuselage - Cabin is sized based on layout (seats, aisle...) and additional rear
    length (Fixed Length).
    """

    # pylint: disable=missing-function-docstring
    # Overriding OpenMDAO setup
    def setup(self):
        self.add_subsystem(
            "number_of_passenger",
            ComputeFuselageNPAX(),
            promotes=["*"],
        )
        self.add_subsystem(
            "passenger_area_length",
            ComputeFuselagePAXLength(),
            promotes=["*"],
        )
        self.add_subsystem(
            "maximum_width",
            ComputeFuselageMaxWidth(),
            promotes=["*"],
        )
        self.add_subsystem(
            "maximum_height",
            ComputeFuselageMaxHeight(),
            promotes=["*"],
        )
        self.add_subsystem(
            "luggage_area_length",
            ComputeFuselageLuggageLength(),
            promotes=["*"],
        )
        self.add_subsystem(
            "cabin_length",
            ComputeFuselageCabinLength(),
            promotes=["*"],
        )
        self.add_subsystem(
            "nose_length",
            ComputeFuselageNoseLength(),
            promotes=["*"],
        )
        self.add_subsystem(
            "fuselage_length",
            ComputeFuselageLengthFL(),
            promotes=["*"],
        )
