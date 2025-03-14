"""
Python module for maximum fuel weight calculation with simple approach, part of the geometry
component.
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

import warnings
import numpy as np
import openmdao.api as om
import fastoad.api as oad

from ...constants import SERVICE_MFW, SUBMODEL_MFW_LEGACY

oad.RegisterSubmodel.active_models[SERVICE_MFW] = SUBMODEL_MFW_LEGACY


@oad.RegisterSubmodel(SERVICE_MFW, SUBMODEL_MFW_LEGACY)
class ComputeMFWSimple(om.ExplicitComponent):
    """Max fuel weight estimation based on RAYMER table 10.5 p269."""

    # pylint: disable=missing-function-docstring
    # Overriding OpenMDAO setup
    def setup(self):
        self.add_input("data:propulsion:fuel_type", val=np.nan)
        self.add_input("data:geometry:wing:area", val=np.nan, units="m**2")
        self.add_input("data:geometry:wing:root:chord", val=np.nan, units="m")
        self.add_input("data:geometry:wing:tip:chord", val=np.nan, units="m")
        self.add_input("data:geometry:wing:root:thickness_ratio", val=np.nan)
        self.add_input("data:geometry:wing:tip:thickness_ratio", val=np.nan)

        self.add_output("data:weight:aircraft:MFW", units="kg")

        self.declare_partials(
            "*",
            [
                "data:geometry:wing:area",
                "data:geometry:wing:root:chord",
                "data:geometry:wing:tip:chord",
                "data:geometry:wing:root:thickness_ratio",
                "data:geometry:wing:tip:thickness_ratio",
            ],
            method="exact",
        )

        self.declare_partials("*", "data:propulsion:fuel_type", method="fd")

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute, not all arguments are used
    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        fuel_type = inputs["data:propulsion:fuel_type"]
        wing_area = inputs["data:geometry:wing:area"]
        root_chord = inputs["data:geometry:wing:root:chord"]
        tip_chord = inputs["data:geometry:wing:tip:chord"]
        root_thickness_ratio = inputs["data:geometry:wing:root:thickness_ratio"]
        tip_thickness_ratio = inputs["data:geometry:wing:tip:thickness_ratio"]

        if fuel_type == 1.0:
            m_vol_fuel = 718.9  # gasoline volume-mass [kg/m**3], cold worst case, Avgas
        elif fuel_type == 2.0:
            m_vol_fuel = 860.0  # Diesel volume-mass [kg/m**3], cold worst case
        elif fuel_type == 3.0:
            m_vol_fuel = 804.0  # Jet-A1 volume mass [kg/m**3], cold worst case
        else:
            m_vol_fuel = 718.9
            warnings.warn("Fuel type {} does not exist, replaced by type 1!".format(fuel_type))

        # Tanks are between 1st (30% MAC) and 3rd (60% MAC) longeron: 30% of the wing
        ave_thickness = (
            0.7 * (root_chord * root_thickness_ratio + tip_chord * tip_thickness_ratio) / 2.0
        )
        mfv = 0.3 * wing_area * ave_thickness
        mfw = mfv * m_vol_fuel

        outputs["data:weight:aircraft:MFW"] = mfw

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute_partials, not all arguments are used
    def compute_partials(self, inputs, partials, discrete_inputs=None):
        fuel_type = inputs["data:propulsion:fuel_type"]
        wing_area = inputs["data:geometry:wing:area"]
        root_chord = inputs["data:geometry:wing:root:chord"]
        tip_chord = inputs["data:geometry:wing:tip:chord"]
        root_thickness_ratio = inputs["data:geometry:wing:root:thickness_ratio"]
        tip_thickness_ratio = inputs["data:geometry:wing:tip:thickness_ratio"]

        if fuel_type == 1.0:
            m_vol_fuel = 718.9  # gasoline volume-mass [kg/m**3], cold worst case, Avgas
        elif fuel_type == 2.0:
            m_vol_fuel = 860.0  # Diesel volume-mass [kg/m**3], cold worst case
        elif fuel_type == 3.0:
            m_vol_fuel = 804.0  # Jet-A1 volume mass [kg/m**3], cold worst case
        else:
            m_vol_fuel = 718.9

        ave_thickness = (
            0.7 * (root_chord * root_thickness_ratio + tip_chord * tip_thickness_ratio) / 2.0
        )
        partials["data:weight:aircraft:MFW", "data:geometry:wing:area"] = (
            0.3 * ave_thickness * m_vol_fuel
        )
        partials["data:weight:aircraft:MFW", "data:geometry:wing:root:chord"] = (
            0.105 * m_vol_fuel * root_thickness_ratio * wing_area
        )
        partials["data:weight:aircraft:MFW", "data:geometry:wing:tip:chord"] = (
            0.105 * m_vol_fuel * tip_thickness_ratio * wing_area
        )
        partials["data:weight:aircraft:MFW", "data:geometry:wing:root:thickness_ratio"] = (
            0.105 * m_vol_fuel * root_chord * wing_area
        )
        partials["data:weight:aircraft:MFW", "data:geometry:wing:tip:thickness_ratio"] = (
            0.105 * m_vol_fuel * tip_chord * wing_area
        )
