"""Estimation of max fuel weight."""
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

import numpy as np

import warnings

from openmdao.core.explicitcomponent import ExplicitComponent

import fastoad.api as oad

from ...constants import SUBMODEL_MFW

oad.RegisterSubmodel.active_models[SUBMODEL_MFW] = "fastga.submodel.geometry.mfw.legacy"


@oad.RegisterSubmodel(SUBMODEL_MFW, "fastga.submodel.geometry.mfw.legacy")
class ComputeMFWSimple(ExplicitComponent):
    """Max fuel weight estimation based o RAYMER table 10.5 p269."""

    def setup(self):

        self.add_input("data:propulsion:fuel_type", val=np.nan)
        self.add_input("data:geometry:wing:area", val=np.nan, units="m**2")
        self.add_input("data:geometry:wing:root:chord", val=np.nan, units="m")
        self.add_input("data:geometry:wing:tip:chord", val=np.nan, units="m")
        self.add_input("data:geometry:wing:root:thickness_ratio", val=np.nan)
        self.add_input("data:geometry:wing:tip:thickness_ratio", val=np.nan)

        self.add_output("data:weight:aircraft:MFW", units="kg")

        self.declare_partials(
            "data:weight:aircraft:MFW",
            [
                "data:geometry:wing:area",
                "data:geometry:wing:root:chord",
                "data:geometry:wing:tip:chord",
                "data:geometry:wing:root:thickness_ratio",
                "data:geometry:wing:tip:thickness_ratio",
            ],
            method="fd",
        )

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
