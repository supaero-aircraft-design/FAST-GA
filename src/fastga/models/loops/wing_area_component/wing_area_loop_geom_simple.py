"""
Computation of wing area update and constraints based on the amount of fuel in the wing with
simple computation.
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

import warnings

import numpy as np
import openmdao.api as om

import fastoad.api as oad

from ..constants import SUBMODEL_WING_AREA_GEOM_LOOP, SUBMODEL_WING_AREA_GEOM_CONS

oad.RegisterSubmodel.active_models[
    SUBMODEL_WING_AREA_GEOM_LOOP
] = "fastga.submodel.loop.wing_area.update.geom.simple"
oad.RegisterSubmodel.active_models[
    SUBMODEL_WING_AREA_GEOM_CONS
] = "fastga.submodel.loop.wing_area.constraint.geom.simple"


@oad.RegisterSubmodel(
    SUBMODEL_WING_AREA_GEOM_LOOP, "fastga.submodel.loop.wing_area.update.geom.simple"
)
class UpdateWingAreaGeomSimple(om.ExplicitComponent):
    """
    Computes needed wing area to be able to load enough fuel to achieve the sizing mission.
    """

    def setup(self):
        self.add_input("data:mission:sizing:fuel", val=np.nan, units="kg")
        self.add_input("data:propulsion:fuel_type", val=np.nan)
        self.add_input("data:geometry:wing:root:chord", val=np.nan, units="m")
        self.add_input("data:geometry:wing:tip:chord", val=np.nan, units="m")
        self.add_input("data:geometry:wing:root:thickness_ratio", val=np.nan)
        self.add_input("data:geometry:wing:tip:thickness_ratio", val=np.nan)

        self.add_output("wing_area", val=10.0, units="m**2")

        self.declare_partials(
            "wing_area",
            [
                "data:mission:sizing:fuel",
                "data:geometry:wing:root:chord",
                "data:geometry:wing:tip:chord",
                "data:geometry:wing:root:thickness_ratio",
                "data:geometry:wing:tip:thickness_ratio",
            ],
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        mfw_mission = inputs["data:mission:sizing:fuel"]
        fuel_type = inputs["data:propulsion:fuel_type"]
        root_chord = inputs["data:geometry:wing:root:chord"]
        tip_chord = inputs["data:geometry:wing:tip:chord"]
        root_thickness_ratio = inputs["data:geometry:wing:root:thickness_ratio"]
        tip_thickness_ratio = inputs["data:geometry:wing:tip:thickness_ratio"]

        if fuel_type == 1.0:
            m_vol_fuel = 730  # gasoline volume-mass [kg/m**3], cold worst case
        elif fuel_type == 2.0:
            m_vol_fuel = 860  # gasoil volume-mass [kg/m**3], cold worst case
        else:
            m_vol_fuel = 730
            warnings.warn("Fuel type %f does not exist, replaced by type 1!" % fuel_type)

        # Tanks are between 1st (30% MAC) and 3rd (60% MAC) longeron: 30% of the wing
        ave_thickness = (
            0.7 * (root_chord * root_thickness_ratio + tip_chord * tip_thickness_ratio) / 2.0
        )
        wing_area_mission = (mfw_mission / m_vol_fuel) / (0.3 * ave_thickness)

        outputs["wing_area"] = wing_area_mission

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        mfw_mission = inputs["data:mission:sizing:fuel"]
        fuel_type = inputs["data:propulsion:fuel_type"]
        root_chord = inputs["data:geometry:wing:root:chord"]
        tip_chord = inputs["data:geometry:wing:tip:chord"]
        root_thickness_ratio = inputs["data:geometry:wing:root:thickness_ratio"]
        tip_thickness_ratio = inputs["data:geometry:wing:tip:thickness_ratio"]

        if fuel_type == 1.0:
            m_vol_fuel = 730  # gasoline volume-mass [kg/m**3], cold worst case
        elif fuel_type == 2.0:
            m_vol_fuel = 860  # gasoil volume-mass [kg/m**3], cold worst case
        else:
            m_vol_fuel = 730
            warnings.warn("Fuel type %f does not exist, replaced by type 1!" % fuel_type)

        ave_thickness = (
            0.7 * (root_chord * root_thickness_ratio + tip_chord * tip_thickness_ratio) / 2.0
        )

        d_avg_th_d_root_c = 0.7 / 2.0 * root_thickness_ratio
        d_avg_th_d_root_tc = 0.7 / 2.0 * root_chord

        d_avg_th_d_tip_c = 0.7 / 2.0 * tip_thickness_ratio
        d_avg_th_d_tip_tc = 0.7 / 2.0 * tip_chord

        d_area_d_avg_th = -(mfw_mission / m_vol_fuel) / (0.3 * ave_thickness ** 2.0)

        partials["wing_area", "data:mission:sizing:fuel"] = (1.0 / m_vol_fuel) / (
            0.3 * ave_thickness
        )
        partials["wing_area", "data:geometry:wing:root:chord"] = d_area_d_avg_th * d_avg_th_d_root_c
        partials["wing_area", "data:geometry:wing:tip:chord"] = d_area_d_avg_th * d_avg_th_d_tip_c
        partials["wing_area", "data:geometry:wing:root:thickness_ratio"] = (
            d_area_d_avg_th * d_avg_th_d_root_tc
        )
        partials["wing_area", "data:geometry:wing:tip:thickness_ratio"] = (
            d_area_d_avg_th * d_avg_th_d_tip_tc
        )


@oad.RegisterSubmodel(
    SUBMODEL_WING_AREA_GEOM_CONS, "fastga.submodel.loop.wing_area.constraint.geom.simple"
)
class ConstraintWingAreaGeomSimple(om.ExplicitComponent):
    """
    Computes the difference between what the wing can store in terms of fuel and the fuel
    needed for the mission to check if the constraints is respected using a simple geometric
    computation.
    """

    def setup(self):

        self.add_input("data:mission:sizing:fuel", val=np.nan, units="kg")
        self.add_input("data:weight:aircraft:MFW", val=np.nan, units="kg")

        self.add_output("data:constraints:wing:additional_fuel_capacity", units="kg")

        self.declare_partials(
            "data:constraints:wing:additional_fuel_capacity",
            ["data:weight:aircraft:MFW", "data:mission:sizing:fuel"],
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        mfw = inputs["data:weight:aircraft:MFW"]
        mission_fuel = inputs["data:mission:sizing:fuel"]

        outputs["data:constraints:wing:additional_fuel_capacity"] = mfw - mission_fuel

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        partials["data:constraints:wing:additional_fuel_capacity", "data:weight:aircraft:MFW"] = 1.0
        partials[
            "data:constraints:wing:additional_fuel_capacity", "data:mission:sizing:fuel"
        ] = -1.0
