"""
Computation of wing area
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

import numpy as np
import openmdao.api as om
from scipy.constants import g
import warnings

from fastoad.module_management.service_registry import RegisterOpenMDAOSystem
from fastoad.module_management.constants import ModelDomain


@RegisterOpenMDAOSystem("fastga.loop.wing_area", domain=ModelDomain.OTHER)
class UpdateWingArea(om.Group):
    """
    Computes needed wing area to:
      - have enough lift at required approach speed
      - be able to load enough fuel to achieve the sizing mission
    """

    def setup(self):
        self.add_subsystem("wing_area", _UpdateWingArea(), promotes=["*"])
        self.add_subsystem("constraints", _ComputeWingAreaConstraints(), promotes=["*"])


class _UpdateWingArea(om.ExplicitComponent):
    """ Computation of wing area from needed approach speed and mission fuel """

    def setup(self):
        self.add_input("data:mission:sizing:fuel", val=np.nan, units="kg")
        self.add_input("data:propulsion:IC_engine:fuel_type", val=np.nan)
        self.add_input("data:geometry:wing:root:chord", val=np.nan, units="m")
        self.add_input("data:geometry:wing:tip:chord", val=np.nan, units="m")
        self.add_input("data:geometry:wing:root:thickness_ratio", val=np.nan)
        self.add_input("data:geometry:wing:tip:thickness_ratio", val=np.nan)

        self.add_input("data:TLAR:v_approach", val=np.nan, units="m/s")
        self.add_input("data:weight:aircraft:MLW", val=np.nan, units="kg")
        self.add_input("data:aerodynamics:aircraft:landing:CL_max", val=np.nan)

        self.add_output("data:geometry:wing:area", val=10.0, units="m**2")
        
        self.declare_partials(
            "data:geometry:wing:area",
            [
                "data:mission:sizing:fuel",
                "data:geometry:wing:root:chord",
                "data:geometry:wing:tip:chord",
                "data:geometry:wing:root:thickness_ratio",
                "data:geometry:wing:tip:thickness_ratio",
            ],
            method="fd",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        mfw_mission = inputs["data:mission:sizing:fuel"]
        fuel_type = inputs["data:propulsion:IC_engine:fuel_type"]
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
            warnings.warn("Fuel type {} does not exist, replaced by type 1!".format(fuel_type))

        # Tanks are between 1st (30% MAC) and 3rd (60% MAC) longeron: 30% of the wing
        ave_thichness = 0.7 * (
                root_chord * root_thickness_ratio
                + tip_chord * tip_thickness_ratio
        ) / 2.0
        wing_area_mission = (mfw_mission / m_vol_fuel) / (0.3 * ave_thichness)

        stall_speed = inputs["data:TLAR:v_approach"]/1.3
        mlw = inputs["data:weight:aircraft:MLW"]
        max_cl = inputs["data:aerodynamics:aircraft:landing:CL_max"]
        wing_area_approach = 2 * mlw * g / (stall_speed ** 2) / (1.225 * max_cl)

        outputs["data:geometry:wing:area"] = max(wing_area_mission, wing_area_approach)


class _ComputeWingAreaConstraints(om.ExplicitComponent):
    
    def setup(self):
        self.add_input("data:mission:sizing:fuel", val=np.nan, units="kg")
        self.add_input("data:weight:aircraft:MFW", val=np.nan, units="kg")

        self.add_input("data:TLAR:v_approach", val=np.nan, units="m/s")
        self.add_input("data:weight:aircraft:MLW", val=np.nan, units="kg")
        self.add_input("data:aerodynamics:aircraft:landing:CL_max", val=np.nan)
        self.add_input("data:geometry:wing:area", val=np.nan, units="m**2")

        self.add_output("data:weight:aircraft:additional_fuel_capacity", units="kg")
        self.add_output("data:aerodynamics:aircraft:landing:additional_CL_capacity")

        self.declare_partials(
            "data:weight:aircraft:additional_fuel_capacity",
            ["data:weight:aircraft:MFW", "data:mission:sizing:fuel"],
            method="fd",
        )
        self.declare_partials(
            "data:aerodynamics:aircraft:landing:additional_CL_capacity",
            [
                "data:TLAR:v_approach",
                "data:weight:aircraft:MLW",
                "data:aerodynamics:aircraft:landing:CL_max",
                "data:geometry:wing:area",
            ],
            method="fd",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        
        mfw = inputs["data:weight:aircraft:MFW"]
        mission_fuel = inputs["data:mission:sizing:fuel"]
        v_stall = inputs["data:TLAR:v_approach"]/1.3
        cl_max = inputs["data:aerodynamics:aircraft:landing:CL_max"]
        mlw = inputs["data:weight:aircraft:MLW"]
        wing_area = inputs["data:geometry:wing:area"]

        outputs["data:weight:aircraft:additional_fuel_capacity"] = mfw - mission_fuel
        outputs["data:aerodynamics:aircraft:landing:additional_CL_capacity"] = cl_max - mlw * g / (
            0.5 * 1.225 * v_stall ** 2 * wing_area
        )
