"""
Computes the structural loads on the wing of the aircraft in the most stringent case according
to aero-structural loads.
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

import numpy as np
import openmdao.api as om

import fastoad.api as oad

from .aerostructural_loads import AerostructuralLoad, SPAN_MESH_POINT_LOADS
from .constants import SUBMODEL_STRUCTURAL_LOADS


@oad.RegisterSubmodel(SUBMODEL_STRUCTURAL_LOADS, "fastga.submodel.loads.wings.structural.legacy")
class StructuralLoads(om.ExplicitComponent):
    def setup(self):

        self.add_input("data:loads:max_shear:load_factor", val=np.nan)
        self.add_input("data:loads:max_rbm:load_factor", val=np.nan)
        self.add_input(
            "data:aerodynamics:wing:low_speed:Y_vector",
            val=np.nan,
            shape_by_conn=True,
            units="m",
        )
        self.add_input(
            "data:aerodynamics:wing:low_speed:chord_vector",
            val=np.nan,
            shape_by_conn=True,
            copy_shape="data:aerodynamics:wing:low_speed:Y_vector",
            units="m",
        )
        # This add_input is needed because in the other module of the wing load computation,
        # the shape of this vector is copied based on the Y_vector and not having it here would
        # cause the code to crash.
        self.add_input(
            "data:aerodynamics:wing:low_speed:CL_vector",
            val=np.nan,
            shape_by_conn=True,
            copy_shape="data:aerodynamics:wing:low_speed:Y_vector",
        )

        self.add_input("data:weight:propulsion:engine:mass", val=np.nan, units="kg")
        self.add_input("data:weight:airframe:landing_gear:main:mass", val=np.nan, units="kg")
        self.add_input("data:weight:airframe:wing:mass", val=np.nan, units="kg")
        self.add_input("data:weight:aircraft_empty:CG:z", val=np.nan, units="m")

        self.add_input("data:geometry:landing_gear:height", val=np.nan, units="m")
        self.add_input("data:geometry:landing_gear:y", val=np.nan, units="m")
        self.add_input("data:geometry:landing_gear:type", val=np.nan)
        self.add_input("data:geometry:wing:root:chord", val=np.nan, units="m")
        self.add_input("data:geometry:wing:tip:chord", val=np.nan, units="m")
        self.add_input("data:geometry:wing:root:y", val=np.nan, units="m")
        self.add_input("data:geometry:wing:tip:y", val=np.nan, units="m")
        self.add_input("data:geometry:wing:root:thickness_ratio", val=np.nan)
        self.add_input("data:geometry:wing:tip:thickness_ratio", val=np.nan)
        self.add_input("data:geometry:wing:span", val=np.nan, units="m")
        self.add_input(
            "data:geometry:propulsion:engine:y_ratio",
            shape_by_conn=True,
        )
        self.add_input("data:geometry:propulsion:engine:layout", val=np.nan)
        self.add_input("data:geometry:propulsion:engine:count", val=np.nan)
        self.add_input("data:geometry:propulsion:nacelle:width", val=np.nan, units="m")
        self.add_input("data:geometry:propulsion:tank:y_ratio_tank_end", val=np.nan)
        self.add_input("data:geometry:propulsion:tank:y_ratio_tank_beginning", val=np.nan)
        self.add_input("data:geometry:propulsion:tank:LE_chord_percentage", val=np.nan)
        self.add_input("data:geometry:propulsion:tank:TE_chord_percentage", val=np.nan)
        self.add_input("data:geometry:flap:chord_ratio", val=np.nan)
        self.add_input("data:geometry:wing:aileron:chord_ratio", val=np.nan)

        self.add_input(
            "data:weight:airframe:wing:punctual_mass:y_ratio",
            shape_by_conn=True,
            val=0.0,
        )
        self.add_input(
            "data:weight:airframe:wing:punctual_mass:mass",
            shape_by_conn=True,
            copy_shape="data:weight:airframe:wing:punctual_mass:y_ratio",
            units="kg",
            val=0.0,
        )

        self.add_input("data:mission:sizing:fuel", val=np.nan, units="kg")

        self.add_input("settings:geometry:fuel_tanks:depth", val=np.nan)

        self.add_output(
            "data:loads:structure:ultimate:force_distribution:wing",
            units="N/m",
            shape=SPAN_MESH_POINT_LOADS,
        )
        self.add_output(
            "data:loads:structure:ultimate:force_distribution:fuel",
            units="N/m",
            shape=SPAN_MESH_POINT_LOADS,
        )
        self.add_output(
            "data:loads:structure:ultimate:force_distribution:point_mass",
            units="N/m",
            shape=SPAN_MESH_POINT_LOADS,
        )

        self.add_output(
            "data:loads:structure:ultimate:shear:wing", units="N", shape=SPAN_MESH_POINT_LOADS
        )
        self.add_output(
            "data:loads:structure:ultimate:shear:fuel", units="N", shape=SPAN_MESH_POINT_LOADS
        )
        self.add_output(
            "data:loads:structure:ultimate:shear:point_mass", units="N", shape=SPAN_MESH_POINT_LOADS
        )

        self.add_output(
            "data:loads:structure:ultimate:root_bending:wing",
            units="N*m",
            shape=SPAN_MESH_POINT_LOADS,
        )
        self.add_output(
            "data:loads:structure:ultimate:root_bending:fuel",
            units="N*m",
            shape=SPAN_MESH_POINT_LOADS,
        )
        self.add_output(
            "data:loads:structure:ultimate:root_bending:point_mass",
            units="N*m",
            shape=SPAN_MESH_POINT_LOADS,
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        # STEP 1/XX - DEFINE OR CALCULATE INPUT DATA FOR LOAD COMPUTATION

        y_vector = inputs["data:aerodynamics:wing:low_speed:Y_vector"]
        chord_vector = inputs["data:aerodynamics:wing:low_speed:chord_vector"]

        semi_span = float(inputs["data:geometry:wing:span"]) / 2.0
        root_chord = inputs["data:geometry:wing:root:chord"]
        tip_chord = inputs["data:geometry:wing:tip:chord"]

        load_factor_shear = float(inputs["data:loads:max_shear:load_factor"])
        load_factor_rbm = float(inputs["data:loads:max_rbm:load_factor"])
        wing_mass = inputs["data:weight:airframe:wing:mass"]
        fuel_mass = inputs["data:mission:sizing:fuel"]

        # STEP 2/XX - DELETE THE ADDITIONAL ZEROS WE HAD TO PUT TO FIT OPENMDAO AND ADD A POINT
        # AT THE ROOT (Y=0) AND AT THE VERY TIP (Y=SPAN/2) TO GET THE WHOLE SPAN OF THE WING IN
        # THE INTERPOLATION WE WILL DO LATER

        # Reformat the y_vector array as was done in the aerostructural component
        y_vector = AerostructuralLoad.delete_additional_zeros(y_vector)
        chord_vector = AerostructuralLoad.delete_additional_zeros(chord_vector)
        y_vector, _ = AerostructuralLoad.insert_in_sorted_array(y_vector, 0.0)
        chord_vector = np.insert(chord_vector, 0, root_chord)
        y_vector_orig, _ = AerostructuralLoad.insert_in_sorted_array(y_vector, semi_span)
        chord_vector_orig = np.append(chord_vector, tip_chord)

        # STEP 3/XX - WE COMPUTE THE BASELINE WEiGHT DISTRIBUTION AND SCALE IT UP ACCORDING TO
        # THE MOST CONSTRAINING CASE IDENTIFIED IN  THE AEROSTRUCTURAL ANALYSIS

        y_vector, point_mass_array_orig = AerostructuralLoad.compute_relief_force(
            inputs, y_vector_orig, chord_vector_orig, 0.0, 0.0
        )
        _, wing_mass_array_orig = AerostructuralLoad.compute_relief_force(
            inputs, y_vector_orig, chord_vector_orig, wing_mass, 0.0, False
        )
        _, fuel_mass_array_orig = AerostructuralLoad.compute_relief_force(
            inputs, y_vector_orig, chord_vector_orig, 0.0, fuel_mass, False
        )

        point_mass_array = max(load_factor_shear, load_factor_rbm) * point_mass_array_orig
        wing_mass_array = max(load_factor_shear, load_factor_rbm) * wing_mass_array_orig
        fuel_mass_array = max(load_factor_shear, load_factor_rbm) * fuel_mass_array_orig

        additional_zeros = np.zeros(SPAN_MESH_POINT_LOADS - len(y_vector))
        point_mass_array_outputs = np.concatenate([point_mass_array, additional_zeros])
        wing_mass_array_outputs = np.concatenate([wing_mass_array, additional_zeros])
        fuel_mass_array_outputs = np.concatenate([fuel_mass_array, additional_zeros])

        outputs["data:loads:structure:ultimate:force_distribution:wing"] = wing_mass_array_outputs
        outputs["data:loads:structure:ultimate:force_distribution:fuel"] = fuel_mass_array_outputs
        outputs[
            "data:loads:structure:ultimate:force_distribution:point_mass"
        ] = point_mass_array_outputs

        point_shear_array = AerostructuralLoad.compute_shear_diagram(
            y_vector, load_factor_shear * point_mass_array_orig
        )
        wing_shear_array = AerostructuralLoad.compute_shear_diagram(
            y_vector, load_factor_shear * wing_mass_array_orig
        )
        fuel_shear_array = AerostructuralLoad.compute_shear_diagram(
            y_vector, load_factor_shear * fuel_mass_array_orig
        )

        # STEP 4/XX - WE ADD ZEROS AT THE END OF THE RESULT LIFT DISTRIBUTION TO FIT THE FORMAT
        # IMPOSED BY OPENMDAO

        point_shear_array = np.concatenate([point_shear_array, additional_zeros])
        wing_shear_array = np.concatenate([wing_shear_array, additional_zeros])
        fuel_shear_array = np.concatenate([fuel_shear_array, additional_zeros])

        outputs["data:loads:structure:ultimate:shear:wing"] = wing_shear_array
        outputs["data:loads:structure:ultimate:shear:fuel"] = fuel_shear_array
        outputs["data:loads:structure:ultimate:shear:point_mass"] = point_shear_array

        point_root_bending_array = AerostructuralLoad.compute_bending_moment_diagram(
            y_vector, load_factor_rbm * point_mass_array_orig
        )
        wing_root_bending_array = AerostructuralLoad.compute_bending_moment_diagram(
            y_vector, load_factor_rbm * wing_mass_array_orig
        )
        fuel_root_bending_array = AerostructuralLoad.compute_bending_moment_diagram(
            y_vector, load_factor_rbm * fuel_mass_array_orig
        )

        # STEP 4/XX - WE ADD ZEROS AT THE END OF THE RESULT LIFT DISTRIBUTION TO FIT THE FORMAT
        # IMPOSED BY OPENMDAO

        point_root_bending_array = np.concatenate([point_root_bending_array, additional_zeros])
        wing_root_bending_array = np.concatenate([wing_root_bending_array, additional_zeros])
        fuel_root_bending_array = np.concatenate([fuel_root_bending_array, additional_zeros])

        outputs["data:loads:structure:ultimate:root_bending:wing"] = wing_root_bending_array
        outputs["data:loads:structure:ultimate:root_bending:fuel"] = fuel_root_bending_array
        outputs["data:loads:structure:ultimate:root_bending:point_mass"] = point_root_bending_array
