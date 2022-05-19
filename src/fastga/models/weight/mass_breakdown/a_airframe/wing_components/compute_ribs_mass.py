"""
Computes the mass of the ribs based on the model presented by Raquel ALONSO
in her MAE research project report.
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

import openmdao.api as om
import numpy as np


class ComputeRibsMass(om.ExplicitComponent):
    def setup(self):

        self.add_input("data:geometry:fuselage:maximum_width", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:maximum_height", val=np.nan, units="m")
        self.add_input("data:geometry:wing:span", val=np.nan, units="m")
        self.add_input("data:geometry:wing:thickness_ratio", val=np.nan)
        self.add_input("data:geometry:wing:taper_ratio", val=np.nan)
        self.add_input("data:geometry:wing:root:chord", val=np.nan, units="m")
        self.add_input("data:geometry:wing:sweep_25", val=np.nan, units="rad")
        self.add_input("data:geometry:propulsion:engine:count", val=np.nan)

        self.add_input(
            "settings:wing:airfoil:skin:ka",
            val=0.92,
            desc="Correction coefficient needed to account for the hypothesis of a rectangular "
            "wingbox",
        )
        self.add_input(
            "settings:wing:airfoil:skin:d_wingbox",
            val=0.4,
            desc="ratio of the wingbox working depth/airfoil chord",
        )
        self.add_input(
            "settings:materials:aluminium:surface_density",
            val=9.6,
            units="kg/m**2",
            desc="Aluminum surface density",
        )

        self.add_output("data:weight:airframe:wing:ribs:mass", units="kg")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        """
        Component that computes the ribs mass necessary to react to the given linear force
        vector, according to the methodology developed by Raquel Alonso Castilla.
        """
        fus_width = inputs["data:geometry:fuselage:maximum_width"]
        fus_height = inputs["data:geometry:fuselage:maximum_height"]
        wing_span = inputs["data:geometry:wing:span"]
        sweep_25 = inputs["data:geometry:wing:sweep_25"]
        root_chord = inputs["data:geometry:wing:root:chord"]
        taper_ratio = inputs["data:geometry:wing:taper_ratio"]
        thickness_ratio = inputs["data:geometry:wing:thickness_ratio"]

        ka = inputs["settings:wing:airfoil:skin:ka"]
        d_wingbox = inputs["settings:wing:airfoil:skin:d_wingbox"]
        delta_ribs = 0.6  # Step between ribs ( hypothesis: constant )

        surface_density = inputs["settings:materials:aluminium:surface_density"]

        fus_radius = np.sqrt(fus_height * fus_width) / 2.0
        sweep_e = np.arctan(
            np.tan(sweep_25)
            + (1.0 - taper_ratio)
            * root_chord
            / (wing_span / 2.0 - fus_radius)
            * (25.0 - 35.0)
            / 100.0
        )
        n_ribs = int((wing_span / 2.0) / (np.cos(sweep_e) * delta_ribs)) + 1.0
        k = (1.0 - taper_ratio) * root_chord / (wing_span / 2.0)
        xe_root = -root_chord * (0.1 / (wing_span / 2.0 - fus_radius) * wing_span / 2.0 - 0.75)
        f_phi_e = 1.0 / (
            np.cos(sweep_e) ** 4.0
            * (1.0 + np.tan(sweep_e) ** 2.0 + (1.0 - xe_root / root_chord) * k * np.tan(sweep_e))
            * (1.0 + np.tan(sweep_e) ** 2.0 - xe_root / root_chord * k * np.tan(sweep_e))
        )
        m_factor = (
            (taper_ratio ** 2.0 + taper_ratio + 1.0) * n_ribs / 3.0
            + (taper_ratio ** 2.0 - 1.0) / 2.0
            + (taper_ratio - 1.0) ** 2.0 / (6.0 * n_ribs)
        )  # integration constant
        ribs_mass = abs(
            2.0
            * surface_density
            * ka
            * d_wingbox
            * np.cos(sweep_e)
            * f_phi_e
            * thickness_ratio
            * root_chord ** 2.0
            * m_factor
        )

        if inputs["data:geometry:propulsion:engine:count"] > 4:
            ribs_mass *= 1.1

        outputs["data:weight:airframe:wing:ribs:mass"] = ribs_mass
