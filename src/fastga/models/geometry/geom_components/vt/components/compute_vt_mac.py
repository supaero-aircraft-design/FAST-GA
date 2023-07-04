"""
    Estimation of vertical tail mean aerodynamic chords
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
from openmdao.core.explicitcomponent import ExplicitComponent
from openmdao.core.group import Group


# class ComputeVTMacFD(ExplicitComponent):
#     # TODO: Document equations. Cite sources
#     """
#     Vertical tail mean aerodynamic chord estimation based on (F)ixed tail (D)istance.
#     """

#     def setup(self):
#         self.add_input("data:geometry:vertical_tail:root:chord", val=np.nan, units="m")
#         self.add_input("data:geometry:vertical_tail:tip:chord", val=np.nan, units="m")
#         self.add_input("data:geometry:vertical_tail:sweep_25", val=np.nan, units="rad")
#         self.add_input("data:geometry:vertical_tail:span", val=np.nan, units="m")

#         self.add_output("data:geometry:vertical_tail:MAC:length", units="m")
#         self.add_output("data:geometry:vertical_tail:MAC:at25percent:x:local", units="m")
#         self.add_output("data:geometry:vertical_tail:MAC:z", units="m")

#         self.declare_partials(
#             "data:geometry:vertical_tail:MAC:length",
#             ["data:geometry:vertical_tail:root:chord", "data:geometry:vertical_tail:tip:chord"],
#             method="fd",
#         )
#         self.declare_partials(
#             "data:geometry:vertical_tail:MAC:at25percent:x:local", "*", method="fd"
#         )
#         self.declare_partials(
#             "data:geometry:vertical_tail:MAC:z",
#             [
#                 "data:geometry:vertical_tail:root:chord",
#                 "data:geometry:vertical_tail:tip:chord",
#                 "data:geometry:vertical_tail:span",
#             ],
#             method="fd",
#         )

#     def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
#         root_chord = inputs["data:geometry:vertical_tail:root:chord"]
#         tip_chord = inputs["data:geometry:vertical_tail:tip:chord"]
#         sweep_25_vt = inputs["data:geometry:vertical_tail:sweep_25"]
#         b_v = inputs["data:geometry:vertical_tail:span"]

#         tmp = root_chord * 0.25 + b_v * np.tan(sweep_25_vt) - tip_chord * 0.25

#         mac_vt = (
#             (root_chord ** 2 + root_chord * tip_chord + tip_chord ** 2)
#             / (tip_chord + root_chord)
#             * 2.0
#             / 3.0
#         )
#         x0_vt = (tmp * (root_chord + 2 * tip_chord)) / (3 * (root_chord + tip_chord))
#         z0_vt = (2 * b_v * (0.5 * root_chord + tip_chord)) / (3 * (root_chord + tip_chord))

#         outputs["data:geometry:vertical_tail:MAC:length"] = mac_vt
#         outputs["data:geometry:vertical_tail:MAC:at25percent:x:local"] = x0_vt
#         outputs["data:geometry:vertical_tail:MAC:z"] = z0_vt


class ComputeVTMacFD(Group):
    # TODO: Document equations. Cite sources
    """
    Vertical tail mean aerodynamic chord estimation based on (F)ixed tail (D)istance.
    """

    def setup(self):

        self.add_subsystem(
            "comp_vt_mac_length_fd", ComputeVTMacLength(), promotes=["*"],
        )
        self.add_subsystem(
            "comp_vt_mac_x25_fd", ComputeVTMacX25(), promotes=["*"],
        )
        self.add_subsystem(
            "comp_vt_mac_z_fd", ComputeVTMacZ(), promotes=["*"],
        )


# class ComputeVTMacFL(ExplicitComponent):
#     # TODO: Document equations. Cite sources
#     """
#     Vertical tail mean aerodynamic chord estimation based on (F)ixed fuselage (L)ength (VTP
#     distance computed).
#     """

#     def setup(self):
#         self.add_input("data:geometry:vertical_tail:root:chord", val=np.nan, units="m")
#         self.add_input("data:geometry:vertical_tail:tip:chord", val=np.nan, units="m")
#         self.add_input("data:geometry:vertical_tail:span", val=np.nan, units="m")

#         self.add_output("data:geometry:vertical_tail:MAC:length", units="m")
#         self.add_output("data:geometry:vertical_tail:MAC:z", units="m")

#         self.declare_partials("*", "*", method="fd")

#     def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
#         root_chord = inputs["data:geometry:vertical_tail:root:chord"]
#         tip_chord = inputs["data:geometry:vertical_tail:tip:chord"]
#         b_v = inputs["data:geometry:vertical_tail:span"]

#         mac_vt = (
#             (root_chord ** 2 + root_chord * tip_chord + tip_chord ** 2)
#             / (tip_chord + root_chord)
#             * 2.0
#             / 3.0
#         )
#         z0_vt = (2 * b_v * (0.5 * root_chord + tip_chord)) / (3 * (root_chord + tip_chord))

#         outputs["data:geometry:vertical_tail:MAC:length"] = mac_vt
#         outputs["data:geometry:vertical_tail:MAC:z"] = z0_vt


class ComputeVTMacFL(Group):
    # TODO: Document equations. Cite sources
    """
    Vertical tail mean aerodynamic chord estimation based on (F)ixed fuselage (L)ength (VTP
    distance computed).
    """

    def setup(self):

        self.add_subsystem(
            "comp_vt_mac_length_fl", ComputeVTMacLength(), promotes=["*"],
        )
        self.add_subsystem(
            "comp_vt_mac_z_fl", ComputeVTMacZ(), promotes=["*"],
        )


class ComputeVTMacLength(ExplicitComponent):
    """
    Compute MAC length of the vertical tail.
    """

    def setup(self):

        self.add_input("data:geometry:vertical_tail:root:chord", val=np.nan, units="m")
        self.add_input("data:geometry:vertical_tail:tip:chord", val=np.nan, units="m")
        
        self.add_output("data:geometry:vertical_tail:MAC:length", units="m")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        
        root_chord = inputs["data:geometry:vertical_tail:root:chord"]
        tip_chord = inputs["data:geometry:vertical_tail:tip:chord"]

        mac_vt = (
            (root_chord ** 2 + root_chord * tip_chord + tip_chord ** 2)
            / (tip_chord + root_chord)
            * 2.0
            / 3.0
        )

        outputs["data:geometry:vertical_tail:MAC:length"] = mac_vt


class ComputeVTMacX25(ExplicitComponent):
    """
    Compute x coordinate (local) at 25% MAC of the vertical tail.
    """

    def setup(self):

        self.add_input("data:geometry:vertical_tail:root:chord", val=np.nan, units="m")
        self.add_input("data:geometry:vertical_tail:tip:chord", val=np.nan, units="m")
        self.add_input("data:geometry:vertical_tail:sweep_25", val=np.nan, units="rad")
        self.add_input("data:geometry:vertical_tail:span", val=np.nan, units="m")

        self.add_output("data:geometry:vertical_tail:MAC:at25percent:x:local", units="m")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        
        root_chord = inputs["data:geometry:vertical_tail:root:chord"]
        tip_chord = inputs["data:geometry:vertical_tail:tip:chord"]
        sweep_25_vt = inputs["data:geometry:vertical_tail:sweep_25"]
        b_v = inputs["data:geometry:vertical_tail:span"]

        tmp = root_chord * 0.25 + b_v * np.tan(sweep_25_vt) - tip_chord * 0.25
        x0_vt = (tmp * (root_chord + 2 * tip_chord)) / (3 * (root_chord + tip_chord))
        
        outputs["data:geometry:vertical_tail:MAC:at25percent:x:local"] = x0_vt


class ComputeVTMacZ(ExplicitComponent):
    """
    Compute z coordinate of the vertical tail's MAC.
    """

    def setup(self):

        self.add_input("data:geometry:vertical_tail:root:chord", val=np.nan, units="m")
        self.add_input("data:geometry:vertical_tail:tip:chord", val=np.nan, units="m")
        self.add_input("data:geometry:vertical_tail:span", val=np.nan, units="m")

        self.add_output("data:geometry:vertical_tail:MAC:z", units="m")

        self.declare_partials("*","*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        
        root_chord = inputs["data:geometry:vertical_tail:root:chord"]
        tip_chord = inputs["data:geometry:vertical_tail:tip:chord"]
        b_v = inputs["data:geometry:vertical_tail:span"]

        z0_vt = (2 * b_v * (0.5 * root_chord + tip_chord)) / (3 * (root_chord + tip_chord))

        outputs["data:geometry:vertical_tail:MAC:z"] = z0_vt
