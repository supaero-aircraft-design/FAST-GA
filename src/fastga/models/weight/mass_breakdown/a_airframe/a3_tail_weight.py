"""
Python module for tail weight calculation consisted with the htp and vtp weight calculations,
part of the airframe mass computation.
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

import fastoad.api as oad
import openmdao.api as om
from .constants import (
    SUBMODEL_HTP_MASS,
    SUBMODEL_TAIL_MASS,
    SUBMODEL_VTP_MASS,
)

oad.RegisterSubmodel.active_models[SUBMODEL_TAIL_MASS] = (
    "fastga.submodel.weight.mass.airframe.tail.legacy"
)


# pylint: disable=too-few-public-methods
@oad.RegisterSubmodel(SUBMODEL_TAIL_MASS, "fastga.submodel.weight.mass.airframe.tail.legacy")
class ComputeTailWeight(om.Group):
    """
    Weight estimation for tail weight

    Based on a statistical analysis. See :cite:`raymer:2012` but can also be found in
    :cite:`gudmundsson:2013`
    """

    def __init__(self, **kwargs):
        """
        Set up corresponded components if user didn't define specifically.
        """
        super().__init__(**kwargs)

        if not oad.RegisterSubmodel.active_models.get(SUBMODEL_HTP_MASS):
            oad.RegisterSubmodel.active_models[SUBMODEL_HTP_MASS] = (
                "fastga.submodel.weight.mass.airframe.htp.legacy"
            )

        if not oad.RegisterSubmodel.active_models.get(SUBMODEL_VTP_MASS):
            oad.RegisterSubmodel.active_models[SUBMODEL_VTP_MASS] = (
                "fastga.submodel.weight.mass.airframe.vtp.legacy"
            )

    # pylint: disable=missing-function-docstring
    # Overriding OpenMDAO setup
    def setup(self):
        self.add_subsystem(
            "empennage_htp_weight",
            oad.RegisterSubmodel.get_submodel(SUBMODEL_HTP_MASS),
            promotes=["*"],
        )
        self.add_subsystem(
            "empennage_vtp_weight",
            oad.RegisterSubmodel.get_submodel(SUBMODEL_VTP_MASS),
            promotes=["*"],
        )


# pylint: disable=too-few-public-methods
@oad.RegisterSubmodel(SUBMODEL_TAIL_MASS, "fastga.submodel.weight.mass.airframe.tail.gd")
class ComputeTailWeightGD(om.Group):
    """
    Weight estimation for tail weight

    Based on a statistical analysis. See :cite:`roskampart5:1985` traditionally used on
    commercial aircraft but found to work fairly well on high performance GA aircraft.
    """

    def __init__(self, **kwargs):
        """
        Set up corresponded components if user didn't define specifically.
        """
        super().__init__(**kwargs)

        if not oad.RegisterSubmodel.active_models.get(SUBMODEL_HTP_MASS):
            oad.RegisterSubmodel.active_models[SUBMODEL_HTP_MASS] = (
                "fastga.submodel.weight.mass.airframe.htp.gd"
            )

        if not oad.RegisterSubmodel.active_models.get(SUBMODEL_VTP_MASS):
            oad.RegisterSubmodel.active_models[SUBMODEL_VTP_MASS] = (
                "fastga.submodel.weight.mass.airframe.vtp.gd"
            )

    # pylint: disable=missing-function-docstring
    # Overriding OpenMDAO setup
    def setup(self):
        self.add_subsystem(
            "empennage_htp_weight",
            oad.RegisterSubmodel.get_submodel(SUBMODEL_HTP_MASS),
            promotes=["*"],
        )
        self.add_subsystem(
            "empennage_vtp_weight",
            oad.RegisterSubmodel.get_submodel(SUBMODEL_VTP_MASS),
            promotes=["*"],
        )


# pylint: disable=too-few-public-methods
@oad.RegisterSubmodel(SUBMODEL_TAIL_MASS, "fastga.submodel.weight.mass.airframe.tail.torenbeek_gd")
class ComputeTailWeightTorenbeekGD(om.Group):
    """
    Weight estimation for tail weight

    Based on a statistical analysis. See :cite:`roskampart5:1985` traditionally used on
    commercial aircraft but found to work fairly well on high performance GA aircraft. Should
    only be used with aircraft having a diving speed above 250 KEAS.
    """

    def __init__(self, **kwargs):
        """
        Set up corresponded components if user didn't define specifically.
        """
        super().__init__(**kwargs)
        if not oad.RegisterSubmodel.active_models.get(SUBMODEL_HTP_MASS):
            oad.RegisterSubmodel.active_models[SUBMODEL_HTP_MASS] = (
                "fastga.submodel.weight.mass.airframe.htp.torenbeek"
            )

        if not oad.RegisterSubmodel.active_models.get(SUBMODEL_VTP_MASS):
            oad.RegisterSubmodel.active_models[SUBMODEL_VTP_MASS] = (
                "fastga.submodel.weight.mass.airframe.vtp.gd"
            )

    # pylint: disable=missing-function-docstring
    # Overriding OpenMDAO setup
    def setup(self):
        self.add_subsystem(
            "empennage_htp_weight",
            oad.RegisterSubmodel.get_submodel(SUBMODEL_HTP_MASS),
            promotes=["*"],
        )
        self.add_subsystem(
            "empennage_vtp_weight",
            oad.RegisterSubmodel.get_submodel(SUBMODEL_VTP_MASS),
            promotes=["*"],
        )
