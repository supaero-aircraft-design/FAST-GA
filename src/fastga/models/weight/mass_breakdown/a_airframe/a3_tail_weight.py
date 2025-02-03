"""
Python module for tail weight calculation consisting of the htp and vtp weight calculations,
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
    SERVICE_HTP_MASS,
    SERVICE_TAIL_MASS,
    SERVICE_VTP_MASS,
    SUBMODEL_TAIL_MASS_LEGACY,
    SUBMODEL_TAIL_MASS_GD,
    SUBMODEL_TAIL_MASS_TORENBEEKGD,
    SUBMODEL_HTP_MASS_LEGACY,
    SUBMODEL_HTP_MASS_GD,
    SUBMODEL_HTP_MASS_TORENBEEK,
    SUBMODEL_VTP_MASS_LEGACY,
    SUBMODEL_VTP_MASS_GD,
)

oad.RegisterSubmodel.active_models[SERVICE_TAIL_MASS] = SUBMODEL_TAIL_MASS_LEGACY


# pylint: disable=too-few-public-methods
@oad.RegisterSubmodel(SERVICE_TAIL_MASS, SUBMODEL_TAIL_MASS_LEGACY)
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

        if not oad.RegisterSubmodel.active_models.get(SERVICE_HTP_MASS):
            oad.RegisterSubmodel.active_models[SERVICE_HTP_MASS] = SUBMODEL_HTP_MASS_LEGACY

        if not oad.RegisterSubmodel.active_models.get(SERVICE_VTP_MASS):
            oad.RegisterSubmodel.active_models[SERVICE_VTP_MASS] = SUBMODEL_VTP_MASS_LEGACY

    # pylint: disable=missing-function-docstring
    # Overriding OpenMDAO setup
    def setup(self):
        self.add_subsystem(
            "empennage_htp_weight",
            oad.RegisterSubmodel.get_submodel(SERVICE_HTP_MASS),
            promotes=["*"],
        )
        self.add_subsystem(
            "empennage_vtp_weight",
            oad.RegisterSubmodel.get_submodel(SERVICE_VTP_MASS),
            promotes=["*"],
        )


# pylint: disable=too-few-public-methods
@oad.RegisterSubmodel(SERVICE_TAIL_MASS, SUBMODEL_TAIL_MASS_GD)
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

        if not oad.RegisterSubmodel.active_models.get(SERVICE_HTP_MASS):
            oad.RegisterSubmodel.active_models[SERVICE_HTP_MASS] = SUBMODEL_HTP_MASS_GD

        if not oad.RegisterSubmodel.active_models.get(SERVICE_VTP_MASS):
            oad.RegisterSubmodel.active_models[SERVICE_VTP_MASS] = SUBMODEL_VTP_MASS_GD

    # pylint: disable=missing-function-docstring
    # Overriding OpenMDAO setup
    def setup(self):
        self.add_subsystem(
            "empennage_htp_weight",
            oad.RegisterSubmodel.get_submodel(SERVICE_HTP_MASS),
            promotes=["*"],
        )
        self.add_subsystem(
            "empennage_vtp_weight",
            oad.RegisterSubmodel.get_submodel(SERVICE_VTP_MASS),
            promotes=["*"],
        )


# pylint: disable=too-few-public-methods
@oad.RegisterSubmodel(SERVICE_TAIL_MASS, SUBMODEL_TAIL_MASS_TORENBEEKGD)
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
        if not oad.RegisterSubmodel.active_models.get(SERVICE_HTP_MASS):
            oad.RegisterSubmodel.active_models[SERVICE_HTP_MASS] = SUBMODEL_HTP_MASS_TORENBEEK

        if not oad.RegisterSubmodel.active_models.get(SERVICE_VTP_MASS):
            oad.RegisterSubmodel.active_models[SERVICE_VTP_MASS] = SUBMODEL_VTP_MASS_GD

    # pylint: disable=missing-function-docstring
    # Overriding OpenMDAO setup
    def setup(self):
        self.add_subsystem(
            "empennage_htp_weight",
            oad.RegisterSubmodel.get_submodel(SERVICE_HTP_MASS),
            promotes=["*"],
        )
        self.add_subsystem(
            "empennage_vtp_weight",
            oad.RegisterSubmodel.get_submodel(SERVICE_VTP_MASS),
            promotes=["*"],
        )
