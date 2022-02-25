"""Module that contains constants used ofr the sizing of the hybrid powertrain."""

# -*- coding: utf-8 -*-
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

# Battery
# Based on : https://commons.erau.edu/edt/393
#            https://repository.tudelft.nl/islandora/object/uuid%3A6e274095-9920-4d20-9e11-d5b76363e709

CELL_WEIGHT_FRACTION = 0.58  # Cell weight fraction used for the computation of weight
BATT_OVERHEAD = 0.60  # Overhead factor - Considering 40% of the battery pack consists of overhead components

# Inverter
# Based on : https://repository.tudelft.nl/islandora/object/uuid%3A6e274095-9920-4d20-9e11-d5b76363e709
#            https://electricalnotes.wordpress.com/2015/10/02/calculate-size-of-inverter-battery-bank/

INV_EFF = 0.94  # Efficiency in current industry
AF = 0.20  # 20% Additional Further Load Expansion

# Liquid cooling subsystem
# Based on : https://www.researchgate.net/publication/319935703

LC_SS_VOLUME_FRACTION = 0.29  # Fraction of FC stack volume
LC_SS_MASS_FRACTION = 0.17  # Fraction of FC stack mass

# Balance of Plant of the FC system
# Based on : https://repository.tudelft.nl/islandora/object/uuid%3A6e274095-9920-4d20-9e11-d5b76363e709

FC_OVERHEAD = 0.30

# Intakes for the sized Heat Exchanger : parameters of the reference NACA intake
# Based on : https://www.researchgate.net/publication/303312026_Numerical_Study_of_the_Performance_Improvement_of_Submerged_Air_Intakes_Using_Vortex_Generators

NACA_INTAKE = {
    'WIDTH': 120,  # [mm]
    'LENGTH': 229.33,  # [mm]
    'DEPTH': 30,  # [mm]
    'MASS_FLOW': 0.260  # [kg/s]
    }
