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

import numpy as np
from fastoad.model_base import Atmosphere
from dataclasses import dataclass

@dataclass
class GH2_storage:
    """Module that contains all methods addressing the sizing of the h2_storage."""
    n_tanks: float = None
    grav_cap_350: float = None #Gravimetric index of storage system for a reference storage pressure of 350bar. (kg/kg_H2)
    grav_cap_700: float = None #Gravimetric index of storage system for a reference storage pressure of 700bar. (kg/kg_H2)
    vol_cap_350: float = None  # Volumetric index of storage system for a reference storage pressure of 350bar. (kg_H2/m**3)
    vol_cap_700: float = None  # Volumetric index of storage system for a reference storage pressure of 700bar. (kg_H2/m**3)
    storage_pressure : float = None #GH2 storage pressure during aircraft sizing (MPa)
    tank_lr_ratio : float = None  # length to radius ratio
    bop_factor : float = None # BoP factor for weight calculation based on physical model
    density_cfp : float = None # Main material density (default cfp) (kg/m**3)
    density_liner : float = None # Internal liner density (kg/m**3)

    FoS : float = 2.25  # Factor of safety
    max_stress : float = None # Maximum stress of tank material (Pa)
    t_fit : float = None # Thickness tunning parameter
    liner_thick : float = None # Liner thickness (mm)

    #data made available after calculation:
    liner_volume : float = None
    thickness : float = None
    tank_ext_radius : float = None
    tank_ext_length : float = None
    tank_int_radius: float = None
    tank_int_length: float = None
    tank_int_volume: float = None
    tank_ext_volume : float = None
    tot_tank_volume : float = None


    def compute_weight_gravimetric_model(self, fuel_weight: float = None):
        """
        Computes GH2 storage weight based on gravimetric indices.
        Gravimetric indices must include storage BoP

        :param fuel_weight: weight of fuel to store in kg
        """
        # Compute storage weight based on gravimetric indices.
        # Gravimetric indices must include storage BoP

        # Interpolation for
        if self.storage_pressure < 35:
            grav_cap = self.grav_cap_350
        elif self.storage_pressure > 70:
            grav_cap = self.grav_cap_700
        else:
            grav_cap = np.interp(self.storage_pressure, [35, 70], np.concatenate((self.grav_cap_350, self.grav_cap_700)))

        return fuel_weight / grav_cap

    def compute_weight_physical_model(self, tank_volume : float,
                                      int_volume : float,
                                      liner_volume : float,
                                      ):
        """
        Computes GH2 storage weight based on simplified physical relationships
        Code is based on the work done in 'FAST-GA-AMPERE' and on the storage model found here :
        https://www.researchgate.net/publication/24316784_Hydrogen_Storage_for_Aircraft_Applications_Overview

        This function needs data made available by the function "Compute_geometric_volumetric_model".

        :param tank_volume: Total tank volume in m**3
        :param int_volume: Internal tank volume available for GH2 (m**3)
        :param liner_volume: Volume occupied by the liner (m**3)
        """

        self.tank_mass = ((tank_volume - liner_volume - int_volume) * self.density_cfp +
                     liner_volume * self.density_liner) * self.bop_factor  # [kg]

        return self.tank_mass


    def compute_geometry_volumetric_model(self, fuel_weight : float = None):
        """
        Computes the dimensions of the tanks based on volumetric indices
        Volumetric indices must include storage BoP
        """

        T_H = Atmosphere(altitude=0).temperature  # [K]
        self.tank_int_volume = self.compute_GH2_vol(fuel_weight,T_H)/self.n_tanks

        # Interpolation for
        if self.storage_pressure < 35:
            grav_cap = self.vol_cap_350
        elif self.storage_pressure > 70:
            grav_cap = self.vol_cap_700
        else:
            grav_cap = np.interp(self.storage_pressure, [35, 70],
                                 np.concatenate((self.vol_cap_350, self.vol_cap_700)))

        self.tank_ext_volume = fuel_weight / grav_cap / self.n_tanks
        self.tank_ext_radius = (self.tank_ext_volume / ((self.tank_lr_ratio - 2 + 4 / 3) * np.pi)) ** (1 / 3)  # [m]
        self.tank_ext_length = self.tank_lr_ratio * self.tank_ext_radius # [m]

        return self.tank_ext_length, self.tank_ext_radius, self.tank_ext_volume

    def compute_geometry_physical_model(self, fuel_weight : float = None, ):

        T_H = Atmosphere(altitude=0).temperature  # [K]

        V_H = self.compute_GH2_vol(fuel_weight, T_H)

        # Determining internal radius-length of a single cylindrical tank
        self.tank_int_volume = V_H / self.n_tanks
        self.tank_int_radius = (self.tank_int_volume / ((self.tank_lr_ratio - 2 + 4 / 3) * np.pi)) ** (1 / 3)  # [m]
        #tank cylindrical part length:
        self.tank_int_length = self.tank_lr_ratio * self.tank_int_radius - 2 * self.tank_int_radius  # [m]

        # Liner thickness
        tank_liner_radius = self.tank_int_radius + self.liner_thick
        tank_liner_length = self.tank_int_length
        self.liner_volume = np.pi * (
                    tank_liner_radius ** 2) * tank_liner_length + 4 / 3 * np.pi * tank_liner_radius ** 3 - self.tank_int_volume

        # Determining wall thickness and tank volume
        self.thickness = self.storage_pressure*1e6 * tank_liner_radius * self.FoS / (2 * self.max_stress)\
                         * self.t_fit  # [m]
        self.tank_ext_radius = tank_liner_radius + self.thickness
        self.tank_ext_length = tank_liner_length
        self.tank_volume = np.pi * (self.tank_ext_radius ** 2) * self.tank_ext_length + 4 / 3 * np.pi *\
                           self.tank_ext_radius ** 3  # [m**3]
        self.tot_tank_volume = self.tank_volume * self.n_tanks  # [m**3]

        return self.tank_ext_length, self.tank_ext_radius, self.tank_volume

    def compute_GH2_vol(self, m_fuel, temperature):
        """
        Computes the volum occupied by compressed gaseous hydrogene

        :param m_fuel: weight of H2 (kg)
        :param temperature: temperature of hydrogene (°K)
        """

        T_H = temperature

        # Loop with mission fuel
        m_H = m_fuel

        # Determining volume of hydrogen needed
        Z = 0.99704 + 6.4149e-9 * self.storage_pressure * 1e6  # Hydrogen compressibility factor
        R = 4157.2  # [Nm/(Kkg)]
        V_H = Z * R * m_H * T_H / (self.storage_pressure * 1e6)  # [m**3]

        return V_H