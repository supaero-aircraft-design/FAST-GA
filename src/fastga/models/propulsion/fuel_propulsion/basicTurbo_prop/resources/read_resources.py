# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import os.path as pth
import pandas as pd


def read_air_coeff():
    file = pth.join(pth.dirname(__file__), "T_Cv_Cp.csv")
    db = pd.read_csv(file)

    temp = db["T"]
    cv_n = db["CV"]
    cp_n = db["CP"]
    gamma_n = db["GAMMA"]

    return temp.to_numpy(), cv_n.to_numpy(), cp_n.to_numpy(), gamma_n.to_numpy()


def read_pressurization_coeff():
    file = pth.join(pth.dirname(__file__), "cabin_pressurisation.csv")
    db = pd.read_csv(file)

    flight_altitude = db["FLIGHT_ALTITUDE"]
    cabin_altitude = db["CABIN_ALTITUDE"]

    return flight_altitude.to_numpy(), cabin_altitude.to_numpy()
