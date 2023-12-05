import os

import pandas as pd

file_path = os.path.dirname(__file__)


class production_data:
    """Example datasets provided by the pyStoNED
    """

    def __init__(self, dmu, x, y, b=None, z=None):
        """General data structure

        Args:
            dmu (String): decision making unit.
            x (Numbers): input variables.
            y (Numbers): output variables.
            b (Numbers, optional): bad output variables. Defaults to None.
            z (Numbers, optional): contextual variables. Defaults to None.
        """
        self.decision_making_unit = dmu
        self.x = x
        self.y = y
        self.b = b
        self.z = z


def load_data_electric():
    return pd.read_csv(file_path + "/data/electricityFirms.csv")

