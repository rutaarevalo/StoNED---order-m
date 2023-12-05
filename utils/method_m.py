import os
import statistics

import numpy as np
import pandas as pd
from pandas import DataFrame
from pandas.io.parsers import TextFileReader
from pystoned import CNLS, StoNED
from pystoned.constant import RTS_VRS, RED_MOM, FUN_PROD, OPT_LOCAL, CET_ADDI

from utils.dataset import production_data

file_path = os.path.dirname(__file__)


def validate_iterations(size: int, iterate: int, len_df: int) -> int:
    if iterate == 0:
        return round(len_df / size)
    elif (size * iterate) > len_df:
        return round(len_df / size)
    return iterate


def get_reduce_df(source: DataFrame, index, size) -> DataFrame:
    new_df = source.copy()
    current = pd.DataFrame(new_df.iloc[index]).transpose()
    for title, data in new_df.dtypes.items():
        current[title] = current[title].astype(data)

    new_df = new_df.drop(index).sample(frac=1)
    start = 0
    end = size - 1
    slip = new_df.iloc[start:end]
    slip = pd.concat([current, slip]).reset_index(drop=True)
    return slip


def transpose_matrix(matrix):
    if not matrix:
        return []

    num_rows, num_cols = len(matrix), len(matrix[0])
    transposed = []

    for col in range(num_cols):
        transposed_row = [matrix[row][col] for row in range(num_rows)]
        transposed.append(transposed_row)

    return transposed


class METHOD_M:

    def __init__(self, df: DataFrame, m: int = 10, b: int = 30,
                 x_select=None,
                 y_select=None,
                 z_select=None):
        """METHOD_M

        Args:
            m (int): size of the dmu to randomly generate
            b (int): number of times the function will be executed for each row of the dataframe.
            df (DataFrame): Dmu, dataset to split.
            function (method): action to execute in this process.
        """
        if z_select is None:
            z_select = ['CAPEX', 'OPEX']
        if y_select is None:
            y_select = ['TOTEX']
        if x_select is None:
            x_select = ['Energy', 'Length', 'Customers']
        self.m = m
        self.b = b
        self.df = df
        self.x = x_select,
        self.y = y_select,
        self.z = z_select
        super().__init__()

    def execute(self):
        list_sigma_u = []
        list_sigma_v = []
        list_mu = []
        epsilon = []
        betas = []

        for index, _ in self.df.iterrows():
            data = self.execute_stoned_by_row(index)
            list_sigma_u.append(data["sigma_u"])
            list_sigma_v.append(data["sigma_v"])
            list_mu.append(data["mu"])
            epsilon.append(data["epsilon"])
            betas.append(data["betas"])

        sigma_u = statistics.mean(list_sigma_u)
        sigma_v = statistics.mean(list_sigma_v)
        mu = statistics.mean(list_mu)

        print(sigma_u)
        print(sigma_v)
        print("mu")
        print(mu)
        print("betas")
        print(np.array(betas))
        print("epsilon")
        print(np.array(epsilon))
        print(f"Finish Execution")

    # @background
    def execute_stoned_by_row(self, index):
        print(f"the row actual is: {index}")

        """define accumulative variable"""
        sigma_u = []
        sigma_v = []
        epsilon = []
        mu = []
        betas = []
        iterator = 0
        while iterator < self.b:
            bt, rd = self.process_stoned(index, iterator)
            if bt is not None and rd is not None:
                if np.isnan(rd.sigma_v) or np.isnan(bt).any():
                    print("nan value")
                else:
                    sigma_v.append(rd.sigma_v)
                    sigma_u.append(rd.sigma_u)
                    epsilon.append(rd.epsilon)
                    betas.append(bt)
                    mu.append(rd.mu)
                    iterator += 1

        row = {
            "sigma_u": statistics.mean(sigma_u),
            "sigma_v": statistics.mean(sigma_v),
            "mu": statistics.mean(mu),
            "epsilon": np.mean(epsilon),
            "betas": np.mean(betas)
        }
        return row

    def process_stoned(self, index, iterator):
        print(f"the iteration actual is: {iterator}")

        try:
            df_reduce = get_reduce_df(self.df, index, self.m)
            data = self.dmu(df_reduce, self.x[0], self.y[0], self.z)
            model = CNLS.CNLS(data.y, data.x, data.z, cet=CET_ADDI, fun=FUN_PROD, rts=RTS_VRS)
            model.optimize(email=OPT_LOCAL)
            rd = StoNED.StoNED(model)
            bt = rd.get_technical_inefficiency(RED_MOM)

            return bt, rd
        except Exception as error:
            print(error.__cause__)
            return None, None

    def mean(self, matrix):
        r = []
        columns = list(zip(*matrix))
        for i in range(self.m):
            r.append(statistics.mean(columns[i]))
        return r

    @staticmethod
    def dmu(df: DataFrame | TextFileReader, x_select: list, y_select: list, z_select: list):

        z = None
        dmu = np.asanyarray(df.index.tolist()).T
        x = np.column_stack(
            [np.asanyarray(df[selected]).T for selected in x_select])
        y = np.column_stack(
            [np.asanyarray(df[selected]).T for selected in y_select])
        if z_select is not None:
            z = np.column_stack(
                [np.asanyarray(df[selected]).T for selected in z_select])
        return production_data(dmu, x, y, z=z)
