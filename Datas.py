import pandas as pd

class Datas:
    @staticmethod
    def load_mortaility_table():
        df = pd.read_csv("Data//mortality_table.csv")
        return df
    @staticmethod
    def load_policy_constants():
        df = pd.read_csv("Data//policy_constants.csv", index_col="parameter")
        return df
    @staticmethod
    def load_risk_factors():
        df = pd.read_csv("Data//risk_factors.csv", index_col=["category", "class"])
        return df
    