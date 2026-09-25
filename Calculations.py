from Datas import Datas
import pandas as pd
import numpy as np


class Calculations:
    def __init__(self, age, sex, smoking, alcohol, extreme_sports, health_class,
                 term_years, face_amount):

        constants = Datas().load_policy_constants()
        min_age = constants.at["min_age", "value"]
        max_age = constants.at["max_age", "value"]

        if age > max_age or age < min_age:
            raise ValueError(f"age must be between {min_age} and {max_age}")
        else:
            self.age = age

        self.sex = sex
        self.smoking = smoking
        self.alcohol = alcohol
        self.extreme_sports = extreme_sports
        self.health_class = health_class
        self.term_years = term_years
        self.face_amount = face_amount

        self.discount_rate = constants.at["discount_rate", "value"]
        self.expense_loading = constants.at["expense_loading", "value"]

        self.mortality_table = Datas().load_mortaility_table()
        self.risk_factors = Datas().load_risk_factors()


    def get_base_qx(self, age):
        column = f"qx_{self.sex}"
        return self.mortality_table.at[age, column]

    def get_risk_multiplier(self):
        rf = self.risk_factors
        multiplier = 1.0
        multiplier *= rf.at[("smoking", self.smoking), "multiplier"]
        multiplier *= rf.at[("alcohol", self.alcohol), "multiplier"]
        multiplier *= rf.at[("extreme_sports", self.extreme_sports), "multiplier"]
        multiplier *= rf.at[("health_class", self.health_class), "multiplier"]
        return multiplier

    def adjusted_qx(self, age):
        adj = self.get_base_qx(age) * self.get_risk_multiplier()
        return min(adj, 1.0)   



    def survival_prob_one_year(self, age):
        return 1 - self.adjusted_qx(age)

    def cumulative_survival_prob(self, years):
        prob = 1.0
        for t in range(years):
            prob *= self.survival_prob_one_year(self.age + t)
        return prob

    def cumulative_death_prob(self, years):
        return 1 - self.cumulative_survival_prob(years)



    def prob_death_in_year(self, k):
        survive_prior_years = self.cumulative_survival_prob(k - 1)
        die_this_year = self.adjusted_qx(self.age + k - 1)
        return survive_prior_years * die_this_year

    # ---------- 6. discounting ----------

    def discount_factor(self, k):
        return 1 / ((1 + self.discount_rate) ** k)


    def epv_benefit(self):
        total = 0
        for k in range(1, self.term_years + 1):
            total += self.prob_death_in_year(k) * self.face_amount * self.discount_factor(k)
        return total

    def epv_premiums_per_dollar(self):
        total = 0
        for k in range(self.term_years):
            total += self.cumulative_survival_prob(k) * self.discount_factor(k)
        return total

    def net_premium(self):
        return self.epv_benefit() / self.epv_premiums_per_dollar()

    def gross_premium(self):
        return self.net_premium() * (1 + self.expense_loading)



    def payout_variance(self):
        n_q_x = self.cumulative_death_prob(self.term_years)
        return (self.face_amount ** 2) * n_q_x * (1 - n_q_x)

    def payout_std_dev(self):
        return self.payout_variance() ** 0.5



    def simulate_policy(self, n_simulations=10000, seed=None):
        rng = np.random.default_rng(seed)
        pv_payouts = np.zeros(n_simulations)

        for i in range(n_simulations):
            for k in range(1, self.term_years + 1):
                if rng.random() < self.adjusted_qx(self.age + k - 1):
                    pv_payouts[i] = self.face_amount * self.discount_factor(k)
                    break  

        return pv_payouts


    def summary(self):
        return {
            "net_premium": round(self.net_premium(), 2),
            "gross_premium": round(self.gross_premium(), 2),
            "epv_benefit": round(self.epv_benefit(), 2),
            "payout_std_dev": round(self.payout_std_dev(), 2),
        }