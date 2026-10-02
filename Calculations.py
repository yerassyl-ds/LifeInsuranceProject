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

        # ---- Precompute everything that doesn't change per call ----
        # Doing pandas .at[] lookups and re-multiplying the risk factors on
        # every single method call is what made this slow at scale (1000s
        # of policies x 25 years x 1000s of simulations). Instead, we look
        # everything up ONCE here in __init__ and cache it as plain numpy
        # arrays / floats, which are much faster to reuse.
        self._risk_multiplier = self._compute_risk_multiplier()

        # adjusted_qx_array[i] = adjusted qx for age (self.age + i),
        # for i = 0 .. (max_age - self.age). Precomputed once.
        column = f"qx_{self.sex}"
        base_qx_series = self.mortality_table.loc[self.age:, column]
        adj = (base_qx_series.to_numpy() * self._risk_multiplier)
        self._adjusted_qx_array = np.minimum(adj, 1.0)

        # survival_prob_array[i] = 1 - adjusted_qx_array[i]
        self._survival_prob_array = 1 - self._adjusted_qx_array

        # cumulative_survival[t] = probability of surviving t full years
        # from self.age. cumulative_survival[0] = 1.0 by definition.
        self._cumulative_survival = np.concatenate(
            ([1.0], np.cumprod(self._survival_prob_array))
        )

    def _compute_risk_multiplier(self):
        rf = self.risk_factors
        multiplier = 1.0
        multiplier *= rf.at[("smoking", self.smoking), "multiplier"]
        multiplier *= rf.at[("alcohol", self.alcohol), "multiplier"]
        multiplier *= rf.at[("extreme_sports", self.extreme_sports), "multiplier"]
        multiplier *= rf.at[("health_class", self.health_class), "multiplier"]
        return multiplier

    # ---------- 1. base and adjusted qx ----------

    def get_risk_multiplier(self):
        return self._risk_multiplier

    def adjusted_qx(self, age):
        """adjusted_qx for a given absolute age (must be >= self.age)."""
        index = age - self.age
        return self._adjusted_qx_array[index]

    # ---------- 2-4. survival / death probabilities ----------

    def survival_prob_one_year(self, age):
        return 1 - self.adjusted_qx(age)

    def cumulative_survival_prob(self, years):
        """Probability of surviving `years` full years from self.age."""
        return self._cumulative_survival[years]

    def cumulative_death_prob(self, years):
        return 1 - self.cumulative_survival_prob(years)

    # ---------- 5. death in a specific year k ----------

    def prob_death_in_year(self, k):
        survive_prior_years = self.cumulative_survival_prob(k - 1)
        die_this_year = self.adjusted_qx(self.age + k - 1)
        return survive_prior_years * die_this_year

    # ---------- 6. discounting ----------

    def discount_factor(self, k):
        return 1 / ((1 + self.discount_rate) ** k)

    # ---------- 7-10. EPVs and premiums ----------

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

    # ---------- 11. variance / std dev of payout ----------

    def payout_variance(self):
        n_q_x = self.cumulative_death_prob(self.term_years)
        return (self.face_amount ** 2) * n_q_x * (1 - n_q_x)

    def payout_std_dev(self):
        return self.payout_variance() ** 0.5

    # ---------- 14. Monte Carlo simulation ----------

    def simulate_policy(self, n_simulations=10000, seed=None):
        """
        Vectorized Monte Carlo simulation. For each of n_simulations
        "parallel worlds", draws one random number per policy-year and
        compares it against that year's adjusted_qx, all at once with
        numpy instead of a Python-level double loop. Much faster than
        a naive year-by-year, simulation-by-simulation loop.
        """
        rng = np.random.default_rng(seed)

        qx_term = self._adjusted_qx_array[:self.term_years]  # shape (term_years,)
        # random numbers: shape (n_simulations, term_years)
        draws = rng.random((n_simulations, self.term_years))

        # True where death "occurs" in that policy-year, for each simulation
        died_this_year = draws < qx_term  # broadcasts qx_term across rows

        # For each simulation, find the FIRST year (smallest index) where
        # death occurred. If no death occurred, argmax returns 0 on an
        # all-False row, so we must separately check "did they ever die".
        ever_died = died_this_year.any(axis=1)
        year_of_death_index = died_this_year.argmax(axis=1)  # 0-based index
        year_of_death = year_of_death_index + 1  # convert to 1-based year k

        discount_factors = 1 / ((1 + self.discount_rate) ** year_of_death)
        pv_payouts = np.where(ever_died, self.face_amount * discount_factors, 0.0)

        return pv_payouts

    # ---------- summary ----------

    def summary(self):
        return {
            "net_premium": round(self.net_premium(), 2),
            "gross_premium": round(self.gross_premium(), 2),
            "epv_benefit": round(self.epv_benefit(), 2),
            "payout_std_dev": round(self.payout_std_dev(), 2),
        }
