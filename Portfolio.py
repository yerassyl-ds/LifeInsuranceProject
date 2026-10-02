from Calculations import Calculations
import numpy as np


class Portfolio:
    """
    Holds a collection of policyholders (Calculations instances) and
    aggregates their risk to demonstrate how pooling many independent
    policies makes the insurer's total loss far more predictable than
    any single policy's outcome.
    """

    def __init__(self):
        self.policies = []

    def add_policy(self, policy: Calculations):
        self.policies.append(policy)

    def add_policies(self, policies):
        self.policies.extend(policies)

    def n_policies(self):
        return len(self.policies)

    # ---------- 12. portfolio-level expected loss and variance ----------

    def total_expected_loss(self):
        # Expectation is additive regardless of independence
        return sum(p.epv_benefit() for p in self.policies)

    def total_variance(self):
        # Variance is additive ONLY if policies are independent.
        # We assume independence here (a standard simplifying assumption:
        # one person's death doesn't affect another's, ignoring pandemics/
        # catastrophic events that would break this).
        return sum(p.payout_variance() for p in self.policies)

    def total_std_dev(self):
        return self.total_variance() ** 0.5

    # ---------- 13. relative risk / coefficient of variation ----------

    def coefficient_of_variation(self):
        mean = self.total_expected_loss()
        if mean == 0:
            return float("nan")
        return self.total_std_dev() / mean

    # ---------- total premium income (for solvency comparison) ----------

    def total_gross_premium_income(self):
        # Sum of one year's worth of gross premiums across all policyholders
        return sum(p.gross_premium() for p in self.policies)

    # ---------- 14. Monte Carlo simulation at portfolio level ----------

    def simulate_portfolio(self, n_simulations=1000, seed=None):
        """
        Simulates the TOTAL portfolio payout n_simulations times.
        Each simulation = one random draw of "does each policyholder die
        within their term, and in which year" -> sum of all their
        discounted payouts = one total portfolio loss outcome.

        Returns an array of length n_simulations: total portfolio loss
        per simulated "world".

        Uses each policy's vectorized simulate_policy() (numpy-based)
        instead of a raw Python triple loop, which is what made this
        too slow at portfolio sizes of 1000+.
        """
        rng = np.random.default_rng(seed)
        totals = np.zeros(n_simulations)

        for policy in self.policies:
            # A different seed per policy (derived from the main rng) keeps
            # policies from being perfectly correlated with each other.
            policy_seed = rng.integers(0, 2**32 - 1)
            pv_payouts = policy.simulate_policy(n_simulations=n_simulations,
                                                 seed=int(policy_seed))
            totals += pv_payouts

        return totals

    # ---------- summary ----------

    def summary(self):
        return {
            "n_policies": self.n_policies(),
            "total_expected_loss": round(self.total_expected_loss(), 2),
            "total_std_dev": round(self.total_std_dev(), 2),
            "coefficient_of_variation": round(self.coefficient_of_variation(), 4),
            "total_gross_premium_income": round(self.total_gross_premium_income(), 2),
        }
