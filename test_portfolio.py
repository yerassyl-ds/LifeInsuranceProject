"""
Demonstrates the core statistical insight behind insurance:
as the number of independent policyholders (N) grows, the insurer's
RELATIVE risk (coefficient of variation = std_dev / mean) shrinks,
even though the total expected loss just keeps growing proportionally.

This is the Law of Large Numbers in action: pooling risk is what makes
insurance viable as a business.
"""

import random
from Calculations import Calculations
from Portfolio import Portfolio


def random_policyholder():
    """Generates a policyholder with randomized (but realistic) attributes,
    so the portfolio isn't just N identical clones of one person."""
    age = random.randint(25, 60)
    sex = random.choice(["male", "female"])
    smoking = random.choices(["nonsmoker", "smoker"], weights=[0.8, 0.2])[0]
    alcohol = random.choices(["none_moderate", "heavy"], weights=[0.85, 0.15])[0]
    extreme_sports = random.choices(
        ["none", "occasional", "regular"], weights=[0.7, 0.25, 0.05]
    )[0]
    health_class = random.choices(
        ["preferred_plus", "preferred", "standard", "substandard", "high_risk"],
        weights=[0.1, 0.25, 0.45, 0.15, 0.05],
    )[0]
    term_years = random.choice([10, 15, 20, 25])
    face_amount = random.choice([50_000, 100_000, 150_000, 250_000])

    return Calculations(age, sex, smoking, alcohol, extreme_sports,
                         health_class, term_years, face_amount)


def section(title):
    print("\n" + "=" * 65)
    print(title)
    print("=" * 65)


# ---------------------------------------------------------------
# 1. Build portfolios of increasing size and watch the
#    coefficient of variation shrink.
# ---------------------------------------------------------------
section("1. Coefficient of variation shrinks as portfolio size grows")

random.seed(42)  # reproducible portfolio composition

portfolio_sizes = [1, 10, 100, 1_000, 10_000]

print(f"{'N':>7} | {'expected loss':>15} | {'std dev':>15} | {'coeff of variation':>20}")
print("-" * 65)

for n in portfolio_sizes:
    portfolio = Portfolio()
    portfolio.add_policies([random_policyholder() for _ in range(n)])

    expected_loss = portfolio.total_expected_loss()
    std_dev = portfolio.total_std_dev()
    cov = portfolio.coefficient_of_variation()

    print(f"{n:>7} | {expected_loss:>15,.2f} | {std_dev:>15,.2f} | {cov:>20.4f}")


# ---------------------------------------------------------------
# 2. Sanity check: does premium income exceed expected loss?
#    (It should, since gross premium includes the expense loading —
#    this is the insurer's built-in margin.)
# ---------------------------------------------------------------
section("2. Premium income vs expected loss (solvency check)")

random.seed(7)
portfolio = Portfolio()
portfolio.add_policies([random_policyholder() for _ in range(1_000)])

# Note: gross_premium is an ANNUAL amount, while total_expected_loss is a
# present value over the full term -- these aren't directly comparable
# apples-to-apples without further discounting, but this print is useful
# to see the raw magnitude relationship.
print(f"Total expected loss (PV):        {portfolio.total_expected_loss():,.2f}")
print(f"Total annual gross premium income: {portfolio.total_gross_premium_income():,.2f}")


# ---------------------------------------------------------------
# 3. Monte Carlo simulation of total portfolio loss, for a few sizes,
#    to visually/empirically confirm the same shrinking-relative-risk effect.
# ---------------------------------------------------------------
section("3. Monte Carlo simulation of total portfolio loss")

random.seed(1)
for n in [10, 100, 1_000]:
    portfolio = Portfolio()
    portfolio.add_policies([random_policyholder() for _ in range(n)])

    sims = portfolio.simulate_portfolio(n_simulations=1000, seed=123)
    sim_mean = sims.mean()
    sim_std = sims.std()
    sim_cov = sim_std / sim_mean if sim_mean != 0 else float("nan")

    print(f"\nN={n}")
    print(f"  Analytic:  expected loss = {portfolio.total_expected_loss():,.2f}, "
          f"CoV = {portfolio.coefficient_of_variation():.4f}")
    print(f"  Simulated: mean loss     = {sim_mean:,.2f}, "
          f"CoV = {sim_cov:.4f}")


# ---------------------------------------------------------------
# 4. Full summary for one portfolio.
# ---------------------------------------------------------------
section("4. Full summary for a 500-policy portfolio")

random.seed(99)
portfolio = Portfolio()
portfolio.add_policies([random_policyholder() for _ in range(500)])
for key, value in portfolio.summary().items():
    print(f"{key}: {value}")
