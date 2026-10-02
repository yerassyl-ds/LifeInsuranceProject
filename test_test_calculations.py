"""
Sanity-check script for the Calculations class.

This is not a formal unit-test suite (no pytest/assert-based pass-fail) —
it's meant to print out numbers you can visually inspect to confirm the
class behaves the way probability theory says it should. Read the
"expectation" comment above each block and compare it to the printed output.
"""

from Calculations import Calculations


def make_policy(age=40, sex="male", smoking="nonsmoker", alcohol="none_moderate",
                 extreme_sports="none", health_class="standard",
                 term_years=20, face_amount=100_000):
    return Calculations(age, sex, smoking, alcohol, extreme_sports,
                         health_class, term_years, face_amount)


def section(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


# ---------------------------------------------------------------
# 1. Does adjusted_qx increase with age?
# Expectation: qx should rise (or at least not fall) as age increases,
# per the Gompertz-Makeham mortality curve.
# ---------------------------------------------------------------
section("1. adjusted_qx should increase with age")
policy = make_policy(age=40)
for test_age in [40, 50, 60, 70, 80]:
    print(f"age {test_age}: qx = {policy.adjusted_qx(test_age):.6f}")


# ---------------------------------------------------------------
# 2. Does a smoker have a higher premium than a nonsmoker, all else equal?
# Expectation: smoker's net/gross premium > nonsmoker's.
# ---------------------------------------------------------------
section("2. Smoker vs nonsmoker premium comparison")
nonsmoker = make_policy(smoking="nonsmoker")
smoker = make_policy(smoking="smoker")
print(f"Nonsmoker net premium: {nonsmoker.net_premium():.2f}")
print(f"Smoker    net premium: {smoker.net_premium():.2f}")
print(f"Smoker premium is higher: {smoker.net_premium() > nonsmoker.net_premium()}")


# ---------------------------------------------------------------
# 3. Does risk stacking behave as expected (extreme sports + health class)?
# Expectation: worse health class and regular extreme sports increases premium further.
# ---------------------------------------------------------------
section("3. Risk factor stacking")
low_risk = make_policy(smoking="nonsmoker", alcohol="none_moderate",
                        extreme_sports="none", health_class="preferred_plus")
high_risk = make_policy(smoking="smoker", alcohol="heavy",
                         extreme_sports="regular", health_class="high_risk")
print(f"Low-risk profile net premium:  {low_risk.net_premium():.2f}")
print(f"High-risk profile net premium: {high_risk.net_premium():.2f}")


# ---------------------------------------------------------------
# 4. Equivalence principle check.
# Expectation: EPV(premiums) * net_premium should equal EPV(benefit),
# since net_premium was solved for exactly that condition.
# This just confirms there's no bug in the algebra.
# ---------------------------------------------------------------
section("4. Equivalence principle check (should match closely)")
policy = make_policy()
epv_benefit = policy.epv_benefit()
epv_premiums_check = policy.epv_premiums_per_dollar() * policy.net_premium()
print(f"EPV(benefit):                     {epv_benefit:.4f}")
print(f"EPV(premiums) * net_premium:      {epv_premiums_check:.4f}")
print(f"Difference:                       {abs(epv_benefit - epv_premiums_check):.8f}")


# ---------------------------------------------------------------
# 5. Monte Carlo simulation vs analytic EPV.
# Expectation: mean of simulated PV payouts should converge close to
# epv_benefit() as n_simulations grows large (Law of Large Numbers).
# ---------------------------------------------------------------
section("5. Monte Carlo simulation vs analytic EPV(benefit)")
policy = make_policy()
analytic_epv = policy.epv_benefit()
print(f"Analytic EPV(benefit): {analytic_epv:.2f}\n")

for n in [100, 1_000, 10_000]:
    sims = policy.simulate_policy(n_simulations=n, seed=42)
    sim_mean = sims.mean()
    pct_error = abs(sim_mean - analytic_epv) / analytic_epv * 100
    print(f"n={n:>6}: simulated mean = {sim_mean:.2f}  "
          f"(error vs analytic: {pct_error:.2f}%)")


# ---------------------------------------------------------------
# 6. Full summary printout for one policy.
# ---------------------------------------------------------------
section("6. Full summary for a sample policy")
policy = make_policy(age=35, sex="female", smoking="nonsmoker",
                      alcohol="none_moderate", extreme_sports="occasional",
                      health_class="preferred", term_years=25, face_amount=250_000)
for key, value in policy.summary().items():
    print(f"{key}: {value}")