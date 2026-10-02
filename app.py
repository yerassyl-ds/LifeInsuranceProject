import random

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from Calculations import Calculations
from Datas import Datas
from Portfolio import Portfolio

st.set_page_config(page_title="Life Insurance Pricing Engine", layout="wide")

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

RISK_FACTORS = Datas().load_risk_factors()


def options_for(category):
    """Returns the available class options for a risk-factor category,
    e.g. options_for('smoking') -> ['nonsmoker', 'smoker']."""
    return RISK_FACTORS.loc[category].index.tolist()


def random_policyholder():
    """Generates a policyholder with randomized (but realistic) attributes,
    used for the portfolio simulation demo."""
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


# ----------------------------------------------------------------------
# Sidebar: policyholder inputs
# ----------------------------------------------------------------------

st.sidebar.header("Policyholder Profile")

age = st.sidebar.slider("Age", 18, 90, 40)
sex = st.sidebar.selectbox("Sex", ["male", "female"])
smoking = st.sidebar.selectbox("Smoking", options_for("smoking"))
alcohol = st.sidebar.selectbox("Alcohol use", options_for("alcohol"))
extreme_sports = st.sidebar.selectbox("Extreme sports", options_for("extreme_sports"))
health_class = st.sidebar.selectbox("Health class", options_for("health_class"))

st.sidebar.header("Policy Terms")
term_years = st.sidebar.slider("Term (years)", 5, 40, 20)
face_amount = st.sidebar.number_input("Face amount ($)", min_value=10_000,
                                       max_value=2_000_000, value=100_000, step=10_000)

# ----------------------------------------------------------------------
# Main page
# ----------------------------------------------------------------------

st.title("Life Insurance Pricing Engine")
st.caption("A probability-theory-driven term life insurance pricing model "
           "(Gompertz-Makeham mortality, risk-adjusted premiums, Monte Carlo validation).")

try:
    policy = Calculations(age, sex, smoking, alcohol, extreme_sports,
                           health_class, term_years, face_amount)
except ValueError as e:
    st.error(str(e))
    st.stop()

tab1, tab2, tab3, tab4 = st.tabs(["Single Policy", "Portfolio Risk Pooling",
                                   "Charts", "About the Model"])

# ---------------- TAB 1: single policy ----------------
with tab1:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Net Premium / yr", f"${policy.net_premium():,.2f}")
    col2.metric("Gross Premium / yr", f"${policy.gross_premium():,.2f}")
    col3.metric("EPV of Benefit", f"${policy.epv_benefit():,.2f}")
    col4.metric("Payout Std Dev", f"${policy.payout_std_dev():,.2f}")

    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("Survival Curve")
        years = np.arange(0, term_years + 1)
        survival = [policy.cumulative_survival_prob(t) for t in years]
        fig, ax = plt.subplots()
        ax.plot(years, survival, marker="o")
        ax.set_xlabel("Years from now")
        ax.set_ylabel("Probability of survival")
        ax.set_ylim(0, 1.05)
        ax.set_title(f"$_tp_x$ for age {age}, {sex}")
        ax.grid(alpha=0.3)
        st.pyplot(fig)

    with right:
        st.subheader("Simulated Payout Distribution")
        n_sims = st.slider("Number of simulations", 1_000, 50_000, 10_000, step=1_000)
        sims = policy.simulate_policy(n_simulations=n_sims, seed=42)
        fig, ax = plt.subplots()
        ax.hist(sims, bins=40, color="steelblue", edgecolor="white")
        ax.axvline(policy.epv_benefit(), color="red", linestyle="--",
                   label=f"Analytic EPV = ${policy.epv_benefit():,.0f}")
        ax.set_xlabel("Present value of payout ($)")
        ax.set_ylabel("Number of simulations")
        ax.set_title("Monte Carlo simulated payout (single policy)")
        ax.legend()
        st.pyplot(fig)

        sim_mean = sims.mean()
        error_pct = abs(sim_mean - policy.epv_benefit()) / policy.epv_benefit() * 100
        st.caption(f"Simulated mean: ${sim_mean:,.2f}  |  "
                   f"Analytic EPV: ${policy.epv_benefit():,.2f}  |  "
                   f"Error: {error_pct:.2f}%")

# ---------------- TAB 2: portfolio ----------------
with tab2:
    st.subheader("Law of Large Numbers: risk pooling across policyholders")
    st.write("As the number of independent policyholders grows, the insurer's "
             "**relative** risk (coefficient of variation = std dev / mean) shrinks, "
             "even though total expected loss keeps growing.")

    seed_val = st.number_input("Random seed (for reproducibility)", value=42, step=1)
    max_n = st.select_slider("Largest portfolio size to test",
                              options=[100, 1_000, 5_000, 10_000], value=1_000)

    portfolio_sizes = [n for n in [1, 10, 100, 1_000, 5_000, 10_000] if n <= max_n]

    if st.button("Run portfolio analysis"):
        random.seed(seed_val)
        results = []
        progress = st.progress(0.0, text="Building portfolios...")

        for i, n in enumerate(portfolio_sizes):
            portfolio = Portfolio()
            portfolio.add_policies([random_policyholder() for _ in range(n)])
            results.append({
                "N": n,
                "expected_loss": portfolio.total_expected_loss(),
                "std_dev": portfolio.total_std_dev(),
                "coefficient_of_variation": portfolio.coefficient_of_variation(),
            })
            progress.progress((i + 1) / len(portfolio_sizes),
                             text=f"Computed N={n}")

        progress.empty()

        ns = [r["N"] for r in results]
        covs = [r["coefficient_of_variation"] for r in results]

        col1, col2 = st.columns(2)
        with col1:
            st.write("**Results table**")
            st.dataframe(
                {
                    "N (policies)": ns,
                    "Expected Loss ($)": [f"{r['expected_loss']:,.0f}" for r in results],
                    "Std Dev ($)": [f"{r['std_dev']:,.0f}" for r in results],
                    "Coeff. of Variation": [f"{r['coefficient_of_variation']:.4f}" for r in results],
                },
                hide_index=True,
            )

        with col2:
            fig, ax = plt.subplots()
            ax.plot(ns, covs, marker="o", color="darkorange")
            ax.set_xscale("log")
            ax.set_xlabel("Number of policyholders (log scale)")
            ax.set_ylabel("Coefficient of variation")
            ax.set_title("Relative risk shrinks as portfolio grows")
            ax.grid(alpha=0.3)
            st.pyplot(fig)

# ---------------- TAB 3: charts ----------------
with tab3:
    st.subheader("Mortality curve")
    st.write("Base yearly death probability ($q_x$) by age, before any "
             "risk-factor adjustment.")

    mortality_table = Datas().load_mortaility_table()
    fig, ax = plt.subplots()
    ax.plot(mortality_table.index, mortality_table["qx_male"], label="Male")
    ax.plot(mortality_table.index, mortality_table["qx_female"], label="Female")
    ax.set_xlabel("Age")
    ax.set_ylabel("$q_x$ (probability of death within one year)")
    ax.set_title("Base mortality curve (Gompertz-Makeham)")
    ax.legend()
    ax.grid(alpha=0.3)
    st.pyplot(fig)

    st.divider()

    st.subheader("Premium vs. age")
    st.write("How the net premium changes with age, holding this policyholder's "
             "other attributes fixed.")

    ages = list(range(25, 81, 5))
    premiums_by_age = []
    for a in ages:
        try:
            p = Calculations(a, sex, smoking, alcohol, extreme_sports,
                              health_class, term_years, face_amount)
            premiums_by_age.append(p.net_premium())
        except ValueError:
            premiums_by_age.append(None)

    fig, ax = plt.subplots()
    ax.plot(ages, premiums_by_age, marker="o", color="darkorange")
    ax.set_xlabel("Age")
    ax.set_ylabel("Net premium ($/yr)")
    ax.set_title(f"Premium vs. age ({sex}, {smoking}, {term_years}-yr term, "
                 f"${face_amount:,.0f})")
    ax.grid(alpha=0.3)
    st.pyplot(fig)

    st.divider()

    st.subheader("Premium vs. risk factor")
    st.write("How the net premium changes if you switch just one risk factor "
             "at a time, holding age and everything else fixed at your "
             "sidebar selections.")

    categories = ["smoking", "alcohol", "extreme_sports", "health_class"]
    selected_category = st.selectbox("Risk factor to vary", categories)

    classes = options_for(selected_category)
    premiums_by_class = []
    for cls in classes:
        kwargs = dict(smoking=smoking, alcohol=alcohol,
                       extreme_sports=extreme_sports, health_class=health_class)
        kwargs[selected_category] = cls
        p = Calculations(age, sex, kwargs["smoking"], kwargs["alcohol"],
                          kwargs["extreme_sports"], kwargs["health_class"],
                          term_years, face_amount)
        premiums_by_class.append(p.net_premium())

    fig, ax = plt.subplots()
    bars = ax.bar(classes, premiums_by_class, color="steelblue")
    current_value = {"smoking": smoking, "alcohol": alcohol,
                      "extreme_sports": extreme_sports,
                      "health_class": health_class}[selected_category]
    for bar, cls in zip(bars, classes):
        if cls == current_value:
            bar.set_color("darkorange")
    ax.set_xlabel(selected_category.replace("_", " ").title())
    ax.set_ylabel("Net premium ($/yr)")
    ax.set_title(f"Premium by {selected_category.replace('_', ' ')} "
                 f"(current selection highlighted)")
    plt.xticks(rotation=20, ha="right")
    st.pyplot(fig)

# ---------------- TAB 4: about ----------------
with tab4:
    st.subheader("Model overview")
    st.markdown("""
    - **Mortality**: base death probabilities ($q_x$) come from a Gompertz-Makeham
      mortality curve, separately calibrated for male/female.
    - **Risk adjustment**: individual multipliers for smoking, alcohol use,
      extreme sports, and overall health class are multiplied together and
      applied to the base $q_x$.
    - **Pricing**: the net premium is solved using the actuarial **equivalence
      principle** (expected present value of premiums = expected present value
      of the death benefit). A gross premium adds an expense/profit loading.
    - **Validation**: a Monte Carlo simulation independently estimates the same
      expected payout by simulating random policyholder lifetimes, to confirm
      the analytic formulas are correct (Law of Large Numbers).
    - **Portfolio pooling**: aggregating many independent policyholders shows
      how relative risk (coefficient of variation) shrinks as the pool grows —
      the statistical reason insurance is viable as a business.
    """)