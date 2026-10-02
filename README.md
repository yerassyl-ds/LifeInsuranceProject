# Life Insurance Pricing Engine

A term life insurance pricing model built for Theory of Probability and
Statistics. Computes risk-adjusted mortality probabilities, net/gross
premiums (via the actuarial equivalence principle), and validates the
pricing formulas with Monte Carlo simulation. Also demonstrates the Law
of Large Numbers by pooling many policyholders into a portfolio and
showing relative risk shrink as the pool grows.

## Project structure

```
.
├── app.py                  # Streamlit web app (UI layer)
├── Calculations.py         # Core pricing/probability engine (single policy)
├── Portfolio.py            # Aggregates many policies, pooled risk analysis
├── Datas.py                # Loads CSV data (mortality table, risk factors, constants)
├── Data/
│   ├── mortality_table.csv
│   ├── risk_factors.csv
│   └── policy_constants.csv
├── test_calculations.py    # Sanity checks for Calculations
├── test_portfolio.py       # Sanity checks for Portfolio
└── requirements.txt
```

## Running locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

This opens the app in your browser at `http://localhost:8501`.

## Deploying online (Streamlit Community Cloud — free)

1. Push this whole project to a **public** GitHub repository (or add your
   professor as a collaborator if you want it private).
   Make sure `app.py`, `requirements.txt`, and the `Data/` folder with all
   three CSVs are committed — the app will fail to load without the CSVs.

2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with
   your GitHub account.

3. Click **"New app"**, then select:
   - Repository: your repo
   - Branch: `main` (or whichever branch has the code)
   - Main file path: `app.py`

4. Click **Deploy**. Streamlit Cloud will install everything from
   `requirements.txt` and launch the app automatically.

5. You'll get a public URL like
   `https://your-app-name.streamlit.app` — this is what you send your
   professor.

Any time you push new commits to the connected branch, the deployed app
updates automatically.

## Notes

- The mortality table is synthetically generated using the
  Gompertz-Makeham law of mortality, not pulled from a real published
  table — this is called out explicitly so it can be cited correctly in
  the project report.
- Risk factors (smoking, alcohol, extreme sports, health class) are
  combined multiplicatively, assuming independence between factors — a
  standard simplifying assumption, also worth noting in the report.
