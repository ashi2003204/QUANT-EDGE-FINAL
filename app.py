"""Quant Edge: a small, readable portfolio risk dashboard."""

from datetime import date
from io import StringIO
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.express as px
from scipy.optimize import minimize
from scipy.stats import chi2, genpareto, norm
from statsmodels.tsa.stattools import adfuller
from scipy.stats import binom

FUNDS = ["SPY", "EFA", "EEM", "AGG", "GLD", "XLE", "XLK"]
st.set_page_config(page_title="Endowment risk", layout="wide")
st.title("Serendib Endowment: portfolio and risk")
st.write("A plain-language look at one possible portfolio. All risk estimates use past daily returns.")

with st.sidebar:
    st.header("Choose dates and settings")
    start = st.date_input("Start date", date(2015, 1, 1))
    end = st.date_input("End date", date.today())
    confidence = st.selectbox("Confidence level", [0.90, 0.95, 0.99], index=1)
    threshold_percent = st.slider("EVT tail cutoff (%)", 90, 97, 95)


@st.cache_data(ttl=3600)
def get_prices(first_day, last_day):
    data = yf.download(FUNDS, start=first_day, end=last_day,
                       auto_adjust=True, progress=False)
    if data.empty:
        return pd.DataFrame()
    prices = data["Close"] if isinstance(data.columns, pd.MultiIndex) else data
    return prices.reindex(columns=FUNDS).ffill(limit=3).dropna()


@st.cache_data(ttl=86400)
def get_treasury_rate(first_day, last_day):
    """Download the daily 3-month Treasury yield from FRED, in percent per year."""
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS3MO"
    table = pd.read_csv(url, na_values=".")
    table["observation_date"] = pd.to_datetime(table["observation_date"])
    table = table.set_index("observation_date").loc[first_day:last_day]
    return pd.to_numeric(table["DGS3MO"], errors="coerce")


def minimum_risk_weights(returns):
    """Choose long-only weights with the smallest past variance, capped at 35%."""
    count = len(returns.columns)
    covariance = returns.cov().to_numpy() * 252
    result = minimize(lambda w: w @ covariance @ w,
                      np.repeat(1 / count, count), method="SLSQP",
                      bounds=[(0, .35)] * count,
                      constraints=[{"type": "eq", "fun": lambda w: sum(w)-1}])
    return result.x if result.success else np.repeat(1 / count, count)


def risk_numbers(returns, level):
    """Historical, normal, and simulated VaR and ES for one day."""
    losses = -returns
    tail = 1 - level
    var_hist = losses.quantile(level)
    es_hist = losses[losses >= var_hist].mean()
    average, daily_sd = returns.mean(), returns.std(ddof=1)
    var_normal = -average + norm.ppf(level) * daily_sd
    es_normal = -average + daily_sd * norm.pdf(norm.ppf(tail)) / tail
    simulated = np.random.default_rng(7).normal(average, daily_sd, 50000)
    var_mc = -np.quantile(simulated, tail)
    es_mc = -simulated[simulated <= -var_mc].mean()
    return {"Historical": (var_hist, es_hist),
            "Normal": (var_normal, es_normal),
            "Monte Carlo": (var_mc, es_mc)}


def kupiec_pvalue(breaches, expected_rate):
    n, x = len(breaches), int(sum(breaches))
    observed = max(x / n, 1e-10)
    expected_rate = max(expected_rate, 1e-10)
    lr = -2 * ((n-x)*np.log(1-expected_rate) + x*np.log(expected_rate)
               - (n-x)*np.log(1-observed) - x*np.log(observed))
    return chi2.sf(lr, 1)


def christoffersen_pvalue(breaches):
    """Test if VaR breaches tend to happen next to other breaches."""
    old, new = breaches[:-1], breaches[1:]
    n00, n01 = sum((old == 0) & (new == 0)), sum((old == 0) & (new == 1))
    n10, n11 = sum((old == 1) & (new == 0)), sum((old == 1) & (new == 1))
    p = (n01+n11) / max(len(new), 1)
    p0 = n01 / max(n00+n01, 1)
    p1 = n11 / max(n10+n11, 1)
    def log_likelihood(a, b, c):
        return a*np.log(max(b,1e-12)) + c*np.log(max(1-b,1e-12))
    lr = -2 * (log_likelihood(n01+n11, p, n00+n10)
              - log_likelihood(n01,p0,n00) - log_likelihood(n11,p1,n10))
    return chi2.sf(lr, 1)


try:
    prices = get_prices(start, end)
    treasury_yield = get_treasury_rate(start, end)
except Exception as error:
    st.error(f"Price download failed: {error}")
    st.stop()
if prices.empty or len(prices) < 260:
    st.error("Not enough price history. Check the dates and internet connection.")
    st.stop()

# Clean prices by aligning dates and allowing only short gaps to be filled.
returns = np.log(prices / prices.shift(1)).dropna()
weights = minimum_risk_weights(returns)
names = returns.columns
portfolio = returns @ weights
daily_rf = (treasury_yield / 100 / 252).reindex(returns.index).ffill().dropna()
returns = returns.loc[daily_rf.index]
portfolio = portfolio.loc[daily_rf.index]
daily_rf = daily_rf.loc[portfolio.index]
wealth = np.exp(portfolio.cumsum())
drawdown = wealth / wealth.cummax() - 1
yearly_return = portfolio.mean() * 252
yearly_risk = portfolio.std(ddof=1) * np.sqrt(252)
worst_fall = drawdown.min()
average_rf = daily_rf.mean() * 252
sharpe = (yearly_return - average_rf) / yearly_risk
benchmark = returns["SPY"]
active = portfolio - benchmark
beta = portfolio.cov(benchmark) / benchmark.var()
jensen_alpha = (portfolio - daily_rf).mean()*252 - beta*(benchmark-daily_rf).mean()*252
treynor = (yearly_return-average_rf)/beta if beta else np.nan
information = active.mean()*252/(active.std(ddof=1)*np.sqrt(252))
downside = np.sqrt(np.minimum(portfolio-daily_rf,0).pow(2).mean()*252)
sortino = (yearly_return-average_rf)/downside if downside else np.nan
calmar = yearly_return/abs(worst_fall) if worst_fall else np.nan

st.header("Portfolio from the selected dates")
st.caption("The app spreads the money across funds to reduce past volatility. Each fund is capped at 35%.")
a, b, c, d = st.columns(4)
a.metric("Past yearly return", f"{yearly_return:.1%}")
b.metric("Past yearly risk", f"{yearly_risk:.1%}")
c.metric("Largest fall", f"{worst_fall:.1%}")
d.metric("Sharpe ratio", f"{sharpe:.2f}")
st.caption(f"Sharpe uses the average FRED 3-month Treasury yield over these dates ({average_rf:.2%} annualized).")
with st.expander("More return measures"):
    st.write(f"Sortino: {sortino:.2f} · Treynor: {treynor:.2f} · Jensen alpha: {jensen_alpha:.2%} · Information ratio vs SPY: {information:.2f} · Calmar: {calmar:.2f}")
    st.caption("These measures use SPY as the market benchmark and are historical estimates.")
weights_table = pd.DataFrame({"Fund": names, "Share": weights})
st.plotly_chart(px.bar(weights_table, x="Fund", y="Share", title="How the $10m is split"), width="stretch")
st.plotly_chart(px.line(wealth, title="What $1 invested would have become"), width="stretch")
st.plotly_chart(px.area(drawdown, title="Falls from a previous high"), width="stretch")

st.header("Possible losses on one day")
st.write("VaR is a loss level that was exceeded on a small number of past days. ES is the average loss on those worse days.")
rows = []
for level in [.95, .99]:
    for method, (var, es) in risk_numbers(portfolio, level).items():
        rows.append({"Confidence": f"{level:.0%}", "Method": method,
                     "VaR on $10m": var*1e7, "ES on $10m": es*1e7})
st.dataframe(pd.DataFrame(rows).style.format({"VaR on $10m":"${:,.0f}","ES on $10m":"${:,.0f}"}), hide_index=True)

st.subheader("Very large losses (EVT)")
losses = -portfolio
cutoff = losses.quantile(threshold_percent/100)
excess = losses[losses > cutoff] - cutoff
if len(excess) >= 30:
    shape, _, scale = genpareto.fit(excess, floc=0)
    tail_rate = (1-confidence) / (len(excess)/len(losses))
    evt_var = cutoff + scale/shape*(tail_rate**(-shape)-1) if abs(shape)>1e-6 else cutoff-scale*np.log(tail_rate)
    st.write(f"EVT estimated loss at {confidence:.0%}: {evt_var:.2%}. It uses {len(excess)} large-loss days and changes when the cutoff changes.")
    # Fit the tail again at nearby cutoffs so readers can see if the answer jumps.
    threshold_rows=[]
    for pct in range(90, 98):
        limit=losses.quantile(pct/100)
        tail=losses[losses>limit]-limit
        if len(tail)>=30:
            shape2,_,scale2=genpareto.fit(tail,floc=0)
            q=(1-confidence)/(len(tail)/len(losses))
            estimate=limit+scale2/shape2*(q**(-shape2)-1) if abs(shape2)>1e-6 else limit-scale2*np.log(q)
            threshold_rows.append({"Cutoff percentile":pct,"EVT loss estimate":estimate})
    st.plotly_chart(px.line(pd.DataFrame(threshold_rows),x="Cutoff percentile",y="EVT loss estimate",title="Does the EVT estimate change with the cutoff?"),width="stretch")
    st.caption("This is a simple cutoff check, not a confidence interval. EVT can be unreliable when there are few tail observations.")
else:
    st.write("Too few large-loss days for this estimate.")

st.header("Check against later data")
split = int(.7*len(portfolio))
training, later = portfolio.iloc[:split], portfolio.iloc[split:]
forecast = -training.quantile(1-confidence)
breaches = (-later > forecast).astype(int).to_numpy()
st.write(f"VaR breaches: {int(sum(breaches))} of {len(breaches)} later days.")
st.write(f"Kupiec test p-value: {kupiec_pvalue(breaches,1-confidence):.3f}")
st.write(f"Christoffersen independence p-value: {christoffersen_pvalue(breaches):.3f}")
exceptions = int(sum(breaches))
if len(breaches) == 250:
    green_limit, yellow_limit = 4, 9
else:
    # Use Basel's 99% VaR binomial rule for the actual test sample length.
    green_limit = int(binom.ppf(.95, len(breaches), .01))
    yellow_limit = int(binom.ppf(.9999, len(breaches), .01))
traffic = "Green" if exceptions <= green_limit else "Yellow" if exceptions <= yellow_limit else "Red"
st.write(f"Basel-style traffic light: **{traffic}** ({exceptions} exceptions). This is an educational indicator, not a regulatory capital result.")
st.caption("A low p-value can mean the model did not match what happened. A passed test does not prove the model is right.")

st.header("Crisis examples")
stress = []
for label, first_day, last_day in [
    ("2008 crisis", "2008-09-01", "2009-03-31"),
    ("2020 COVID crash", "2020-02-19", "2020-03-23"),
    ("2022 rate rises", "2022-01-03", "2022-10-12")]:
    crisis = returns.loc[first_day:last_day]
    if len(crisis):
        stress.append({"Scenario":label,"Portfolio change":np.exp((crisis@weights).sum())-1})
shock = pd.Series({fund:(-.15 if fund in ["SPY","EFA","EEM","XLK"] else -.10 if fund=="XLE" else -.04 if fund=="AGG" else .05) for fund in names})
stress.append({"Scenario":"Made-up: shares down, gold up","Portfolio change":weights@shock})
st.dataframe(pd.DataFrame(stress).style.format({"Portfolio change":"{:.1%}"}), hide_index=True)

st.header("Where the portfolio risk comes from")
daily_cov=returns.cov().to_numpy()
portfolio_sd=np.sqrt(weights @ daily_cov @ weights)
risk_shares=weights*(daily_cov@weights)/portfolio_sd
risk_table=pd.DataFrame({"Fund":names,"Weight":weights,"Share of daily risk":risk_shares/risk_shares.sum()})
st.dataframe(risk_table.style.format({"Weight":"{:.1%}","Share of daily risk":"{:.1%}"}),hide_index=True)
st.write("A fund's share of daily risk can be larger or smaller than its share of the money. This table is a simple risk budget.")

st.header("Data and things to check")
adf_rows=[]
for fund in names:
    pvalue=adfuller(returns[fund])[1]
    adf_rows.append({"Fund":fund,"ADF p-value":pvalue,"Looks stationary at 5%":pvalue<.05})
st.dataframe(pd.DataFrame(adf_rows),hide_index=True)
st.write("Prices use Yahoo Finance adjusted close. Dates are aligned; gaps of up to three days are filled. Review unusual jumps before using results.")
st.write("Risk-free returns use FRED's daily DGS3MO series, converted from annual percent to a daily rate and aligned to ETF dates. The optimizer uses past covariance, which may not describe the future. The made-up crisis is only an example.")
st.write("The app is a competition project, not investment advice. The team must check the numbers and explain the choices.")
