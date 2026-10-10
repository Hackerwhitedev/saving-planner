from dataclasses import dataclass, field
import io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
import pandas as pd
import plotly.express as px
import streamlit as st

INFLATION = 0.06

ASSET_RETURNS = {"Equity": 0.12, "Debt": 0.07, "Gold": 0.08, "Liquid": 0.05}
SCENARIO_SHIFT = {"Pessimistic": -0.03, "Expected": 0.0, "Optimistic": 0.03}

BASE_ALLOCATION = {
    "Conservative": {"Equity": 20, "Debt": 55, "Gold": 10, "Liquid": 15},
    "Moderate": {"Equity": 50, "Debt": 30, "Gold": 10, "Liquid": 10},
    "Aggressive": {"Equity": 75, "Debt": 12, "Gold": 8, "Liquid": 5},
}

RISK_QUESTIONS = [
    ("What is your age group?",
     ["Above 40", "30 to 40", "Below 30"]),
    ("When do you expect to need most of this money?",
     ["Within 3 years", "3 to 7 years", "More than 7 years"]),
    ("Your portfolio falls 20% in a month. You would...",
     ["Sell everything", "Hold and wait", "Invest more"]),
    ("How stable is your income?",
     ["Variable or uncertain", "Fairly stable", "Very stable"]),
    ("How many people depend on your income?",
     ["Three or more", "One or two", "None"]),
    ("Your experience with investing?",
     ["None", "Some (FD, savings)", "Mutual funds or stocks"]),
]

DISCLAIMER = ("This report is for educational purposes only and is not licensed financial advice. "
              "Returns are assumptions, not guarantees. Consult a SEBI-registered advisor before investing.")


def risk_profile(answers):
    score = sum(a + 1 for a in answers)
    max_score = 3 * len(answers)
    ratio = score / max_score
    if ratio < 0.5:
        label = "Conservative"
    elif ratio < 0.75:
        label = "Moderate"
    else:
        label = "Aggressive"
    return label, score, max_score


def investable_surplus(income, expenses):
    return max(income - expenses, 0.0)


def horizon_adjusted_allocation(risk, months):
    alloc = dict(BASE_ALLOCATION[risk])
    years = months / 12
    if years <= 2:
        cap = 0
    elif years <= 3:
        cap = 20
    elif years <= 5:
        cap = 40
    elif years <= 7:
        cap = 60
    else:
        cap = 100
    if alloc["Equity"] > cap:
        excess = alloc["Equity"] - cap
        alloc["Equity"] = cap
        target = "Liquid" if years <= 2 else "Debt"
        alloc[target] += excess
    return alloc


def blended_return(alloc, shift=0.0):
    return sum((ASSET_RETURNS[a] + shift) * w / 100 for a, w in alloc.items())


def future_value_of_sip(sip, annual_rate, months):
    r = annual_rate / 12
    if months <= 0:
        return 0.0
    if r == 0:
        return sip * months
    return sip * (((1 + r) ** months - 1) / r)


def required_sip(target_fv, annual_rate, months):
    if months <= 0:
        return float(target_fv)
    r = annual_rate / 12
    if r == 0:
        return target_fv / months
    return target_fv * r / ((1 + r) ** months - 1)


def months_needed(target_fv, sip, annual_rate, cap=600):
    if sip <= 0:
        return None
    for n in range(1, cap + 1):
        if future_value_of_sip(sip, annual_rate, n) >= target_fv:
            return n
    return None


@dataclass
class Goal:
    name: str
    target_today: float
    months: int
    is_emergency: bool = False


@dataclass
class GoalPlan:
    goal: Goal
    target_future: float
    allocation: dict
    expected_return: float
    sip: float
    sip_by_asset: dict = field(default_factory=dict)


def build_goal_plans(goals, risk, monthly_expenses, existing_savings):
    plans = []
    savings_left = existing_savings
    for g in goals:
        if g.is_emergency:
            alloc = {"Equity": 0, "Debt": 0, "Gold": 0, "Liquid": 100}
            target = g.target_today
            gap = max(target - savings_left, 0)
            savings_left = max(savings_left - target, 0)
            ret = blended_return(alloc)
            sip = required_sip(gap, ret, g.months)
            plans.append(GoalPlan(g, target, alloc, ret, sip))
        else:
            alloc = horizon_adjusted_allocation(risk, g.months)
            ret = blended_return(alloc)
            fut = g.target_today * (1 + INFLATION) ** (g.months / 12)
            sip = required_sip(fut, ret, g.months)
            plans.append(GoalPlan(g, fut, alloc, ret, sip))
    for p in plans:
        p.sip_by_asset = {a: p.sip * w / 100 for a, w in p.allocation.items()}
    return plans


def feasibility(plans, surplus):
    total = sum(p.sip for p in plans)
    result = {"total_sip": total, "surplus": surplus, "feasible": total <= surplus + 1e-9,
              "gap": max(total - surplus, 0.0), "suggestions": []}
    if result["feasible"]:
        return result
    emergency = sum(p.sip for p in plans if p.goal.is_emergency)
    others = [p for p in plans if not p.goal.is_emergency]
    budget = max(surplus - emergency, 0)
    other_total = sum(p.sip for p in others) or 1
    for p in others:
        share = budget * (p.sip / other_total)
        n = months_needed(p.target_future, share, p.expected_return)
        if n is None:
            result["suggestions"].append(
                f"'{p.goal.name}': not reachable at the budget available. Reduce the target or increase income.")
        else:
            result["suggestions"].append(
                f"'{p.goal.name}': extend from {p.goal.months} to about {n} months "
                f"(SIP of about Rs {share:,.0f}/month), or reduce the target.")
    if budget == 0:
        result["suggestions"].insert(
            0, "Surplus does not even cover the emergency fund SIP. Lengthen the emergency fund timeline first.")
    return result


def scale_to_budget(plans, surplus):
    total = sum(p.sip for p in plans)
    if total <= surplus or total == 0:
        return {p.goal.name: p.sip for p in plans}
    emergency = sum(p.sip for p in plans if p.goal.is_emergency)
    others_total = total - emergency
    budget = max(surplus - emergency, 0)
    factor = budget / others_total if others_total else 0
    return {p.goal.name: (p.sip if p.goal.is_emergency else p.sip * factor) for p in plans}


def aggregate_allocation(plans, sips=None):
    totals = {a: 0.0 for a in ASSET_RETURNS}
    for p in plans:
        sip = sips[p.goal.name] if sips else p.sip
        for a, w in p.allocation.items():
            totals[a] += sip * w / 100
    return totals


def projection(plans, sips, years):
    months = int(years * 12)
    out = {}
    for scen, shift in SCENARIO_SHIFT.items():
        series = [0.0] * (months + 1)
        for p in plans:
            r = blended_return(p.allocation, shift) / 12
            sip = sips[p.goal.name]
            bal = 0.0
            for m in range(1, months + 1):
                if m <= p.goal.months:
                    bal = bal * (1 + r) + sip
                else:
                    bal = bal * (1 + r)
                series[m] += bal
        out[scen] = series
    return out


SAMPLE_PERSONA = {
    "income": 60000, "expenses": 32000, "savings": 40000,
    "answers": [2, 1, 1, 1, 2, 0],
    "goals": [
        Goal("Emergency fund", 6 * 32000, 12, True),
        Goal("Bike", 150000, 24),
        Goal("House down payment", 1500000, 84),
    ],
}


def _fig_to_image(fig, width_cm=16):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    img = Image(buf)
    ratio = img.imageHeight / img.imageWidth
    img.drawWidth = width_cm * cm
    img.drawHeight = width_cm * cm * ratio
    return img


def allocation_chart(totals):
    labels = [k for k, v in totals.items() if v > 0]
    values = [totals[k] for k in labels]
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    ax.pie(values, labels=labels, autopct="%1.0f%%", startangle=90,
           colors=["#534AB7", "#1D9E75", "#BA7517", "#888780"][: len(labels)])
    ax.set_title("Monthly SIP split by asset class")
    return fig


def projection_chart(proj):
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    for scen, series in proj.items():
        ax.plot([m / 12 for m in range(len(series))], [v / 1e5 for v in series], label=scen)
    ax.set_xlabel("Years")
    ax.set_ylabel("Corpus (Rs lakh)")
    ax.set_title("Projected corpus under three return scenarios")
    ax.legend()
    ax.grid(alpha=0.3)
    return fig


def build_pdf(name, income, expenses, surplus, risk, plans, sips, feas, totals, proj):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                            topMargin=2 * cm, bottomMargin=2 * cm)
    styles = getSampleStyleSheet()
    el = [Paragraph(f"Savings and investment plan{' for ' + name if name else ''}", styles["Title"]), Spacer(1, 8)]

    summary = [["Monthly income", f"Rs {income:,.0f}"], ["Monthly expenses", f"Rs {expenses:,.0f}"],
               ["Investable surplus", f"Rs {surplus:,.0f}"], ["Risk profile", risk],
               ["Total SIP planned", f"Rs {sum(sips.values()):,.0f} per month"]]
    t = Table(summary, colWidths=[6 * cm, 9 * cm])
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                           ("BACKGROUND", (0, 0), (0, -1), colors.whitesmoke)]))
    el += [t, Spacer(1, 12)]

    if not feas["feasible"]:
        el.append(Paragraph(f"<b>Heads up:</b> required SIPs exceed your surplus by Rs {feas['gap']:,.0f} per month. "
                            "SIPs below are scaled to fit, with the emergency fund protected.", styles["BodyText"]))
        for s in feas["suggestions"]:
            el.append(Paragraph("- " + s, styles["BodyText"]))
        el.append(Spacer(1, 10))

    el.append(Paragraph("SIP schedule by goal", styles["Heading2"]))
    rows = [["Goal", "Months", "Target (future)", "Asset mix (E/D/G/L)", "SIP per month"]]
    for p in plans:
        a = p.allocation
        rows.append([p.goal.name, str(p.goal.months), f"Rs {p.target_future:,.0f}",
                     f"{a['Equity']}/{a['Debt']}/{a['Gold']}/{a['Liquid']}", f"Rs {sips[p.goal.name]:,.0f}"])
    t = Table(rows, repeatRows=1)
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                           ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey), ("FONTSIZE", (0, 0), (-1, -1), 9)]))
    el += [t, Spacer(1, 12), _fig_to_image(allocation_chart(totals), 9), Spacer(1, 8),
           _fig_to_image(projection_chart(proj), 15), Spacer(1, 12)]

    el.append(Paragraph("Tax tips to explore", styles["Heading2"]))
    el.append(Paragraph("ELSS funds (Section 80C, old regime), NPS (extra deduction under 80CCD(1B)). "
                        "Check which tax regime suits you before investing.", styles["BodyText"]))
    el += [Spacer(1, 12), Paragraph(f"<i>{DISCLAIMER}</i>", styles["BodyText"])]
    doc.build(el)
    return buf.getvalue()


st.set_page_config(page_title="New employee savings planner", page_icon=":moneybag:", layout="wide")
st.title("Savings, SIP and investment planner")
st.caption("For new employees. " + DISCLAIMER)

if "sample" not in st.session_state:
    st.session_state.sample = False

if st.sidebar.button("Load sample persona"):
    st.session_state.sample = True
    for k in list(st.session_state.keys()):
        if k.startswith("q") or k in ("income", "expenses", "savings", "n_goals"):
            del st.session_state[k]
sample = SAMPLE_PERSONA if st.session_state.sample else None

tab1, tab2, tab3, tab4 = st.tabs(["1. Income", "2. Risk profile", "3. Goals", "4. Plan and report"])

with tab1:
    name = st.text_input("Your name (optional)", value="Asha" if sample else "")
    c1, c2, c3 = st.columns(3)
    income = c1.number_input("Monthly take-home (Rs)", 0, 5_000_000, sample["income"] if sample else 50000, 1000, key="income")
    expenses = c2.number_input("Monthly fixed expenses incl. rent and EMIs (Rs)", 0, 5_000_000,
                               sample["expenses"] if sample else 30000, 1000, key="expenses")
    savings = c3.number_input("Existing savings (Rs)", 0, 50_000_000, sample["savings"] if sample else 0, 5000, key="savings")
    surplus = investable_surplus(income, expenses)
    st.metric("Investable surplus per month", f"Rs {surplus:,.0f}")

with tab2:
    answers = []
    for i, (q, opts) in enumerate(RISK_QUESTIONS):
        default = sample["answers"][i] if sample else 1
        answers.append(opts.index(st.radio(q, opts, index=default, key=f"q{i}", horizontal=True)))
    risk, score, mx = risk_profile(answers)
    st.metric("Risk profile", risk, f"score {score} of {mx}")

with tab3:
    st.write("The emergency fund (6 months of expenses) is always included and funded first.")
    n_goals = st.number_input("Other goals", 0, 6, 2 if sample else 1, key="n_goals")
    goals = [Goal("Emergency fund", 6 * expenses, st.slider("Emergency fund timeline (months)", 3, 36, 12), True)]
    for i in range(int(n_goals)):
        d = sample["goals"][i + 1] if sample and i + 1 < len(sample["goals"]) else None
        c1, c2, c3 = st.columns([2, 2, 1])
        gname = c1.text_input("Goal", d.name if d else f"Goal {i + 1}", key=f"gn{i}")
        gt = c2.number_input("Target in today's money (Rs)", 0, 100_000_000, int(d.target_today) if d else 200000, 10000, key=f"gt{i}")
        gm = c3.number_input("Months", 6, 480, d.months if d else 36, 6, key=f"gm{i}")
        goals.append(Goal(gname, gt, int(gm)))

with tab4:
    plans = build_goal_plans(goals, risk, expenses, savings)
    feas = feasibility(plans, surplus)
    sips = scale_to_budget(plans, surplus)
    totals = aggregate_allocation(plans, sips)
    horizon = max(g.months for g in goals) / 12
    proj = projection(plans, sips, horizon)

    if feas["feasible"]:
        st.success(f"Plan fits: Rs {feas['total_sip']:,.0f} of Rs {surplus:,.0f} surplus is used.")
    else:
        st.warning(f"Required SIPs total Rs {feas['total_sip']:,.0f}, which is Rs {feas['gap']:,.0f} more than your surplus. "
                   "The plan below scales SIPs to fit (emergency fund protected). To stay on target:")
        for s in feas["suggestions"]:
            st.write("- " + s)

    rows = [{"Goal": p.goal.name, "Months": p.goal.months, "Target (future Rs)": round(p.target_future),
             "Equity %": p.allocation["Equity"], "Debt %": p.allocation["Debt"],
             "Gold %": p.allocation["Gold"], "Liquid %": p.allocation["Liquid"],
             "Needed SIP": round(p.sip), "Planned SIP": round(sips[p.goal.name])} for p in plans]
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    c1, c2 = st.columns(2)
    pie = px.pie(names=[k for k, v in totals.items() if v > 0], values=[v for v in totals.values() if v > 0],
                 title="Monthly SIP by asset class")
    c1.plotly_chart(pie, use_container_width=True)
    pdf_df = pd.DataFrame({"Year": [m / 12 for m in range(len(proj["Expected"]))], **{k: v for k, v in proj.items()}})
    line = px.line(pdf_df, x="Year", y=list(proj.keys()), title="Projected corpus (Rs)")
    c2.plotly_chart(line, use_container_width=True)

    st.caption("Assumed yearly returns: Equity 12%, Debt 7%, Gold 8%, Liquid 5% (expected case, plus or minus 3 points for the other scenarios). "
               "Goal targets are inflated at 6% per year.")
    pdf = build_pdf(name, income, expenses, surplus, risk, plans, sips, feas, totals, proj)
    st.download_button("Download PDF report", pdf, file_name="savings_plan.pdf", mime="application/pdf")