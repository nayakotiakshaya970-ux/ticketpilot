import streamlit as st
import joblib
import pandas as pd
from datetime import datetime, timedelta

st.set_page_config(page_title="TicketPilot", page_icon="🎫", layout="centered")

@st.cache_resource
def load_models():
    vectorizer = joblib.load("models/vectorizer.pkl")
    category_model = joblib.load("models/category_model.pkl")
    urgency_model = joblib.load("models/urgency_model.pkl")
    return vectorizer, category_model, urgency_model

vectorizer, category_model, urgency_model = load_models()

routing_map = {
    "Billing": "Finance Team",
    "Technical": "Engineering Team",
    "Account": "Account Support Team",
    "General Inquiry": "Customer Success Team",
}

# SLA: hours allowed to resolve a ticket, by urgency
SLA_HOURS = {"High": 4, "Medium": 24, "Low": 72}

def get_sla_status(deadline, now):
    """Return (status label, hours remaining) based on % of SLA time left."""
    remaining_h = (deadline - now).total_seconds() / 3600
    if remaining_h <= 0:
        return "🔴 Breached", remaining_h
    total_h = SLA_HOURS.get("High", 4)  # placeholder, overwritten below
    return None, remaining_h

def sla_status(ticket, now):
    total_h = SLA_HOURS.get(ticket["urgency"], 24)
    remaining_h = (ticket["deadline"] - now).total_seconds() / 3600
    pct_left = remaining_h / total_h
    if remaining_h <= 0:
        label = "🔴 Breached"
    elif pct_left <= 0.25:
        label = "🟠 At Risk"
    elif pct_left <= 0.50:
        label = "🟡 Watch Closely"
    else:
        label = "🟢 On Track"
    return label, remaining_h

def format_remaining(hours):
    if hours <= 0:
        return f"Overdue by {abs(hours):.1f} h"
    return f"{hours:.1f} h left"

st.title("🎫 TicketPilot")
st.caption("Support Ticket Classifier, Auto-Router & SLA Breach Predictor")

# Sidebar: simulate time passing so you can demo breaches instantly
st.sidebar.header("⏱️ SLA Demo Control")
sim_hours = st.sidebar.slider("Simulate hours passed", 0, 80, 0)
st.sidebar.caption("Move the slider to see tickets move from On Track to Breached.")
st.sidebar.markdown("**SLA targets**\n- High: 4 h\n- Medium: 24 h\n- Low: 72 h")
now = datetime.now() + timedelta(hours=sim_hours)

if "ticket_log" not in st.session_state:
    st.session_state.ticket_log = []

st.write("Paste a customer ticket below to classify it and route it automatically.")
ticket_text = st.text_area(
    "Enter ticket text:",
    height=120,
    placeholder="e.g., I was charged twice for my subscription this month",
)

if st.button("Classify Ticket"):
    if ticket_text.strip() == "":
        st.warning("Please enter some ticket text.")
    else:
        text_vec = vectorizer.transform([ticket_text])
        category = category_model.predict(text_vec)[0]
        confidence = category_model.predict_proba(text_vec).max()
        urgency = urgency_model.predict(text_vec)[0]
        routed_team = routing_map.get(category, "General Support")

        submitted = datetime.now()
        deadline = submitted + timedelta(hours=SLA_HOURS.get(urgency, 24))

        st.success("Ticket classified successfully!")
        c1, c2, c3 = st.columns(3)
        c1.metric("Category", category)
        c2.metric("Urgency", urgency)
        c3.metric("Confidence", f"{confidence:.0%}")
        st.info(f"📤 Routed to: **{routed_team}**")
        st.info(f"⏰ SLA deadline: **{deadline.strftime('%Y-%m-%d %H:%M')}** "
                f"({SLA_HOURS.get(urgency, 24)} hours)")

        if confidence < 0.5:
            st.warning("Low confidence. Consider manual review.")

        st.session_state.ticket_log.append({
            "submitted": submitted,
            "deadline": deadline,
            "ticket_text": ticket_text,
            "category": category,
            "urgency": urgency,
            "routed_to": routed_team,
        })

if st.session_state.ticket_log:
    # SLA Risk Monitor: only at-risk or breached tickets
    st.subheader("🚨 SLA Risk Monitor")
    alerts = 0
    for t in st.session_state.ticket_log:
        label, rem = sla_status(t, now)
        if "Breached" in label:
            st.error(f"{label} | {t['category']} → {t['routed_to']} | "
                     f"{format_remaining(rem)} | \"{t['ticket_text'][:60]}\"")
            alerts += 1
        elif "At Risk" in label:
            st.warning(f"{label} | {t['category']} → {t['routed_to']} | "
                       f"{format_remaining(rem)} | \"{t['ticket_text'][:60]}\"")
            alerts += 1
    if alerts == 0:
        st.success("No tickets at risk right now. 🎉")

    # Full ticket table with live SLA status
    st.subheader("📋 All Tickets")
    rows = []
    for t in st.session_state.ticket_log:
        label, rem = sla_status(t, now)
        rows.append({
            "submitted": t["submitted"].strftime("%Y-%m-%d %H:%M"),
            "ticket_text": t["ticket_text"],
            "category": t["category"],
            "urgency": t["urgency"],
            "routed_to": t["routed_to"],
            "sla_deadline": t["deadline"].strftime("%Y-%m-%d %H:%M"),
            "time_left": format_remaining(rem),
            "sla_status": label,
        })
    st.dataframe(pd.DataFrame(rows))