# 🎫 TicketPilot: Support Ticket Classifier, Auto-Router & SLA Breach Predictor

Customer support teams spend a lot of time reading incoming tickets and deciding who should handle them. When triage is manual, replies get slower and urgent tickets can sit unnoticed until they miss their resolution deadline.

TicketPilot reads the text of a support ticket, predicts its **category** and **urgency**, routes it to the right team, and tracks **SLA deadlines** in real time so at-risk tickets are flagged before they breach.

## Features
- **Ticket classification:** Billing, Technical, Account or General Inquiry
- **Urgency prediction:** High, Medium or Low
- **Auto-routing:** assigns each ticket to the right team, with a confidence score
- **SLA Breach Risk Monitor:** deadlines by urgency (High 4h, Medium 24h, Low 72h) with status: On Track, Watch Closely, At Risk or Breached
- **Alert panel** showing only tickets that need attention
- **Demo slider** to simulate time passing and see tickets move toward breach

## How it works
1. Ticket text is converted to features using TF-IDF.
2. Two Logistic Regression models predict category and urgency.
3. A routing layer maps each category to a team.
4. An SLA engine calculates deadlines and risk status.
5. A Streamlit interface ties it together.

## Results
| Task | Accuracy (test set) |
|------|---------------------|
| Category classification | 100% |
| Urgency classification | 70% |

> **Note:** The model was trained on a synthetic dataset I generated. The category score validates the pipeline but does not represent real-world performance. Urgency labels include deliberate randomness, so 70% is realistic for that task.

## Tech stack
Python, scikit-learn, pandas, TF-IDF, Logistic Regression, Streamlit, Git/GitHub

## How to run locally
```bash
pip install -r requirements.txt
python generate_data.py
python train_model.py
streamlit run app.py
```

## Live demo
[Add your streamlit.app link here]

## Future improvements
- Train and test on a real-world ticket dataset
- Replace TF-IDF with a fine-tuned transformer such as BERT
- Predict SLA breach risk using ticket workload and agent availability
- Integrate with Zendesk or Freshdesk through their APIs