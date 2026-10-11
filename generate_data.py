import pandas as pd
import random

random.seed(42)

categories = {
    "Billing": [
        "I was charged twice for my subscription this month",
        "My invoice shows an incorrect amount",
        "Refund not received after cancellation",
        "Payment failed but the amount was deducted",
        "Need help understanding my bill charges",
        "Why was I charged a late fee on my account",
        "The discount code was not applied to my order",
        "I want to change my billing date",
        "Credit card was declined but I have enough balance",
        "Please send me a copy of my last invoice",
    ],
    "Technical": [
        "The app crashes every time I open it",
        "Unable to login, the page keeps loading forever",
        "Page not loading, showing error 500",
        "Software update broke the export feature",
        "Getting a blank screen after installation",
        "The upload button does nothing when I click it",
        "Notifications are not working on my phone",
        "The website is extremely slow today",
        "I get a timeout error when saving my work",
        "The mobile app freezes on the dashboard screen",
    ],
    "Account": [
        "I forgot my password and the reset link is not working",
        "Need to update my email address on file",
        "My account got suspended without any reason",
        "Unable to change my profile details",
        "Two-factor authentication code is not arriving",
        "How do I delete my account permanently",
        "I want to add another user to my account",
        "My username shows the wrong name",
        "Please merge my two accounts into one",
        "I cannot verify my phone number",
    ],
           "General Inquiry": [
        "What are your business hours",
        "Do you offer student discounts",
        "How can I contact customer support",
        "Where is your nearest office location",
        "What is your refund policy",
        "Do you have a referral program",
        "Is there an API available for developers",
        "Which countries do you ship to",
        "Can I get a demo of your product",
        "Do you have any upcoming webinars",
        "What time does your support team work",
        "What time does your team open on weekends",
        "When is your customer service available",
        "Is your support team open on holidays",
        "What are your working hours",
        "How do I reach someone on your team",
    ],
}

# Base urgency per category, with some randomness so it isn't a perfect lookup
urgency_weights = {
    "Billing": (["High", "Medium", "Low"], [0.6, 0.3, 0.1]),
    "Technical": (["High", "Medium", "Low"], [0.6, 0.3, 0.1]),
    "Account": (["High", "Medium", "Low"], [0.2, 0.6, 0.2]),
    "General Inquiry": (["High", "Medium", "Low"], [0.05, 0.25, 0.7]),
}

prefixes = ["", "Hi, ", "Hello, ", "Please help: ", "Urgent: ", "Hey team, "]
suffixes = ["", " Please help.", " Thanks!", " This is frustrating.", " Need this fixed asap.", " Waiting for a reply."]

rows = []
for category, samples in categories.items():
    labels, weights = urgency_weights[category]
    for _ in range(150):  # 150 per category = 600 total
        text = random.choice(prefixes) + random.choice(samples) + random.choice(suffixes)
        urgency = random.choices(labels, weights=weights)[0]
        if "Urgent" in text or "asap" in text:
            urgency = "High"
        rows.append({"ticket_text": text, "category": category, "urgency": urgency})

df = pd.DataFrame(rows)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df.to_csv("data/support_tickets.csv", index=False)
print(f"Generated {len(df)} tickets")
print(df["category"].value_counts())
print(df["urgency"].value_counts())