
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ---------- Configuration ----------
st.set_page_config(
    page_title="ChurnIQ | Customer Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent.parent
model = joblib.load(BASE_DIR / "models" / "logistic_model.pkl")

# ---------- Styling ----------
st.markdown("""
<style>
.stApp {
    background: #f4f7fb;
    color: #17233c;
}
[data-testid="stSidebar"] {
    background: #101d35;
}
[data-testid="stSidebar"] * {
    color: #ffffff !important;
}
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1250px;
}
.hero {
    background: linear-gradient(120deg, #132544, #176b78);
    padding: 30px;
    border-radius: 18px;
    color: white;
    margin-bottom: 25px;
}
.hero h1 {
    color: white;
    font-size: 34px;
    margin-bottom: 8px;
}
.hero p {
    color: #dcecf4;
    font-size: 16px;
}
.metric-card {
    background: white;
    padding: 20px;
    border: 1px solid #e3eaf2;
    border-radius: 15px;
    box-shadow: 0 3px 12px rgba(25, 45, 75, 0.05);
    min-height: 115px;
}
.metric-label {
    color: #68778e;
    font-size: 14px;
}
.metric-value {
    color: #17233c;
    font-size: 27px;
    font-weight: 700;
    margin-top: 8px;
}
.section-card {
    background: white;
    padding: 22px;
    border: 1px solid #e3eaf2;
    border-radius: 16px;
    margin-bottom: 18px;
}
div.stButton > button[kind="primary"],
div.stFormSubmitButton > button {
    background: #087f8c;
    color: white;
    border: none;
    border-radius: 9px;
    padding: 12px 20px;
    font-weight: 600;
    width: 100%;
}
div.stButton > button:hover,
div.stFormSubmitButton > button:hover {
    background: #066672;
    color: white;
}
h2, h3 {
    color: #17233c;
}
</style>
""", unsafe_allow_html=True)

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("# 📊 ChurnIQ")
    st.caption("CUSTOMER ANALYTICS")
    st.divider()

    page = st.radio(
        "NAVIGATION",
        ["Dashboard", "Predict Churn", "About Project"]
    )

    st.divider()
    st.caption("MACHINE LEARNING")
    st.caption("Logistic Regression")
    st.caption("Model accuracy: 80.38%")
    st.caption("Dataset: Telco Customer Churn")

# ---------- Dashboard ----------
if page == "Dashboard":
    st.markdown("""
    <div class="hero">
        <h1>Customer Intelligence Dashboard</h1>
        <p>Understand customer retention and identify customers
        who may be at risk of leaving.</p>
    </div>
    """, unsafe_allow_html=True)

    data_path = BASE_DIR / "data" / "cleaned_customer_churn.csv"
    df = pd.read_csv(data_path)

    total = len(df)
    churn_count = int((df["Churn"] == "Yes").sum())
    churn_rate = churn_count / total * 100

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Customers</div>
            <div class="metric-value">{total:,}</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Customers Churned</div>
            <div class="metric-value">{churn_count:,}</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Historical Churn Rate</div>
            <div class="metric-value">{churn_rate:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    left, right = st.columns(2)

    with left:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Customer Churn Distribution")
        counts = df["Churn"].value_counts().rename(
            index={"No": "Stayed", "Yes": "Churned"}
        )
        st.bar_chart(counts)
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Churn by Contract Type")
        contract_data = pd.crosstab(
            df["Contract"], df["Churn"]
        ).rename(columns={"No": "Stayed", "Yes": "Churned"})
        st.bar_chart(contract_data)
        st.markdown('</div>', unsafe_allow_html=True)

    st.info(
        "Historical statistics describe the dataset. "
        "They are not predictions for an individual customer."
    )

# ---------- Prediction ----------
elif page == "Predict Churn":
    st.markdown("""
    <div class="hero">
        <h1>Customer Churn Prediction</h1>
        <p>Enter customer information to estimate churn risk.</p>
    </div>
    """, unsafe_allow_html=True)

    with st.form("customer_form"):
        st.subheader("01 · Customer Profile")
        a, b, c = st.columns(3)

        gender = a.selectbox("Gender", ["Female", "Male"])
        senior = b.selectbox("Senior Citizen", [0, 1])
        partner = c.selectbox("Has Partner?", ["Yes", "No"])

        a, b = st.columns(2)
        dependents = a.selectbox("Has Dependents?", ["Yes", "No"])
        tenure = b.slider("Tenure (months)", 0, 72, 12)

        st.divider()
        st.subheader("02 · Services")

        a, b, c = st.columns(3)
        phone = a.selectbox("Phone Service", ["Yes", "No"])
        multiple = b.selectbox(
            "Multiple Lines", ["No", "Yes", "No phone service"]
        )
        internet = c.selectbox(
            "Internet Service", ["DSL", "Fiber optic", "No"]
        )

        a, b, c = st.columns(3)
        security = a.selectbox(
            "Online Security", ["No", "Yes", "No internet service"]
        )
        backup = b.selectbox(
            "Online Backup", ["No", "Yes", "No internet service"]
        )
        protection = c.selectbox(
            "Device Protection", ["No", "Yes", "No internet service"]
        )

        a, b, c = st.columns(3)
        support = a.selectbox(
            "Tech Support", ["No", "Yes", "No internet service"]
        )
        tv = b.selectbox(
            "Streaming TV", ["No", "Yes", "No internet service"]
        )
        movies = c.selectbox(
            "Streaming Movies", ["No", "Yes", "No internet service"]
        )

        st.divider()
        st.subheader("03 · Billing & Contract")

        a, b = st.columns(2)
        contract = a.selectbox(
            "Contract",
            ["Month-to-month", "One year", "Two year"]
        )
        paperless = b.selectbox("Paperless Billing", ["Yes", "No"])

        a, b = st.columns(2)
        payment = a.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )
        monthly = b.number_input(
            "Monthly Charges ($)", min_value=0.0, value=70.0
        )

        total = st.number_input(
            "Total Charges ($)", min_value=0.0, value=840.0
        )

        submitted = st.form_submit_button(
            "🔍  Predict Customer Churn",
            type="primary",
            use_container_width=True
        )

    if submitted:
        customer = pd.DataFrame([{
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone,
            "MultipleLines": multiple,
            "InternetService": internet,
            "OnlineSecurity": security,
            "OnlineBackup": backup,
            "DeviceProtection": protection,
            "TechSupport": support,
            "StreamingTV": tv,
            "StreamingMovies": movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly,
            "TotalCharges": total
        }])

        prediction = int(model.predict(customer)[0])
        probability = float(model.predict_proba(customer)[0][1])

        st.divider()
        st.subheader("Prediction Result")

        a, b = st.columns(2)

        with a:
            st.metric("Estimated Churn Probability", f"{probability:.1%}")
            st.progress(probability)

        with b:
            if prediction == 1:
                st.error("⚠️ Higher churn risk predicted")
                st.write("The model predicts this customer may leave.")
            else:
                st.success("✓ Customer predicted to stay")
                st.write("The model predicts this customer may stay.")

        st.caption(
            "This is a model estimate, not a guarantee of future behavior."
        )

# ---------- About ----------
else:
    st.markdown("""
    <div class="hero">
        <h1>About ChurnIQ</h1>
        <p>A machine learning project for customer retention analysis.</p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Project Overview")
    st.write(
        "This project uses the IBM Telco Customer Churn dataset "
        "to estimate whether a customer may discontinue a service."
    )

    st.subheader("Technology Stack")
    st.markdown("""
    - **Python** — core programming language
    - **Pandas** — data cleaning and analysis
    - **Scikit-learn** — preprocessing and model training
    - **Logistic Regression** — selected prediction model
    - **Streamlit** — interactive web interface
    - **Matplotlib / Seaborn** — exploratory data visualization
    """)

    st.subheader("Model Evaluation")
    st.write("Test accuracy: 80.38%")
    st.write("Churn-class recall: 57%")
    st.write("Churn-class F1-score: 0.61")
    st.caption(
        "Metrics are from the current test split; performance may "
        "vary on new customers."
    )
