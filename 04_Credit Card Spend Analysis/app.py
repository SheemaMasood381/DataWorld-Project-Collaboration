import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet

# ✅ Load the trained Prophet model **only once** using session state
if "model" not in st.session_state:
    with open("prophet_model.pkl", "rb") as f:
        st.session_state["model"] = pickle.load(f)

model = st.session_state["model"]  # Use session state model

# ✅ Load data (Keep it outside user input)
df = pd.read_csv("credit_card_transactions.csv")

# ✅ Convert datetime column & extract transaction date
df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
df["transaction_date"] = df["trans_date_trans_time"].dt.date
df["transaction_date"] = pd.to_datetime(df["transaction_date"])

# ✅ Rename column if needed
if "amount" not in df.columns and "amt" in df.columns:
    df.rename(columns={"amt": "amount"}, inplace=True)

# Clean merchant names (remove "fraud_" prefix)
df["merchant"] = df["merchant"].str.replace("fraud_", "", regex=False)

# ✅ Compute Total Spending & Transactions per Customer
customer_summary = df.groupby("cc_num").agg(
    Total_Spending=("amount", "sum"),       # Total spending per customer
    Avg_Spending=("amount", "mean"),        # Average transaction amount
    Transaction_Count=("trans_num", "count") # Total transactions
).reset_index()

# ✅ Handle missing values (if any)
customer_summary.fillna(0, inplace=True)

# ✅ Perform K-Means Clustering
from sklearn.cluster import KMeans

num_clusters = 4  # You can adjust the number of clusters based on analysis
kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
customer_summary["KMeans_Cluster"] = kmeans.fit_predict(
    customer_summary[["Total_Spending", "Avg_Spending", "Transaction_Count"]]
)

# ✅ Merge back with main dataset
df = df.merge(customer_summary, on="cc_num", how="left")


st.title(" 💳Credit Card Spend Analysis & Forecasting💳")
st.markdown("Analyze your credit card spending trends, detect anomalies, and explore insights.")

# ✅ Tabs for different analysis
# ✅ Add Customer Segmentation Tab
tab1, tab2, tab3, tab4 = st.tabs(["🌍 Geographical Analysis", "🏪 Categories & Merchants",  "🧑‍💼 Customer Segmentation","⏳ Time Analysis",])


# ==============================
# 📊 TAB 1: Spending Insights (Cities, States)
# ==============================
with tab1:
    st.subheader("📍 Spending Distribution Across Cities & States")

    # 📊 **Top 20 Cities by Spending**
    city_spending = df.groupby("city")["amount"].sum().reset_index()
    city_spending = city_spending.sort_values(by="amount", ascending=False).head(20)

    # ✅ Plotly Bar Chart for City Spending
    fig_city = px.bar(city_spending, x="amount", y="city", orientation="h", 
                      title="🏙️ Top Cities by Total Spending",
                      labels={"amount": "Total Spending ($)", "city": "City"},
                      color="amount", color_continuous_scale="viridis")
    st.plotly_chart(fig_city)

    # 🌎 **Heatmap for State-wise Spending**
    state_spending = df.groupby("state")["amount"].sum().reset_index()

    fig_state = px.choropleth(state_spending, locations="state", locationmode="USA-states", 
                              color="amount", title="🗺️ Total Spending by State",
                              color_continuous_scale="Viridis")
    fig_state.update_geos(scope="usa")

    st.plotly_chart(fig_state)  # Show the map in Streamlit


# ==============================
# 🏪 TAB 2: Top Categories & Merchants
# ==============================
import streamlit as st
import pandas as pd
import plotly.express as px  # ✅ Using Plotly for consistency

with tab2:
    st.subheader("🏪 Spending by Category & Merchant")

    # ✅ Spending by Category
    category_stats = df.groupby("category").agg(
        transaction_volume=("amount", "count"),  
        total_amount_spent=("amount", "sum")
    ).sort_values(by="total_amount_spent", ascending=False)

    st.write("### 🛒 Top Spending Categories🛍️")

    # ✅ Plotly Bar Chart for Categories
    fig_category = px.bar(category_stats, 
                          x=category_stats.index, 
                          y="total_amount_spent",
                          labels={"x": "Category", "total_amount_spent": "Total Amount Spent ($)"},
                          color="total_amount_spent", 
                          color_continuous_scale="viridis")

    st.plotly_chart(fig_category)

    # ✅ Spending by Merchant
    st.subheader("💳 Spending by Merchant")

    # ✅ Top Spending Merchants
    merchant_spend = df.groupby("merchant")["amount"].sum().reset_index()
    merchant_spend = merchant_spend.sort_values("amount", ascending=False).head(10)

    st.write("### 💰 Top Spending Merchants By Amount")

    # ✅ Plotly Bar Chart for Merchants (Total Spending)
    fig_merchant = px.bar(merchant_spend, 
                          x="amount", 
                          y="merchant", 
                          orientation="h",
                          labels={"amount": "Total Amount Spent ($)", "merchant": "Merchant"},
                          color="amount", 
                          color_continuous_scale="viridis")

    st.plotly_chart(fig_merchant)

    # ✅ Top Merchants by Transaction Volume
    merchant_transactions = df["merchant"].value_counts().reset_index().head(10)
    merchant_transactions.columns = ["merchant", "transaction_count"]

    st.write("### 🔄 Top Merchants by Transaction Volume")

    # ✅ Plotly Bar Chart for Merchants (Transaction Volume)
    fig_merchant_txn = px.bar(merchant_transactions, 
                              x="transaction_count", 
                              y="merchant", 
                              orientation="h",
                              labels={"transaction_count": "Number of Transactions", "merchant": "Merchant"},
                              color="transaction_count", 
                              color_continuous_scale="magma")

    st.plotly_chart(fig_merchant_txn)

# ==============================
# ⏳ TAB 3: 
# ==============================
with tab3:
    st.subheader("📈 Customer Segmentation using Clustering")

    # ✅ 3D Scatter Plot of Customer Clusters
    fig_cluster = px.scatter_3d(
    customer_summary,  # Use customer_summary instead of df
    x="Total_Spending",
    y="Avg_Spending",
    z="Transaction_Count",
    color=customer_summary["KMeans_Cluster"].astype(str),
    hover_data={"cc_num": True, "Total_Spending": True, "Avg_Spending": True, "Transaction_Count": True},
    title="3D Scatter Plot of Customer Clusters")

    st.plotly_chart(fig_cluster)


    # ✅ Cluster Summary Table (Fixed)
    st.write("### 🔍 Cluster Summary")
    cluster_summary = df.groupby("KMeans_Cluster").agg({
        "cc_num": "count",  # 🔥 FIX: Replace 'Customer_ID' with 'cc_num'
        "Total_Spending": "mean",
        "Avg_Spending": "mean",
        "Transaction_Count": "mean"
    }).reset_index()
    
    cluster_summary.rename(columns={"cc_num": "Customer_Count"}, inplace=True)  # ✅ Rename correctly
    st.dataframe(cluster_summary)


    # ✅ Download CSV Report
    st.write("### 📂 Download Customer Segmentation Report")
    csv = cluster_summary.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download CSV",
        data=csv,
        file_name="customer_segmentation_report.csv",
        mime="text/csv",
    )

# ==============================
# ⏳ TAB 4: Time Series Analysis (Forecasting & Anomalies)
# ==============================
with tab4:
    st.subheader("📈 Time Series Analysis & Forecasting")

    # ✅ Initialize session state variables (if not present)
    if "forecast" not in st.session_state:
        st.session_state["forecast"] = None
    if "last_days" not in st.session_state:
        st.session_state["last_days"] = None
    if "last_customer" not in st.session_state:
        st.session_state["last_customer"] = None
    if "last_category" not in st.session_state:
        st.session_state["last_category"] = None

    # ✅ User Input Panel
    st.sidebar.header("🎯 User Input Panel")

    # ✅ Select Customer ID (Dropdown)
    customer_ids = df["cc_num"].unique().tolist()
    selected_customer = st.sidebar.selectbox("Select Customer ID", customer_ids)

    # ✅ Select Forecast Period (Slider)
    days = st.sidebar.slider("Select Forecast Period (Days)", min_value=7, max_value=180, value=30)

    # ✅ Select Spending Category (Dropdown)
    categories = df["category"].unique().tolist()
    selected_category = st.sidebar.selectbox("Select Spending Category", ["All"] + categories, key="category_select")

    # ✅ Filter dataset for selected customer
    df_filtered = df[df["cc_num"] == selected_customer]

    # ✅ Filter by category (if not "All")
    if selected_category != "All":
        df_filtered = df_filtered[df_filtered["category"] == selected_category]

    # ✅ Group by transaction date and aggregate amount
    df_filtered = df_filtered.groupby("transaction_date")["amount"].sum().reset_index()
    df_filtered.columns = ["ds", "y"]  # Rename for Prophet model

    # ✅ Ensure df_filtered has correct data
    if df_filtered.empty:
        st.warning("No transaction data available for the selected filters.")
        st.stop()  # Stop execution if no data available

    # ✅ Load & Predict only when needed (fix category change issue)
    if (st.session_state["forecast"] is None 
        or st.session_state["last_days"] != days 
        or st.session_state["last_customer"] != selected_customer
        or st.session_state["last_category"] != selected_category):
        
        # 🔮 Generate Future Dates & Predict
        future = model.make_future_dataframe(periods=days)
        forecast = model.predict(future)

        # ✅ Store in session state
        st.session_state["forecast"] = forecast
        st.session_state["last_days"] = days
        st.session_state["last_customer"] = selected_customer
        st.session_state["last_category"] = selected_category

    forecast = st.session_state["forecast"]

    # ✅ Historical Spending Trend 📊
    st.subheader("📊 Historical Spending Trend")
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=df_filtered["ds"], y=df_filtered["y"], mode="lines+markers",
                              name="Historical Spending", marker=dict(color="blue")))
    fig1.update_layout(title="Past Spending Trend", xaxis_title="Date", yaxis_title="Transaction Amount")
    st.plotly_chart(fig1)

    # ✅ Display Forecast Graphs 📊
    st.subheader("🔮 Future Spending Forecast")
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat"], mode="lines", name="Predicted Spending", line=dict(color="green")))
    fig2.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat_upper"], mode="lines", name="Upper Bound", line=dict(dash="dot", color="gray")))
    fig2.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat_lower"], mode="lines", name="Lower Bound", line=dict(dash="dot", color="gray")))
    fig2.update_layout(title="Future Spending Prediction", xaxis_title="Date", yaxis_title="Transaction Amount")
    st.plotly_chart(fig2)

    # ✅ Merge Predictions for Anomaly Detection
    df_filtered = df_filtered.merge(forecast[["ds", "yhat"]], on="ds", how="left")
    df_filtered["yhat"] = df_filtered["yhat"].fillna(df_filtered["y"].median())


    # ✅ Compute residuals & detect anomalies
    df_filtered["residual"] = df_filtered["y"] - df_filtered["yhat"]
    threshold = 3 * np.std(df_filtered["residual"])
    df_filtered["anomaly"] = np.abs(df_filtered["residual"]) > threshold
    anomalies = df_filtered[df_filtered["anomaly"]]

    # ✅ Anomaly Detection Graph ⚠️
    st.subheader("⚠️ Anomaly Detection in Spending")
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=df_filtered["ds"], y=df_filtered["y"], mode="lines+markers",
                              name="Actual Spending", marker=dict(color="blue")))
    fig3.add_trace(go.Scatter(x=df_filtered["ds"], y=df_filtered["yhat"], mode="lines",
                              name="Forecasted Spending", line=dict(dash="dot", color="green")))
    fig3.add_trace(go.Scatter(x=anomalies["ds"], y=anomalies["y"], mode="markers",
                              name="Anomalies", marker=dict(color="red", size=10)))
    fig3.update_layout(title="Transaction Amount Forecast with Anomalies",
                       xaxis_title="Date", yaxis_title="Transaction Amount",
                       legend=dict(x=0, y=1), height=500)
    st.plotly_chart(fig3)


    # ✅ Display anomalies table
    if not anomalies.empty:
        st.write("### 🚨 Detected Anomalies")
        st.dataframe(anomalies)
    else:
        st.write("✅ No anomalies detected.")


# 📌 Sticky Footer (Static)
footer = """
<style>
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background-color: #f8f9fa;
        text-align: center;
        padding: 10px;
        font-size: 14px;
        color: #333;
        border-top: 1px solid #ddd;
    }
</style>
<div class="footer">
    Developed by <strong>Sheema Masood</strong> | Powered by Streamlit 🚀
</div>
"""
st.markdown(footer, unsafe_allow_html=True)

