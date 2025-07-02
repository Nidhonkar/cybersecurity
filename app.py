import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import Ridge, Lasso
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, roc_curve, auc
import joblib
from io import BytesIO
from mlxtend.frequent_patterns import apriori, association_rules

import utils  # local module

st.set_page_config(page_title="Cyber Awareness Dashboard", layout="wide")

# ---------------- Sidebar – Dataset ----------------
st.sidebar.header("Data Source")
default_path = "cyber_awareness_synthetic.csv"
uploaded_file = st.sidebar.file_uploader(
    "Upload CSV (optional, else defaults to cyber_awareness_synthetic.csv)",
    type=["csv"]
)
df = utils.load_data(default_path, uploaded_file)
st.sidebar.success(f"Dataset rows: {len(df):,}")

# ---------------- Tabs ----------------
tabs = st.tabs(["Data Visualisation",
                "Classification",
                "Clustering",
                "Association Rules",
                "Regression Insights"])

# ------------------------------------------------------------------
# 1. DATA VISUALISATION
# ------------------------------------------------------------------
with tabs[0]:
    st.header("Descriptive Insights")

    # Example complex insight #1 – Incidents by Department & Job Level
    st.subheader("Incident distribution by Department & Job Level")
    pivot = (df.groupby(['Department','Job_Level'])['Incidents_Raw_Count']
               .mean()
               .reset_index(name='Mean_Incidents'))
    fig1 = px.bar(pivot, x='Department', y='Mean_Incidents',
                  color='Job_Level', barmode='group',
                  title="Average Incidents per Employee")
    st.plotly_chart(fig1, use_container_width=True)

    # Insight #2 – Phishing pass rate across training recency
    st.subheader("Phishing Pass‑Rate vs. Training Recency")
    pass_rate = (df.groupby('Training_Recency')['Phishing_Pass']
                   .apply(lambda s: (s=='Yes').mean())
                   .reset_index(name='Pass_Rate'))
    fig2 = px.line(pass_rate, x='Training_Recency', y='Pass_Rate',
                   markers=True)
    st.plotly_chart(fig2, use_container_width=True)

    # Additional charts (8 more) can be toggled via selectbox
    st.subheader("More visualisations")
    chart_choice = st.selectbox(
        "Select a visual", [
            "Stress vs Incidents",
            "Self‑Rated Knowledge Distribution",
            "MFA Usage by Department",
            "Remote vs On‑Site Incident Rate",
            "Confusion between Job Level & Tenure",
            "ROC curves (quick view)",
            "Incidents Heatmap (Dept x Work Arrangement)",
            "Incident Histogram"
        ])

    if chart_choice == "Stress vs Incidents":
        fig = px.scatter(df, x="Stress_Level", y="Incidents_Raw_Count",
                         trendline="ols", color="Work_Arrangement",
                         title="Stress Level vs Incidents")
        st.plotly_chart(fig, use_container_width=True)

    elif chart_choice == "Self‑Rated Knowledge Distribution":
        fig = px.histogram(df, x="Self_Rated_Knowledge",
                           nbins=5, color="Phishing_Pass",
                           barmode="group")
        st.plotly_chart(fig, use_container_width=True)

    elif chart_choice == "MFA Usage by Department":
        crosstab = pd.crosstab(df['Department'], df['MFA_Use'],
                               normalize='index')
        fig = px.bar(crosstab, barmode='stack',
                     title="MFA Usage split by Department")
        st.plotly_chart(fig, use_container_width=True)

    elif chart_choice == "Remote vs On‑Site Incident Rate":
        rate = (df.groupby('Work_Arrangement')['Incidents_Raw_Count']
                  .mean()
                  .reset_index())
        fig = px.bar(rate, x='Work_Arrangement', y='Incidents_Raw_Count')
        st.plotly_chart(fig, use_container_width=True)

    elif chart_choice == "Confusion between Job Level & Tenure":
        crosstab = pd.crosstab(df['Job_Level'], df['Tenure'])
        fig = px.imshow(crosstab,
                        labels=dict(x="Tenure", y="Job Level",
                                    color="Count"),
                        title="Job Level vs Tenure Heatmap")
        st.plotly_chart(fig, use_container_width=True)

    elif chart_choice == "ROC curves (quick view)":
        st.write("Switch to the *Classification* tab for full ROC overlay.")

    elif chart_choice == "Incidents Heatmap (Dept x Work Arrangement)":
        pivot = pd.pivot_table(df,
                               values='Incidents_Raw_Count',
                               index='Department',
                               columns='Work_Arrangement',
                               aggfunc='mean')
        fig = px.imshow(pivot,
                        labels=dict(color="Mean Incidents"),
                        title="Incidents by Dept & Work Arrangement")
        st.plotly_chart(fig, use_container_width=True)

    elif chart_choice == "Incident Histogram":
        fig = px.histogram(df, x="Incidents_Raw_Count",
                           title="Distribution of Incident Counts")
        st.plotly_chart(fig, use_container_width=True)

    st.info("Each plot provides an at‑a‑glance story; hover for tooltips or zoom for details.")

# ------------------------------------------------------------------
# 2. CLASSIFICATION
# ------------------------------------------------------------------
with tabs[1]:
    st.header("Predict Phishing‑Simulation Outcomes")

    # Algorithm map
    algo_map = {
        'K‑Nearest Neighbours': KNeighborsClassifier(n_neighbors=5),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(
            n_estimators=200, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42)
    }
    algo_names = list(algo_map.keys())
    selected_algo = st.selectbox("Choose algorithm", algo_names)

    if st.button("Train Models"):
        X, y = utils.preprocess_for_classification(df)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y)

        metrics_table = []
        roc_fig = go.Figure()
        for name, model in algo_map.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            metrics = utils.compute_clf_metrics(y_test, y_pred)
            metrics_table.append(
                dict(Algorithm=name, **{k: round(v, 3)
                      for k, v in metrics.items()}))

            # ROC
            if hasattr(model, "predict_proba"):
                y_score = model.predict_proba(X_test)[:, 1]
            else:
                y_score = model.decision_function(X_test)
            fpr, tpr, _ = roc_curve(y_test, y_score)
            roc_auc = auc(fpr, tpr)
            roc_fig.add_trace(go.Scatter(x=fpr, y=tpr,
                                         name=f"{name} (AUC={roc_auc:.2f})",
                                         mode='lines'))

        st.subheader("Performance Table")
        st.dataframe(pd.DataFrame(metrics_table).set_index('Algorithm'))

        st.subheader("ROC Curves (all algorithms)")
        roc_fig.update_layout(xaxis_title="False Positive Rate",
                              yaxis_title="True Positive Rate",
                              yaxis_range=[0, 1],
                              xaxis_range=[0, 1],
                              template='plotly_white')
        st.plotly_chart(roc_fig, use_container_width=True)

        # Store best model for inference
        best_model_name = max(metrics_table,
                              key=lambda d: d['f1'])['Algorithm']
        best_model = algo_map[best_model_name]
        st.success(f"Best F1: {best_model_name}")
        joblib.dump(best_model, "best_clf.pkl")

    # ---- Confusion Matrix Toggle ----
    if os.path.exists("best_clf.pkl"):
        clf = joblib.load("best_clf.pkl")
        X, y = utils.preprocess_for_classification(df)
        _, X_test, _, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y)
        if st.checkbox("Show confusion matrix for best model"):
            y_pred = clf.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)
            cm_fig = px.imshow(cm,
                               text_auto=True,
                               x=['Pred No', 'Pred Yes'],
                               y=['True No', 'True Yes'],
                               aspect='auto',
                               title="Confusion Matrix")
            st.plotly_chart(cm_fig, use_container_width=False)

    # ---- Upload new data for prediction ----
    st.subheader("Upload data for prediction")
    new_data_file = st.file_uploader(
        "Upload CSV without target variable", key='pred')
    if new_data_file and os.path.exists("best_clf.pkl"):
        new_df = pd.read_csv(new_data_file)
        new_X = pd.get_dummies(new_df, columns=utils.CATEGORICAL_COLS,
                               drop_first=True)
        # Align columns
        model = joblib.load("best_clf.pkl")
        full_X, _ = utils.preprocess_for_classification(df)
        new_X = new_X.reindex(columns=full_X.columns, fill_value=0)
        predictions = model.predict(new_X)
        result_df = new_df.copy()
        result_df['Predicted_Phish_Pass'] = np.where(predictions==1,'Yes','No')

        # Download button
        buffer = BytesIO()
        result_df.to_csv(buffer, index=False)
        buffer.seek(0)
        st.download_button("Download predictions CSV",
                           data=buffer,
                           file_name="predictions.csv",
                           mime="text/csv")
        st.success("Predictions ready!")

# ------------------------------------------------------------------
# 3. CLUSTERING
# ------------------------------------------------------------------
with tabs[2]:
    st.header("Employee Segmentation (K‑Means)")

    numeric_cols = ['Self_Rated_Knowledge','Phishing_Score',
                    'Security_Culture_Perception','Stress_Level',
                    'Time_Pressure','IT_Support_Confidence',
                    'Incidents_Raw_Count']
    num_df = df[numeric_cols].copy()
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaled = scaler.fit_transform(num_df)

    # ----- Elbow -----
    if 'elbow_sse' not in st.session_state:
        sse = []
        for k in range(2, 11):
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(scaled)
            sse.append(kmeans.inertia_)
        st.session_state.elbow_sse = sse

    elbow_fig = go.Figure(data=go.Scatter(
        x=list(range(2, 11)),
        y=st.session_state.elbow_sse,
        mode='lines+markers'))
    elbow_fig.update_layout(title="Elbow Method for Optimal k",
                            xaxis_title="k", yaxis_title="SSE")
    st.plotly_chart(elbow_fig, use_container_width=False)

    # ----- Slider for k -----
    k_choice = st.slider("Select number of clusters", 2, 10, 3, 1)
    kmeans = KMeans(n_clusters=k_choice, random_state=42)
    clusters = kmeans.fit_predict(scaled)
    df['Cluster'] = clusters

    # Cluster persona table
    persona = (df.groupby('Cluster')[numeric_cols]
                 .mean()
                 .round(2))
    st.subheader("Cluster Persona (mean values)")
    st.dataframe(persona)

    # Download dataset with cluster labels
    buffer = BytesIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)
    st.download_button("Download data with clusters",
                       data=buffer,
                       file_name="clustered_data.csv",
                       mime="text/csv")
    st.info("Elbow curve helps decide k; slider lets you inspect personas.")

# ------------------------------------------------------------------
# 4. ASSOCIATION RULES
# ------------------------------------------------------------------
with tabs[3]:
    st.header("Association Rule Mining (Apriori)")

    # Select columns
    transaction_col = st.selectbox(
        "Select transaction‑style column", df.columns.tolist(),
        index=df.columns.get_loc('Security_Behavior_Tags'))

    min_support = st.slider("Min Support", 0.01, 0.5, 0.05, 0.01)
    min_conf = st.slider("Min Confidence", 0.1, 1.0, 0.3, 0.05)

    # Prepare transaction list
    transactions = df[transaction_col].fillna('').str.split(',')
    from mlxtend.preprocessing import TransactionEncoder
    te = TransactionEncoder()
    trans_array = te.fit(transactions).transform(transactions)
    trans_df = pd.DataFrame(trans_array, columns=te.columns_)

    frequent = apriori(trans_df, min_support=min_support, use_colnames=True)
    rules = (association_rules(frequent, metric="confidence",
                               min_threshold=min_conf)
             .sort_values("confidence", ascending=False)
             .head(10))
    st.dataframe(rules[['antecedents','consequents','support',
                        'confidence','lift']])

# ------------------------------------------------------------------
# 5. REGRESSION INSIGHTS
# ------------------------------------------------------------------
with tabs[4]:
    st.header("Regression Models for Phishing Score")

    X, y = utils.preprocess_for_regression(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42)

    models = {
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=0.1),
        'Decision Tree': DecisionTreeRegressor(max_depth=6, random_state=42)
    }
    perf = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        r2 = model.score(X_test, y_test)
        perf[name] = round(r2, 3)

    st.subheader("R² Scores")
    st.bar_chart(perf)

    st.write("""**Insight** – Ridge and Lasso act as baselines, while the
             Decision Tree captures non‑linearities, often yielding the
             highest explanatory power for Phishing Score.""")