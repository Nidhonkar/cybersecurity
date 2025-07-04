import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def run_kmeans(df, features, n_clusters):
    scaler = StandardScaler()
    X = scaler.fit_transform(df[features])
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X)
    df_with_clusters = df.copy()
    df_with_clusters['Cluster'] = clusters
    return kmeans, df_with_clusters

def cluster_persona_summary(df_with_clusters, n_clusters, features):
    persona = []
    for i in range(n_clusters):
        cluster_df = df_with_clusters[df_with_clusters['Cluster'] == i]
        summary = {feat: cluster_df[feat].mean() for feat in features}
        summary['Count'] = len(cluster_df)
        persona.append({'Cluster': i, **summary})
    return pd.DataFrame(persona)
