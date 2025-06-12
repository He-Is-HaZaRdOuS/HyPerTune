import pandas as pd

df = pd.read_csv("features_with_cosine_similarity_local.csv")

# Remove rows with missing cosine similarities
df_clean = df.dropna(subset=["cosine_similarity_local"])

grouped = df_clean.groupby(["Operation", "Method", "Match NNZ"])

# Calculate the mean cosine similarity for each (operation, method) pair
avg_cosine_sim = grouped["cosine_similarity_local"].mean().reset_index()

# Sort
avg_cosine_sim = avg_cosine_sim.sort_values(
    by="cosine_similarity_local", ascending=False
)

avg_cosine_sim.to_csv("avg_cosine_similarity_per_method_nnz_pair.csv", index=False)

print("Done! Averages written to avg_cosine_similarity_per_method_nnz_pair.csv.")
