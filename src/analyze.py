import mlflow

# Connect to the experiment
experiment = mlflow.get_experiment_by_name("heart-disease-prediction")

# Search for all completed runs, sorted by F1 score
runs = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string="status = 'FINISHED'",
    order_by=["metrics.f1_score DESC"],
    output_format="pandas"
)

# Show the top 5 runs
print("Top 5 Runs by F1 Score:")
print("=" * 80)
for i, row in runs.head(5).iterrows():
    print(f"\nRun: {row['run_id'][:8]}...")
    print(f"  Model:    {row['params.model_type']}")
    print(f"  F1:       {row['metrics.f1_score']:.4f}")
    print(f"  Accuracy: {row['metrics.accuracy']:.4f}")
    print(f"  AUC-ROC:  {row['metrics.auc_roc']:.4f}")

# Find the best run
best_run = runs.iloc[0]
print(f"\n{'=' * 80}")
print(f"BEST MODEL")
print(f"{'=' * 80}")
print(f"Run ID:     {best_run['run_id']}")
print(f"Model Type: {best_run['params.model_type']}")
print(f"F1 Score:   {best_run['metrics.f1_score']:.4f}")
print(f"Accuracy:   {best_run['metrics.accuracy']:.4f}")
print(f"AUC-ROC:    {best_run['metrics.auc_roc']:.4f}")

# Bonus: show how metrics vary by model type
print(f"\n{'=' * 80}")
print("Average F1 Score by Model Type:")
print("=" * 80)
summary = runs.groupby("params.model_type")[
    "metrics.f1_score"].agg(["mean", "max", "count"])
summary.columns = ["avg_f1", "best_f1", "num_runs"]
print(summary.sort_values("best_f1", ascending=False).to_string())
