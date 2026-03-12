

from src.features.store import load_features
import shap
import mlflow.lightgbm
import matplotlib.pyplot as plt                                                                                            

# The run id of the p50 model                                                                                                                              
run_id = "25edfa4de4044027bb61638f62fa57e3"                                                                                 
model = mlflow.lightgbm.load_model(f"runs:/{run_id}/lgbm_p50")

features_df = load_features()

features_df.drop("settlement_period", axis=1, inplace=True)
features_df.drop("carbon_intensity", axis=1, inplace=True)  
explainer = shap.TreeExplainer(model)
shap_values = explainer(features_df)

# Beeswarm plot
shap.plots.beeswarm(shap_values, show=False)
plt.savefig("beeswarm.png", bbox_inches="tight") 
plt.clf()
# Barplot
shap.plots.bar(shap_values, show=False) 
plt.savefig("bar_plot.png")
