from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score


from utils import DataSet

lec_data = DataSet("LogExtCoeff", fill_na="drop")

cols_to_drop = lec_data.X.select_dtypes(include=['int']).columns.to_list()
cols_to_drop.append('Ipc')
continous_lec_data = DataSet("LogExtCoeff", fill_na="drop", drop_features=cols_to_drop)

y_train = lec_data.y_train["LogExtCoeff"]
y_test = lec_data.y_test["LogExtCoeff"]


scaler = StandardScaler()
estimator = Ridge(alpha=0.055, random_state=42)

pipeline = Pipeline([('scaler', scaler), ('estimator', estimator)])
pipeline.fit(continous_lec_data.X_train, y_train)
print(f"R2 score descriptors only: {r2_score(y_test, pipeline.predict(continous_lec_data.X_test))}")