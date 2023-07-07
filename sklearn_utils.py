from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
import joblib
from dataclasses import dataclass
import pandas as pd


@dataclass
class DataSet:
    '''
    A class to hold the data for a machine learning problem.  The data is split into training and test sets.  The training set is used to train the model, and the test set is used to evaluate the model.  The data is stored in pandas DataFrames and Series.
    '''
    target_name: str
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series

    @classmethod
    def from_data(cls, target_name: str, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42):
        '''Create a DataSet from the given data.'''
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        return cls(target_name, X_train, X_test, y_train, y_test)
    
    @classmethod
    def from_parquet(cls, target_name: str, parquet_path: str, test_size: float = 0.2, random_state: int = 42):
        '''Create a DataSet from the given parquet file.'''
        df = pd.read_parquet(parquet_path)
        X = df.drop(columns=[target_name])
        y = df[target_name]
        return cls.from_data(target_name, X, y, test_size=test_size, random_state=random_state)
    
    def save(self, path: str):
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str):
        return joblib.load(path)
    
    def __repr__(self):
        return f"DataSet(target_name={self.target_name}, X_train={self.X_train.shape}, X_test={self.X_test.shape}, y_train={self.y_train.shape}, y_test={self.y_test.shape})"
    
    @property
    def X(self):
        return pd.concat([self.X_train, self.X_test])
    
    @property
    def y(self):
        return pd.concat([self.y_train, self.y_test])
    
    @property
    def n_samples(self):
        return len(self.y)
    
    @property
    def n_features(self):
        return len(self.X.columns)
    
    @property
    def n_train(self):
        return len(self.y_train)
    
    @property
    def n_test(self):
        return len(self.y_test)
    
    @property
    def train_ratio(self):
        return self.n_train / self.n_samples
    
    @property
    def test_ratio(self):
        return self.n_test / self.n_samples
    
    @property
    def feature_names(self):
        return self.X.columns


def grid_search(estimator, params: dict, data: DataSet, scaler=None, save=True, verbose=3):
    '''
    Perform a grid search on the given estimator using the given parameters and data.  The data is split into training and test sets.  The training set is used to train the model, and the test set is used to evaluate the model.  The best estimator is saved to a file if desired.
    
    Parameters:
    - estimator - `sklearn` estimator obj - to use in the grid search.
    - params - dict - the estimators hyper-parameters to use in the grid search.
    - data - Dataset - data to use in the grid search.
    - scaler - `sklean` scaler obj - The scaler to use in the grid search.  If None, no scaling is performed.
    - save - bool -  If True, the best estimator is saved to a file.
    - verbose - bool - The verbosity level of the grid search.
    
    Returns:
    - grid: The grid search object.
    - y_pred: The predictions of the best estimator on the test set.
    - r2: The R2 score of the best estimator on the test set.
    '''

    # Create a the GridSearchCV object (wrapped in a pipeline if scaling)
    if scaler is None:
        grid = GridSearchCV(estimator, params, cv=5, n_jobs=-1, verbose=verbose)
    else:
        pipeline = Pipeline([("scaler", scaler), ("est", estimator)])
        new_params = {f'est__{k}': v for k, v in params.items()}
        grid = GridSearchCV(pipeline, new_params, cv=5, n_jobs=-1, verbose=verbose)

    # Fit the grid search object
    grid.fit(data.X_train, data.y_train)

    # Test it's performance on the test set
    y_pred = grid.best_estimator_.predict(data.X_test)
    r2 = r2_score(data.y_test, y_pred)

    # Cast r2 to float to be sure it's not a numpy float
    r2 = float(r2)

    if verbose > 0:
        print(f"Best estimator score: {grid.best_score_}")
        print(f"Best params: {grid.best_params_}")
        print(f"Test R2 score: {r2}")

    # Save the best estimator if desired
    if save:
        title = f"models/RD_{estimator.__class__.__name__}_{data.target_name}"
        para_str = "_".join([f"{k}_{v}" for k, v in grid.best_params_.items()])
        filename = f"{title}_{para_str}.joblib"
        joblib.dump(grid.best_estimator_, filename=filename)


    return grid, y_pred, r2