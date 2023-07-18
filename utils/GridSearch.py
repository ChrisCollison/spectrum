import joblib
from typing import Union

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline

from .DataSet import DataSet



def grid_search(estimator, params: dict, data: DataSet, scaler=None, model_name: Union[str, None] = None, verbose=3):
    '''
    Perform a grid search on the given estimator using the given parameters and data with a pipeline if a scaler is given.
    
    Parameters:
    - estimator - `sklearn` estimator obj - to use in the grid search.
    - params - dict - the estimators hyper-parameters to use in the grid search.
    - data - Dataset - data to use in the grid search.
    - scaler - `sklean` scaler obj - The scaler to use in the grid search.  If None, no scaling is performed.
    - model_name - str - The name of the model to try to load if already created or save if not.  If None model is neither loaded or saved.  Default is None.
    - verbose - bool - The verbosity level of the grid search.
    
    Returns:
    - grid: The grid search object.
    - y_pred: The predictions of the best estimator on the test set.
    - r2: The R2 score of the best estimator on the test set.
    '''
    model = None
    did_create_model = False
    if model_name is not None:
        try:
            model = joblib.load(f"models/{model_name}.joblib")
            print(f"Loading model: {model_name}")
        except FileNotFoundError:
            print(f"Model: {model_name} not found, creating new model...")

    if model is None:
        print("Performing grid search...")
        # GridSearchCV object (wrapped in a pipeline if scaling)
        if scaler is None:
            grid = GridSearchCV(estimator, params, cv=5, n_jobs=-1, verbose=verbose)
        else:
            pipeline = Pipeline([("scaler", scaler), ("est", estimator)])
            new_params = {f'est__{k}': v for k, v in params.items()}
            grid = GridSearchCV(pipeline, new_params, cv=5, n_jobs=-1, verbose=verbose)

        # Fit the grid search object
        grid.fit(data.X_train, data.y_train[data.target_name])
        if verbose > 0:
            print(f"Best estimator score: {grid.best_score_}")
            print(f"Best params: {grid.best_params_}")
        model = grid.best_estimator_
        did_create_model = True

    # Test it's performance on the test set
    y_pred = model.predict(data.X_test)
    r2 = r2_score(data.y_test[data.target_name], y_pred) # type: ignore

    # Cast r2 to float to be sure it's not a numpy float
    r2 = float(r2)
    if verbose > 0:
        print(f"Test R2 score: {r2:.3f}")

    # Save the best estimator if desired
    if model_name is not None and did_create_model:
        file_name = f"models/{model_name}.joblib"
        joblib.dump(model, filename=file_name)
        print(f"Saved model to: {file_name}")

    return model, y_pred, r2