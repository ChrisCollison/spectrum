import pandas as pd
from pathlib import Path  # type: ignore
from rdkit import Chem
from mordred import Calculator, descriptors
from typing import Union, List, Tuple  # type: ignore
import numpy as np
import matplotlib.pyplot as plt


# Constant for the base data set column names
base_data_columns = [
    "SMILES",
    "LambdaMaxAbs",
    "LambdaMaxEm",
    "Lifetime",
    "QY",
    "LogExtCoeff",
    "AbsFWHMcm-1",
    "EmFWHMcm-1",
    "AbsFWHMnm",
    "EmFWHMnm",
    "MolarMass",
]


def trim_outliers(
    X_data: pd.DataFrame,
    y_data: pd.Series,
    trim_quantile: Union[float, None] = None,
    min_max_values: Union[Tuple[float, float], None] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Trims outliers from y_data and matched index in X_data.

    Params:
    - `X_data` - pd.DataFrame -  features
    - `y_data` - pd.Series - target values
    - `trim_quantile` - Union[float, None] - Quantile to trim from both ends of y_data. If None, min_max_values must be specified.
    - `min_max_values` - Union[Tuple[float, float], None] - Tuple of (min_value, max_value) to trim from y_data. If None, trim_quantile must be specified.

    Returns:
    - `X_trimmed` - pd.DataFrame - features with outliers removed
    - `y_trimmed` - pd.Series - target variable with outliers removed
    """
    if trim_quantile is None and min_max_values is None:
        raise ValueError("Either trim_quantile or min_max_values must be specified.")
    if trim_quantile is not None and min_max_values is not None:
        raise ValueError(
            "Only one of trim_quantile or min_max_values can be specified."
        )
    if min_max_values is not None:
        min_value, max_value = min_max_values
    else:
        min_value = y_data.quantile(trim_quantile)  # type: ignore
        max_value = y_data.quantile(1.0 - trim_quantile)  # type: ignore

    y_trimmed = y_data[(y_data >= min_value) & (y_data <= max_value)]
    X_trimmed = X_data.loc[y_trimmed.index]
    return X_trimmed, y_trimmed


def get_target_data(
    data_path: Union[str, Path],
    target: str,
    mordred_descriptors: Union[List[str], None] = None,
    frac: float = 1.0,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Used to get the data for the model. In which the "target" is the base data chromophore properties to be predicted, and the "features" are the Mordred descriptors.

    Params:
    - `filepath` : str or pathlib.Path - Path to the file containing the data if not already generated then it will be generated from the base data set using `get_data()` and saved in the `data` directory.
    - `target` : str - the target chromophore output label to be used. For example, `"LambdaMaxAbs"`.
    - `mordred_descriptors` : list of str - The Mordred descriptors to be used as features. If None, all descriptors will be used. Default is None.
    - `frac` : float  - Fraction of the base data set to be used to generate the data.  Default is 1.0.
    - 'random_state' : int - Random state for reproducibility. Default is 42.

    Returns:
    - `X` : pandas dataframe - Training data
    - `y` : pandas data series - Training labels
    """

    # Get data
    data = get_chromophore_data(data_path, frac=frac, seed=random_state)

    # Check label is one of the base data columns
    if not target in base_data_columns:
        raise ValueError(
            f"Label must be one of the base data columns: {base_data_columns}"
        )

    # Check all requested descriptors are in the data if descriptors are provided
    feature_names = None
    if mordred_descriptors is not None:
        if not all([descriptor in data.columns
                    for descriptor in mordred_descriptors]):
            raise ValueError(
                f"Not all requested descriptors are in the data. Requested descriptors: {descriptors}. Data descriptors: {data.columns}"
            )
        feature_names = descriptors
    else:
        feature_names = [col for col in data.columns if col not in base_data_columns]

    # Filter out any rows in the data that have NaN values for the label
    data = data.dropna(subset=[target], axis=0, how="any")

    # Get sample parameters
    sample_frac = frac if isinstance(frac, float) else None
    sample_n = frac if isinstance(frac, int) else None

    # Get data
    y_data = data[target].sample(
        n=sample_n, frac=sample_frac, random_state=random_state
    )

    x_data = data[feature_names].sample(
        n=sample_n, frac=sample_frac, random_state=random_state
    )

    return x_data, y_data  # type: ignore


def plot_results(y, y_hat, title: str, r2_score: float, save: bool = True):
    """
    Plots the results of a model.

    Params:
    - `y` : array-like - The actual values.
    - `y_hat` : array-like - The predicted values.
    - `title` : str - The title of the plot.
    - `save` : bool - Whether to save the plot as a png. Default is False.
    """
    # Create a figure and a set of subplots
    _, ax = plt.subplots(figsize=(5, 4))
    ax.plot(y, y_hat, ".")
    ax.plot(y, y, linestyle=":")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title(title)

    # Add the correlation of determination (R2) to the plot
    ax.text(0.8, 0.1, f"R2: {r2_score:.2f}", transform=ax.transAxes)

    if save:
        snake_case_title = title.replace(" ", "_")
        file_path = f"models/plots/{snake_case_title}.png"
        plt.savefig(file_path, dpi=300, bbox_inches="tight")
    plt.show()


def get_chromophore_data(filename: Union[str, Path], frac=1.0, seed=42) -> pd.DataFrame:
    """
    Returns a pandas dataframe of the provided filename.

    If the file does not exist, it will be generated from the base data set authored by [Joonyoung Francis Joung, Minhi Han, Minseok Jeong, Sungnam Park](https://figshare.com/articles/dataset/DB_for_chromophore/12045567/2) with the features calculated based on the SMILES representation for all available Mordred molecule descriptors. The list of descriptors can be found [here](https://mordred-descriptor.github.io/documentation/master/descriptors.html). This can take a long time to run depending on `frac` requested. The generated file will be saved in the `data` directory.

    - `filename` : str or pathlib.Path - Name of the file to be loaded. If the file does not exist, it will be generated from the base data set. The generated file will be saved in the `data` directory.

    - `frac` : float - Fraction of the base data set to be used to generate the data if the file cannot be loaded. Default is 0.8.

    - `seed` : int - Seed to be used for the random number generator in sampling the base data if the file cannot be loaded. Default is 42.
    """
    if isinstance(filename, str):
        filename = Path.cwd() / "data" / filename

    try:
        if filename.suffix == ".csv":
            data = pd.read_csv(filename)
        elif filename.suffix == ".parquet":
            data = pd.read_parquet(filename)
        else:
            raise ValueError(f"File extension {filename.suffix} not supported")
    except FileNotFoundError:
        print(f"File {filename} not found")
        print(f"Generating data from base data set...")
        data = create_data(frac, seed)
        if filename.suffix == ".csv":
            data.to_csv(filename)
        else:
            data.to_parquet(filename)
    return data


def create_data(frac=0.8, seed=42) -> pd.DataFrame:
    """
    Generates a pandas dataframe from the base data set authored by [Joonyoung Francis Joung, Minhi Han, Minseok Jeong, Sungnam Park](https://figshare.com/articles/dataset/DB_for_chromophore/12045567/2) with the features calculated based on the SMILES representation for all available Mordred molecule descriptors. The list of descriptors can be found [here](https://mordred-descriptor.github.io/documentation/master/descriptors.html). This can take a long time to run depending on `frac` requested. The generated file will be saved in the `data` directory.

    - `frac` : float - Fraction of the base data set to be used to generate the data. Default is 0.8.

    - `seed` : int - Seed to be used for the random number generator in sampling the base data. Default is 42.
    """
    data = gen_chromophore_data(frac, seed)
    features = get_mordred_values(data)
    cleaned_features, missing_rows = clean_data(features)
    combined_data = combine_data(data, cleaned_features, missing_rows)
    return combined_data


def clean_data(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Index]:
    """
    Cleans the provided dataframe as follows:
    - Removes columns with all NaN values
    - Drops rows where all the values are missing and store the index of these rows
    - Converts the dataframe to float64
    - Converts remaining NaN values to 0

    - `data` : pandas dataframe - Dataframe containing the Chromophore data with the smiles representation of a molecule as an index - e.g. as generated by `gen_chromophore_data()` and the calculated Mordred descriptors values as columns - e.g. as generated by `add_mordred_data()`
    """

    print("Cleaning data...")
    print(f"\tInitial data shape: {data.shape}")

    # Remove columns with all NaN values
    data = data.dropna(axis=1, how="all")

    # Index of rows where all values are missing
    dropped_row_idx = data.index[data.isnull().all(axis=1)]
    data = data.drop(dropped_row_idx)

    # Convert all to float64
    data = data.astype("float64")

    # Convert remaining NaN values to 0
    data = data.fillna(0)

    print(f"\tFinal data shape: {data.shape}\n")

    return data, dropped_row_idx


def get_mordred_values(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the Mordred descriptor values to the provided dataframe. The Mordred descriptors are calculated using the SMILES representation of the molecules and are not based on experimental data. The list of descriptors can be found [here](https://mordred-descriptor.github.io/documentation/master/descriptors.html). This can take a while to run depending on the size of the dataset and the number of features.

    - `data` : pandas dataframe - Dataframe containing the Chromophore data with the smiles representation of a molecule as an index - e.g. as generated by `gen_chromophore_data()`
    """

    # Convert smiles to RDKit molecules
    molecules = []
    for i, smile in enumerate(data["SMILES"]):
        canon_smile = Chem.CanonSmiles(smile)
        data.at[i, "SMILES"] = canon_smile
        mol = Chem.MolFromSmiles(canon_smile, sanitize=True)  # type: ignore
        molecules.append(mol)

    # Calculate Mordred descriptors
    mordred_calc = Calculator(descriptors, ignore_3D=True)

    # Create dataframe of Mordred calculated descriptors values
    print(
        f"Calculating {len(mordred_calc.descriptors)} Mordred descriptors for each row...\n"
    )
    mordred_data = mordred_calc.pandas(mols=molecules)

    return mordred_data


def combine_data(
    data: pd.DataFrame, features: pd.DataFrame, missing_rows: pd.Index
) -> pd.DataFrame:
    """
    Combines the provided dataframes as follows:
    - Drops rows where all the values are missing from the `data` dataframe and store the index of these rows
    - Drops the rows with the same index from the `features` dataframe
    - Concatenates the two dataframes and resets the index

    - `data` : pandas dataframe - Dataframe containing the Chromophore data with the smiles representation of a molecule as an index - e.g. as generated by `gen_chromophore_data()`
    - `features` : pandas dataframe - Dataframe containing the calculated Mordred descriptors values as columns - e.g. as generated by `add_mordred_data()`
    - `missing_rows` : pandas index - Index of rows where all values are missing from the `data` dataframe
    """
    data = data.drop(missing_rows)
    features = features.drop(missing_rows)
    return pd.concat([data, features], axis=1)


def gen_chromophore_data(frac=1.0, seed=42):
    """
    Generate a pandas dataframe of index(smiles molecules) and their features based on the data set authored by  [Joonyoung Francis Joung, Minhi Han, Minseok Jeong, Sungnam Park](https://figshare.com/articles/dataset/DB_for_chromophore/12045567/2).

    - `frac` : float, optional - Fraction of the data to return. The default is 0.8.
    - `seed` : int, optional - Random seed. The default is 42.

    Features:
    - `SMILES` : str - SMILES representation of the molecule
    - `LambdaMaxAbs` : float - Absorption maximum wavelength (nm)
    - `LambdaMaxEm` : float - Emission maximum wavelength (nm)
    - `Lifetime` : float - Fluorescence lifetime (ns)
    - `QY` : float - Quantum yield
    - `LogExtCoeff` : float - Log of extinction coefficient (M-1cm-1)
    - `AbsFWHMcm-1` : float - Absorption full width at half maximum (cm-1)
    - `EmFWHMcm-1` : float - Emission full width at half maximum (cm-1)
    - `AbsFWHMnm` : float - Absorption full width at half maximum (nm)
    - `EmFWHMnm` : float - Emission full width at half maximum (nm)
    - `MolarMass` : float - Molar mass (g/mol)
    """

    # Create path to data file
    infile = Path.cwd() / "data" / "DB_for_chromophore_Sci_Data_rev02.csv"

    # Read data
    print(f"Reading data from {infile}...\n")
    data = pd.read_csv(infile)

    # Drop unnecessary columns
    data = data.drop(["Solvent", "Reference", "Tag"], axis=1)

    # Rename columns
    data.columns = base_data_columns

    # Take a random sample of the data
    print(f"Taking a random sample of {round(frac * len(data))} rows...\n")
    data = data.sample(frac=frac, random_state=seed).reset_index(drop=True)

    return data
