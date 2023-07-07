import pandas as pd
import os
from rdkit import Chem
from rdkit.Chem import Descriptors
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Union


def get_data_for_target(target, replace_na_value:Union[None, float] = None):
    '''
    Get the target values and the associated rdkit descriptor values.  The data is filtered to remove any rows where the target value is null.  The descriptor values are also filtered to remove any descriptors that are all null.  If `replace_na_value` is not `None`, then any null values in the descriptor data are replaced with the value of `replace_na_value`.

    Files Used:
    - `data/Chromophore_Sci_data.parquet` - The data file containing the target values.
    - `data/rdkit_descriptor_values.parquet` - The data file containing the rdkit descriptor values.
    
    Parameters:
    - `target` - str - The target to get the data for.  Must be one of the following:
        - `LambdaMaxAbs`- Absorption maximum wavelength (nm)
        - `LambdaMaxEm` - Emission maximum wavelength (nm)
        - `Lifetime` - Fluorescence lifetime (ns)
        - `QY` - Quantum yield
        - `LogExtCoeff` - Log of extinction coefficient (M-1cm-1)
        - `AbsFWHMcm-1` - Absorption full width at half maximum (cm-1)
        - `EmFWHMcm-1`  - Emission full width at half maximum (cm-1)
        - `AbsFWHMnm`  - Absorption full width at half maximum (nm)
        - `EmFWHMnm`  - Emission full width at half maximum (nm)
        - `MolarMass`  - Molar mass (g/mol)
    - `replace_na_value` - float, optional - The value to use for missing values.  If `None` then no replacement is made. The default is `None`.
    
    Returns:
    - `target_data` - pd.Series - The target data.
    - `descriptor_data` - pd.DataFrame - The descriptor data.'''

    _check_datafiles_exist()

    chromophore_data = pd.read_parquet("data/Chromophore_Sci_Data.parquet")
    descriptor_data = pd.read_parquet("data/rdkit_descriptor_values.parquet")

    valid_target_indices = chromophore_data[target].notnull()
    chromophore_data = chromophore_data[valid_target_indices]
    descriptor_data = descriptor_data[valid_target_indices]
    target_data = chromophore_data[target]

    # Remove any features that are all NA
    descriptor_data = descriptor_data.dropna(axis=1, how="all")

    if replace_na_value is not None:
        descriptor_data = descriptor_data.fillna(replace_na_value)

    return target_data, descriptor_data


def get_Mol_Descriptors(mol, missing_value=None, log_missing=False):
    '''For a given molecule, calculate all of the descriptors in the rdDescriptors module.  Full list of descriptors can be found [here](https://www.rdkit.org/docs/GettingStartedInPython.html#list-of-available-descriptors).

    Parameters:
    - `mol` - rdkit.Chem.rdchem.Mol - The molecule to calculate the descriptors for.
    - `missing_value` - float, optional - The value to use for descriptors that cannot be calculated.  The default is None.
    - `log_missing` - bool, optional -If True, print a message to the console when a descriptor cannot be calculated.  The default is False.

    Returns:
    - dict [str, Union[np.float32, None]] - A dictionary of the descriptors and their values.
    '''
    result_dict = {}
    for i, (descriptor_name, descriptor_function) in enumerate(Descriptors._descList):
        try:
            descriptor_value = descriptor_function(mol)
            descriptor_value = np.float32(descriptor_value)
        except:
            descriptor_value = missing_value
            if log_missing:
                print(f"Index {i} - could not calculate {descriptor_name}")
        result_dict[descriptor_name] = descriptor_value
    return result_dict

def plot_results(y: pd.Series, y_hat, title: str, r2_score: Union[float, np.float64], save: bool = True):
    """
    Plots the results of a model.

    Params:
    - `y` : array-like - The actual values.
    - `y_hat` : array-like - The predicted values.
    - `title` : str - The title of the plot.
    - `save` : bool - Whether to save the plot as a png. Default is False.
    """

    if len(y) != len(y_hat):
        raise ValueError("y and y_hat must be the same length")
    
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


def create_parquet_chromophore_data():
    """
    Generate the parquet datafile (if not already generated) of index(smiles molecules) and their features based on the data set authored by  [Joonyoung Francis Joung, Minhi Han, Minseok Jeong, Sungnam Park](https://figshare.com/articles/dataset/DB_for_chromophore/12045567/2). The data file is saved in the data folder as Chromophore_Sci_Data.parquet.
    """

    if not os.path.exists("data/DB_for_chromophore_Sci_Data_rev02.csv"):
        raise FileNotFoundError(
            "The data file DB_for_chromophore_Sci_Data_rev02.csv does not exist in the data folder.\n" + \
            "Please download it from https://figshare.com/articles/dataset/DB_for_chromophore/12045567/2 and place it in the data folder."
        )
    
    if os.path.exists("data/Chromophore_Sci_Data.parquet"):
        raise FileExistsError(
            "The parquet file Chromophore_Sci_Data.parquet already exists in the data folder.\n" + \
            "Please delete it if you want to generate a new one."
        )
    print("Generating parquet file Chromophore_Sci_Data.parquet from DB_for_chromophore_Sci_Data_rev02.csv...\n")
    data = pd.read_csv("data/DB_for_chromophore_Sci_Data_rev02.csv")
    data = data.drop(["Solvent", "Reference", "Tag"], axis=1)
    data.columns = [
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
    print("Writing parquet file Chromophore_Sci_Data.parquet to data folder...\n")
    data.to_parquet("data/Chromophore_Sci_Data.parquet")


def generate_rdkit_desciptor_values_parquet_file():
    '''
    Generate the parquet datafile of rdkit descriptors based on the "Chromphore" smiles representations contained in the data set authored by  [Joonyoung Francis Joung, Minhi Han, Minseok Jeong, Sungnam Park](https://figshare.com/articles/dataset/DB_for_chromophore/12045567/2).

    The rdkit descriptors are generated using the `rdkit.Chem.Descriptors` module. A full list of the descriptors can be found [here](https://www.rdkit.org/docs/GettingStartedInPython.html#list-of-available-descriptors).

    The generated file is `data/rdkit_desciptor_values.parquet`.

    Note that the generation of this file can take a long time (~approx 7 mins on a 2021 MacBook Pro M1).
    '''

    if not os.path.exists("data/DB_for_chromophore_Sci_Data_rev02.csv"):
        raise FileNotFoundError(
            "The data file DB_for_chromophore_Sci_Data_rev02.csv does not exist in the data folder.\n" + \
            "Please download it from https://figshare.com/articles/dataset/DB_for_chromophore/12045567/2 and place it in the data folder."
        )
    
    if os.path.exists("data/rdkit_desciptor_values.parquet"):
        raise FileExistsError(
            "The parquet file rdkit_desciptor_values.parquet already exists in the data folder.\n" + \
            "Please delete it if you want to generate a new one."
        )
    
    print("Generating parquet file rdkit_desciptor_values.parquet from Chromophore_Sci_Data.parquet...\n")

    suppl = Chem.SmilesMolSupplier('data/DB_for_chromophore_Sci_Data_rev02.csv', delimiter=',', smilesColumn=1, nameColumn=-1, titleLine=True, sanitize=True) #type: ignore
    mols = [m for m in suppl]
    print(f"Number of molecules: {len(mols)}")
    print("Generating descriptors...")
    descriptors = [get_Mol_Descriptors(mol) for mol in mols]
    print("Writing parquet file rdkit_desciptor_values.parquet to data folder...\n")
    pd.DataFrame(descriptors).to_parquet("data/rdkit_desciptor_values.parquet")


def _check_datafiles_exist():
    if not os.path.exists("data/Chromophore_Sci_Data.parquet"):
        raise FileNotFoundError(
            "The parquet file Chromophore_Sci_Data.parquet does not exist in the data folder.\n" + \
            "Please run create_parquet_chromophore_data() to generate it."
        )
    if not os.path.exists("data/rdkit_descriptor_values.parquet"):
        raise FileNotFoundError(
            "The parquet file rdkit_descriptor_values.parquet does not exist in the data folder.\n" + \
            "Please run generate_rdkit_desciptor_values_parquet_file() to generate it."
        )