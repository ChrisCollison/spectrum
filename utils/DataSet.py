import os
import urllib.request
from typing import Union, Tuple, List

import pandas as pd
from rdkit import Chem
import selfies as sf
from sklearn.model_selection import train_test_split

from .GetMolDescriptors import get_Mol_Descriptors


class DataSet:
    """
    A custom class to hold data relating to the Chromophore dataset authored by[Joonyoung Francis Joung, Minhi Han, Minseok Jeong, Sungnam Park](https://figshare.com/articles/dataset/DB_for_chromophore/12045567/2).
    """

    def __init__(
        self,
        target: str,
        fill_na: Union[str, float, None] = None,
        drop_na_selfies: bool = True,
        descriptors: str = "all",
        test_ratio: float = 0.2,
        random_state: int = 42,
        drop_features: Union[list, None] = None,
    ):
        """
        Create a ChromophoreDataSet object for the given target variable.

        Parameters:
        - target: The name of the target variable - must be one of the following:
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
        - fill_na: The strategy to fill missing values. Defaults to None. Options are:
            - None: Don't fill missing values
            - "mean": Fill missing values with the mean of the descriptor
            - "median": Fill missing values with the median of the descriptor
            - "mode": Fill missing values with the mode of the descriptor
            - "drop": Drop any rows with missing values
            - float: Fill missing values with the given value
        - drop_na_selfies: Whether to drop any rows with missing SELFIES. Defaults to True.
        - descriptors: The descriptors to use. Defaults to "all". Options are:
            - "all": Use all the RDKit descriptors
            - "discrete": Use only the discrete RDKit descriptors
            - "continuous": Use only the continuous RDKit descriptors
        - test_ratio: The fraction of the data to use for the test set. Defaults to 0.2.
        - random_state: The random state to use for the train/test split
        """
        self.target_name = target
        self.fill_na = fill_na
        self.drop_na_selfies = drop_na_selfies
        self.descriptors = descriptors
        self.test_ratio = test_ratio
        self.random_state = random_state

        self.generate_datafiles()

        self.y, self.X = self.get_data_for_target(
            target, fill_na, drop_na_selfies, descriptors
        )

        if drop_features is not None:
            self.X.drop(drop_features, axis=1, inplace=True)

        self.descriptor_names = self.X.columns.to_list()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_ratio, random_state=random_state
        )

    def drop_features(self, features: list, inplace: bool = True):
        """
        Drop the given features from the dataset.

        Parameters:
        - features: A list of features to drop
        - inplace: Whether to drop the features in place or return a new dataset. Defaults to True.
        """
        if inplace:
            self.X.drop(features, axis=1, inplace=True)
            self.X_train.drop(features, axis=1, inplace=True)
            self.X_test.drop(features, axis=1, inplace=True)
            self.descriptor_names = self.X.columns.to_list()
            return self
        else:
            return DataSet(
                self.target_name,
                self.fill_na,
                self.drop_na_selfies,
                self.descriptors,
                self.test_ratio,
                self.random_state,
                drop_features=features,
            )

    # Class constants
    target_names = [
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

    # Class methods for loading and generating the data files
    @classmethod
    def get_data_for_target(
        cls,
        target,
        replace_na_value: Union[None, float, str] = None,
        drop_na_selfies: bool = True,
        descriptors: str = "all",
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load the data for the given target variable.

        Parameters:
        - target: The name of the target variable
        - replace_na_value: The strategy to fill missing values. Defaults to None. Options are:
            - None: Don't fill missing values
            - "mean": Fill missing values with the mean of the descriptor
            - "median": Fill missing values with the median of the descriptor
            - "mode": Fill missing values with the mode of the descriptor
            - "drop": Drop any rows with missing values
            - float: Fill missing values with the given value
        - drop_na_selfies: Whether to drop any rows with missing SELFIES. Defaults to True.
        - descriptors: The descriptors to use. Defaults to "all". Options are:
            - "all": Use all the RDKit descriptors
            - "discrete": Use only the discrete RDKit descriptors
            - "continuous": Use only the continuous RDKit descriptors
        Returns:
        - y_data: The target data dataframe containing the SMILES, SELFIES, and target value
        - descriptor_data: The descriptor dataframe
        """
        # Check for valid inputs
        if target not in cls.target_names:
            raise ValueError(
                f"Invalid target variable {target}. Must be one of {cls.target_names}"
            )

        if descriptors not in ["all", "discrete", "continuous"]:
            raise ValueError(
                f"Invalid descriptors {descriptors}. Must be one of ['all', 'discrete', 'continuous']"
            )

        # Load the data
        chromophore_data = pd.read_parquet("data/Chromophore_Sci_Data.parquet")
        descriptor_data = pd.read_parquet("data/rdkit_descriptor_values.parquet")

        # Drop any rows with missing target values
        valid_target_indices = chromophore_data[target].notnull()

        # Optionally Drop any rows with missing SELFIES
        if drop_na_selfies:
            valid_target_indices = (
                valid_target_indices & chromophore_data["SELFIES"].notnull()
            )

        # Optionally limit features to discrete or continuous
        if descriptors == "discrete":
            discrete_columns = descriptor_data.select_dtypes(exclude="float").columns
            descriptor_data = descriptor_data[discrete_columns]
        elif descriptors == "continuous":
            continuous_columns = descriptor_data.select_dtypes(include="float").columns
            descriptor_data = descriptor_data[continuous_columns]

        chromophore_data = chromophore_data[valid_target_indices]
        descriptor_data = descriptor_data[valid_target_indices]
        target_data = chromophore_data[["SMILES", "SELFIES", target]]

        # Remove any features that are all NA
        descriptor_data = descriptor_data.dropna(axis=1, how="all")

        # Fill in missing values with the given value
        if replace_na_value is not None:
            if replace_na_value == "mean":
                replace_na_value = descriptor_data.mean()
            elif replace_na_value == "median":
                replace_na_value = descriptor_data.median()
            elif replace_na_value == "mode":
                replace_na_value = descriptor_data.mode()
            elif replace_na_value == "drop":
                na_rows = descriptor_data.isna().any(axis=1)
                descriptor_data = descriptor_data[~na_rows]
                target_data = target_data[~na_rows]
            else:
                replace_na_value = float(replace_na_value)
            descriptor_data = descriptor_data.fillna(replace_na_value)

        return target_data, descriptor_data

    @classmethod
    def generate_datafiles(cls):
        """Check that the data files required for the project exist in the data folder.  If they do not exist, generate them."""
        if not os.path.exists("data/Chromophore_Sci_Data.parquet"):
            print(
                "The parquet file Chromophore_Sci_Data.parquet does not exist in the data folder.\n"
                + "Generating it now..."
            )
            cls.create_parquet_chromophore_data()
        if not os.path.exists("data/rdkit_descriptor_values.parquet"):
            print(
                "The parquet file rdkit_descriptor_values.parquet does not exist in the data folder.\n"
                + "Generating it now..."
            )
            cls.generate_rdkit_desciptor_values_parquet_file()

    @classmethod
    def create_parquet_chromophore_data(cls):
        """
        Generate the parquet datafile (if not already generated) of the data set authored by  [Joonyoung Francis Joung, Minhi Han, Minseok Jeong, Sungnam Park](https://figshare.com/articles/dataset/DB_for_chromophore/12045567/2).

        The data file is saved in the data folder as `Chromophore_Sci_Data.parquet`.
        """
        if not os.path.exists("data/DB_for_chromophore_Sci_Data_rev02.csv"):
            data_url = "https://figshare.com/ndownloader/files/23637518"
            data_file_path = "data/DB_for_chromophore_Sci_Data_rev02.csv"
            try:
                urllib.request.urlretrieve(data_url, data_file_path)
                print(
                    "Downloading data file DB_for_chromophore_Sci_Data_rev02.csv from https://figshare.com/ndownloader/files/23637518 and saving it in the data folder.\n"
                )
            except:
                raise Exception(
                    "The data file DB_for_chromophore_Sci_Data_rev02.csv does not exist in the data folder.\n"
                    + " and could not be downloaded from https://ndownloader.figshare.com/files/23637518/.\n"
                    + "Please manaully download it from https://figshare.com/articles/dataset/DB_for_chromophore/12045567/2 and place it in the data folder."
                )

        if os.path.exists("data/Chromophore_Sci_Data.parquet"):
            raise FileExistsError(
                "The parquet file Chromophore_Sci_Data.parquet already exists in the data folder.\n"
                + "Please delete it if you want to generate a new one."
            )
        print(
            "Generating parquet file Chromophore_Sci_Data.parquet from DB_for_chromophore_Sci_Data_rev02.csv...\n"
        )
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

        # Sanitize and canonicalize the SMILES strings
        data["SMILES"] = data["SMILES"].apply(lambda x: Chem.MolToSmiles(Chem.MolFromSmiles(x)))  # type: ignore

        # Add the SELFIES representation of the SMILES strings
        selfies = []
        for smiles in data["SMILES"]:
            try:
                selfie = sf.encoder(smiles)
                selfies.append(selfie)
            except:
                print(
                    "The SELFIES representation of the SMILES string "
                    + smiles
                    + " could not be generated.  SELFIE set to None."
                )
                selfies.append(None)
        data["SELFIES"] = selfies

        print("Writing parquet file Chromophore_Sci_Data.parquet to data folder...\n")
        data.to_parquet("data/Chromophore_Sci_Data.parquet")

    @classmethod
    def generate_rdkit_desciptor_values_parquet_file(cls):
        """
        Generate the parquet datafile of rdkit descriptors based on the "Chromphore" smiles representations contained in the data set authored by  [Joonyoung Francis Joung, Minhi Han, Minseok Jeong, Sungnam Park](https://figshare.com/articles/dataset/DB_for_chromophore/12045567/2).

        The rdkit descriptors are generated using the `rdkit.Chem.Descriptors` module. A full list of the descriptors can be found [here](https://www.rdkit.org/docs/GettingStartedInPython.html#list-of-available-descriptors).

        The generated file is `data/rdkit_desciptor_values.parquet`.

        Note that the generation of this file can take a long time (~approx 7 mins on a 2021 MacBook Pro M1).
        """

        if not os.path.exists("data/DB_for_chromophore_Sci_Data_rev02.csv"):
            raise FileNotFoundError(
                "The data file DB_for_chromophore_Sci_Data_rev02.csv does not exist in the data folder.\n"
                + "Please download it from https://figshare.com/articles/dataset/DB_for_chromophore/12045567/2 and place it in the data folder."
            )

        if os.path.exists("data/rdkit_desciptor_values.parquet"):
            raise FileExistsError(
                "The parquet file rdkit_desciptor_values.parquet already exists in the data folder.\n"
                + "Please delete it if you want to generate a new one."
            )

        print(
            "Generating parquet file rdkit_desciptor_values.parquet from Chromophore_Sci_Data.parquet...\n"
        )

        suppl = Chem.SmilesMolSupplier("data/DB_for_chromophore_Sci_Data_rev02.csv", delimiter=",", smilesColumn=1, nameColumn=-1, titleLine=True, sanitize=True)  # type: ignore
        mols = [m for m in suppl]
        print(f"Number of molecules: {len(mols)}")
        print("Generating descriptors...")
        descriptors = [get_Mol_Descriptors(mol) for mol in mols]
        print("Writing parquet file rdkit_desciptor_values.parquet to data folder...\n")
        pd.DataFrame(descriptors).to_parquet("data/rdkit_descriptor_values.parquet")

    def save_used_features(self, file_name):
        """Save the list of features used in the dataset to a file.

        Parameters:
        - `df` - pandas.DataFrame - The DataFrame containing the features.
        - `file_name` - str - The name of the file to save the features to.
        """
        used_features = list(self.X.columns)
        with open(f"{file_name}.txt", "w") as f:
            for item in used_features:
                f.write(f"{item}\n")
            print(f"Saved used features to {file_name}.txt")
