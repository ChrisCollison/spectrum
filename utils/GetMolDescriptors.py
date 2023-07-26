from typing import List, Union

from rdkit import Chem
from rdkit.Chem import Descriptors
import pandas as pd


def get_Mol_Descriptors(mol, descriptors: Union[None, List[str]]=None,missing_value=None, log_missing=False):
    """For a given molecule, calculate all of the descriptors in the rdDescriptors module.  Full list of descriptors can be found [here](https://www.rdkit.org/docs/GettingStartedInPython.html#list-of-available-descriptors).

    Parameters:
    - `mol` - rdkit.Chem.rdchem.Mol - The molecule to calculate the descriptors for.
    - `descriptors` - list, optional - List of descriptor names to be computed.  If None, all descriptors will be computed.  The default is None.
    - `missing_value` - float, optional - The value to use for descriptors that cannot be calculated.  The default is None.
    - `log_missing` - bool, optional -If True, print a message to the console when a descriptor cannot be calculated.  The default is False.

    Returns:
    - dict [str, Union[np.float32, None]] - A dictionary of the descriptors and their values.
    """
    result_dict = {}
    if descriptors is None:
        descriptors = [descriptor_name for descriptor_name, _ in Descriptors._descList]

    for i, descriptor_name in enumerate(descriptors):
        descriptor_function = getattr(Descriptors, descriptor_name)
        try:
            descriptor_value = descriptor_function(mol)
        except:
            descriptor_value = missing_value
            if log_missing:
                print(f"Index {i} - could not calculate {descriptor_name}")
        result_dict[descriptor_name] = descriptor_value
    return result_dict


def generate_features_for_smiles(
        smiles: List[str],
        descriptors: Union[None, List[str]] = None,
        missing_value: Union[None, float] = None,
        log_missing=True,
    ) -> pd.DataFrame:
        """
        Generate the features for the given SMILES strings.

        Parameters:
        - smiles: A list of SMILES strings
        - descriptors: A list of descriptor names to be computed.  If None, all descriptors will be computed.  The default is None.
        - missing_value: The value to use for descriptors that cannot be calculated.  The default is None.  If None, any rows with missing values will be dropped.
        - log_missing: If True, print a message to the console when a descriptor cannot be calculated.  The default is True.
        Returns:
        - A DataFrame containing the features for the given SMILES strings
        """
        mols = []
        for i, smile in enumerate(smiles):
            mol = Chem.MolFromSmiles(smile)  # type: ignore
            if mol is not None:
                mols.append(mol)
            else:
                print(f"Could not generate molecule from SMILES {i}: {smile}")

        df = pd.DataFrame(
            [
                get_Mol_Descriptors(
                    mol,
                    descriptors=descriptors,
                    missing_value=missing_value,
                    log_missing=log_missing,
                )
                for mol in mols
            ]
        )

        if missing_value is None:
            df.dropna(axis=0, inplace=True)
        return df
