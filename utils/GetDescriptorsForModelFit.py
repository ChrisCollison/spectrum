import rdkit
import pandas as pd

def generate_descriptors_for_model (df, smiles_column: str, used_features_filename: str):
    df_mols = mol_column_from_smiles_column(df, smiles_column)
    used_features = read_used_features_from_txt(used_features_filename)
    df_descriptors = df_from_selected_descriptors(df_mols, used_features)
    return df_descriptors

def mol_column_from_smiles_column (df, smiles_column_name: str):
    # Convert the SMILES strings to molecules
    df['mol'] = df[smiles_column_name].apply(rdkit.Chem.MolFromSmiles)
    # Remove any rows where the conversion failed
    df = df[df['mol'].notnull()]
    # Reset the index
    df = df.reset_index(drop=True)
    return df


def read_used_features_from_txt (file_name):
    used_features = []
    with open(f"{file_name}.txt", 'r') as f:
        for line in f:
            used_features.append(line.strip())
    return used_features

def get_selected_Mol_Descriptors(mol, descriptor_names, missing_value=None, log_missing=False):
    """For a given molecule, calculate selected descriptors from the descriptor_names list.

    Parameters:
    - `mol` - rdkit.Chem.rdchem.Mol - The molecule to calculate the descriptors for.
    - `descriptor_names` - list - List of descriptor names to be computed.
    - `missing_value` - float, optional - The value to use for descriptors that cannot be calculated.  The default is None.
    - `log_missing` - bool, optional -If True, print a message to the console when a descriptor cannot be calculated.  The default is False.

    Returns:
    - dict [str, Union[np.float32, None]] - A dictionary of the descriptors and their values.
    """
    result_dict = {}
    descriptor_dict = dict(rdkit.Chem.Descriptors._descList)
    for descriptor_name in descriptor_names:
        if descriptor_name in descriptor_dict:
            descriptor_function = descriptor_dict[descriptor_name]
            try:
                descriptor_value = descriptor_function(mol)
            except:
                descriptor_value = missing_value
                if log_missing:
                    print(f"Could not calculate {descriptor_name}")
            result_dict[descriptor_name] = descriptor_value
        else:
            print(f"Descriptor {descriptor_name} not found in available descriptors.")
    return result_dict

def df_from_selected_descriptors (df, used_features: list):
    # Compute the descriptors for each molecule and store the results in a list
    results = [get_selected_Mol_Descriptors(mol, used_features) for mol in df['mol']]
    # Convert the list of dictionaries into a DataFrame
    descriptor_df = pd.DataFrame(results)
    # combine the descriptor_df with the original dataframe
    merged_descriptor_df = pd.concat([df, descriptor_df], axis=1)
    return merged_descriptor_df