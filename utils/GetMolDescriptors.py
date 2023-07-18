from rdkit.Chem import Descriptors


def get_Mol_Descriptors(mol, missing_value=None, log_missing=False):
    """For a given molecule, calculate all of the descriptors in the rdDescriptors module.  Full list of descriptors can be found [here](https://www.rdkit.org/docs/GettingStartedInPython.html#list-of-available-descriptors).

    Parameters:
    - `mol` - rdkit.Chem.rdchem.Mol - The molecule to calculate the descriptors for.
    - `missing_value` - float, optional - The value to use for descriptors that cannot be calculated.  The default is None.
    - `log_missing` - bool, optional -If True, print a message to the console when a descriptor cannot be calculated.  The default is False.

    Returns:
    - dict [str, Union[np.float32, None]] - A dictionary of the descriptors and their values.
    """
    result_dict = {}
    for i, (descriptor_name, descriptor_function) in enumerate(Descriptors._descList):
        try:
            descriptor_value = descriptor_function(mol)
        except:
            descriptor_value = missing_value
            if log_missing:
                print(f"Index {i} - could not calculate {descriptor_name}")
        result_dict[descriptor_name] = descriptor_value
    return result_dict