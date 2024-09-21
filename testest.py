def flatten_and_compare_dict_values(dict1, dict2):
    # Flatten the values of the first dictionary
    flat_values1 = sum(dict1.values(), [])

    # Flatten the values of the second dictionary
    flat_values2 = sum(dict2.values(), [])

    # Check if the lengths of the flattened lists are the same
    if len(flat_values1) != len(flat_values2):
        print(f"Different number of elements: {len(flat_values1)} vs {len(flat_values2)}")
        return

    # Compare the flattened lists
    differences = [(i, v1, v2) for i, (v1, v2) in enumerate(zip(flat_values1, flat_values2)) if v1 != v2]

    if not differences:
        print("Flattened lists are identical")
    else:
        print(f"Found {len(differences)} differences:")
        for idx, val1, val2 in differences:
            print(f"Difference at index {idx}:")
            print(f"Original value: {val1}")
            print(f"Received value: {val2}")


# Example usage:
original_dict = {
    'set1': [1, 2, 3],
    'set2': [4, 5]
}

received_dict = {
    'set1': [1, 2, 3],
    'set2': [4, 5]
}

flatten_and_compare_dict_values(original_dict, received_dict)