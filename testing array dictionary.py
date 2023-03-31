def process_items_in_dict_array(data):
    for key, array in data.items():
        print(f"Key: {key}, Array: {array}")
        for index, item in enumerate(array):
            print(f"Key: {key}, Index: {index}, Item: {item}")


if __name__ == "__main__":
    example_dict = {
        "fruits": ["apple", "banana", "orange"],
        "colors": ["red", "blue", "green"],
    }

    process_items_in_dict_array(example_dict)
