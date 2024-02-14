def encode_animals(animals):
    encoded_values = []
    label_map = {}
    current_label = 0
    for animal in animals:
        if animal not in label_map:
            label_map[animal] = current_label
            current_label += 1
        encoded_values.append(label_map[animal])
    return encoded_values

animals = ['dog', 'cat', 'bird', 'dog', 'bird', 'cat','elephant','tiger','fox','tiger']
encoded_values = encode_animals(animals)
print(encoded_values)
