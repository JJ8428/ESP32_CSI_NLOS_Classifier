def k_fold_cv_assist_TVT(key_array, k, n):

    if n > k:
        print('Error: n should be less than k')
        return

    part_size = len(key_array) // k
    
    index1 = n * part_size
    index2 = (n + 1) * part_size
    test_key_array = key_array[index1: index2]
    
    val_key_array = []
    index = index2
    while True:
        if index > len(key_array) - 1:
            index = 0
        val_key_array.append(key_array[index])
        index += 1
        part_size -= 1
        if part_size == 0:
            break

    train_key_array = [key for key in key_array if key not in test_key_array and key not in val_key_array]

    return train_key_array, val_key_array, test_key_array


def k_fold_cv_assist(key_array, k, n):

    if n > k:
        print('Error: n should be less than k')
        return

    part_size = len(key_array) // k
    start_index = n * part_size
    end_index = (n + 1) * part_size if n < k - 1 else len(key_array)
    train_key_array = key_array[:start_index] + key_array[end_index:]
    test_key_array = key_array[start_index:end_index]
    
    return train_key_array, test_key_array
