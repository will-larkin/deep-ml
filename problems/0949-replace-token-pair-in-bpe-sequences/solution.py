def replace_pair(sequences, pair, new_id):
    # Your code here

    for sequence in sequences:
        i = 0
        while i+1 < len(sequence):
            if (sequence[i], sequence[i+1]) == pair:
                sequence[i] = new_id
                del sequence[i+1]
            i += 1

    return sequences