from config import *


def detect_hamming_error(arr, nr):
    n = len(arr)
    res = -1

    # Calculate parity bits again
    for i in range(nr):
        val = 0
        for j in range(1, n + 1):
            if (j & (2 ** i) == (2 ** i)):
                val = val ^ int(arr[-1 * j])

            # Create a binary no by appending
        # parity bits together.

        res = res + val * (10 ** i)

    # Convert binary to decimal
    return int(str(res), 2)

def detect_parity_error(data):
    parity = int(data[0])
    par_data = data[1::]
    int_data = list(map(int, par_data))
    count = sum(int_data)
    par = count % 2
    if par != parity:
        # return None
        print("Parity error")
        return par_data
    else:
        return par_data


if __name__ == "__main__":
    a = '0101010100101101'
    unpar_data = detect_parity_error(a)
    if unpar_data:
        error = detect_hamming_error(unpar_data, 4)
        if error == -1:
            print("Recv:", unpar_data)
        else:
            unpar_data = list(unpar_data)
            unpar_data[error] = '1' if unpar_data[error] == '0' else '0'
            print("Recv:", ''.join(unpar_data))

    else:
        print("Parity Error!")

