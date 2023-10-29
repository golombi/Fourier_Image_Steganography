import numpy as np


def get_nth_bit_float64(float_value, n):
    # Check if the bit position is valid (0 to 51 for a 64-bit float)
    if n < 0 or n >= 52:
        raise ValueError("Invalid bit position; must be between 0 and 51")

    # Convert the float64 to an int64 without changing the bit pattern
    int_bits = float_value.view(np.int64)

    # Extract the nth bit value using bitwise AND with a mask
    bit_value = (int_bits >> n) & 1

    return bit_value

def set_nth_bit_float64(float_value, n, bit_value):
    # Check if the bit position is valid (0 to 51 for a 64-bit float)
    if n < 0 or n >= 52:
        raise ValueError("Invalid bit position; must be between 0 and 51")

    # Convert the float64 to an int64 without changing the bit pattern
    int_bits = float_value.view(np.int64)

    # Create a mask with only the nth bit set to 1
    mask = np.int64(1 << n)

    if bit_value == 0:
        # To set the nth bit to 0, invert the mask and perform bitwise AND
        mask = ~mask
        int_result = int_bits & mask
    elif bit_value == 1:
        # To set the nth bit to 1, perform bitwise OR with the mask
        int_result = int_bits | mask
    else:
        raise ValueError("Invalid bit_value; it should be 0 or 1")

    # Convert the result back to a float64 using view
    float_result = int_result.view(np.float64)

    return float_result


def get_nth_bit_complex128(complex_value, n, b_real):
    if b_real:
        return get_nth_bit_float64(complex_value.real, n)
    else:
        return get_nth_bit_float64(complex_value.imag, n)

def set_nth_bit_complex128(complex_value, n, bit_value, b_real):
    if b_real:
        return complex(set_nth_bit_float64(complex_value.real, n, bit_value), complex_value.imag)
    else:
        return complex(complex_value.real, set_nth_bit_float64(complex_value.imag, n, bit_value))


if __name__ == "__main__":
    # Example usage:
    original_value = np.float64(50000)
    bit_position_to_set = 10
    desired_bit_value = 1  # Set the nth bit to 1
    result = set_nth_bit_float64(original_value, bit_position_to_set, desired_bit_value)

    print(bin(original_value.view(np.int64)))
    print(bin(result.view(np.int64)))
    print(get_nth_bit_float64(original_value, 10))
    print(get_nth_bit_float64(result, 10))
