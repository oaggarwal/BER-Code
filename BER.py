import pandas as pd
import numpy as np

# Step 1: Import CSV file and extract voltage values and time increment
def load_csv(file_path):
    """
    Load the CSV file and extract voltage values, time increment, and start time.
    """
    # Load the CSV file using pandas
    data = pd.read_csv(file_path)

    # Convert CH1 and CH2 columns to numeric values
    ch1 = pd.to_numeric(data['CH1'], errors='coerce').values  # Convert to float, handle errors
    ch2 = pd.to_numeric(data['CH2'], errors='coerce').values  # Convert to float, handle errors

    # Extract the time increment (assume it's constant)
    time_increment = float(data['Increment'].iloc[0])  # Convert to float

    return ch1, ch2, time_increment

# Step 2: User-defined clock frequency
def calculate_clock_period(clock_frequency):
    """
    Calculate the clock period based on the user-defined clock frequency.
    """
    return 1 / clock_frequency

# Step 3: Calculate the number of voltage values per symbol
def calculate_samples_per_symbol(clock_period, time_increment):
    """
    Calculate the number of samples per symbol by dividing the clock period by the time increment.
    """
    return int(clock_period / time_increment)

# Step 4: Compute the differential signal
def compute_differential_signal(ch1, ch2):
    """
    Compute the differential signal as CH1 - CH2.
    """
    return ch1 - ch2

# Step 5: Divide voltage values into blocks
def divide_into_blocks(voltages, samples_per_symbol):
    """
    Divide the voltage values into blocks, where each block contains the number of samples per symbol.
    """
    num_blocks = len(voltages) // samples_per_symbol
    blocks = [voltages[i * samples_per_symbol:(i + 1) * samples_per_symbol] for i in range(num_blocks)]
    return blocks


# Step 6: Extract the middle value of each block
def extract_middle_values(blocks):
    """
    Extract the middle value of each block of data.
    """
    middle_values = [block[len(block) // 2] for block in blocks]
    return middle_values

# Step 7: Map middle values to symbols based on thresholds
def map_to_symbols(middle_values):
    symbol_sequence = []
    for value in middle_values:
        abs_val = abs(value)
        
        if 0.0000 <= abs_val < 0.0054:
            symbol_sequence.append("00")
        elif 0.0054 < abs_val < 0.0128:
            symbol_sequence.append("10")
        elif 0.0128 < abs_val < 0.0162:
            symbol_sequence.append("01")
        elif 0.0162 < abs_val < 0.0252:
            symbol_sequence.append("11")
        else:
            print(f"Warning: Middle value {value} (abs {abs_val}) is out of range and will not be mapped.")
            continue

        print(f"Middle value: {value}, Abs: {abs_val}, Mapped symbol: {symbol_sequence[-1]}")
    
    return symbol_sequence


# Step 8: Stop after 128 symbols
def stop_at_128_symbols(symbol_sequence):
    """
    Stop processing after 128 symbols.
    """
    max_symbols = 128
    return symbol_sequence[:max_symbols]

# Step 9: Compare 128-symbol sections, errors get counted up and divided by total bits for BER calculation
def calculate_ber(received_sequence, reference_sequence):
    """
    Calculate the Bit Error Rate (BER) by comparing the received symbol sequence to the reference sequence.
    """
    errors = sum(1 for received, reference in zip(received_sequence, reference_sequence) if received != reference)
    ber = errors / ((len(reference_sequence) + len(received_sequence))*2)
    return ber

# Step 10: Main function to process the signal and calculate BER
def process_signal(file_path, clock_frequency):
    """
    Main function to process the signal, map it to symbols, and calculate BER.
    """
    # Load the CSV file
    ch1, ch2, time_increment = load_csv(file_path)

    # Calculate clock period and samples per symbol
    clock_period = calculate_clock_period(clock_frequency)
    samples_per_symbol = calculate_samples_per_symbol(clock_period, time_increment)

    # Compute the differential signal
    differential_signal = compute_differential_signal(ch1, ch2)

    # Divide voltages into blocks and extract middle values
    blocks = divide_into_blocks(differential_signal, samples_per_symbol)
    # Debugging: Print the blocks
    print(f"Blocks: {blocks}")

    middle_values = extract_middle_values(blocks)
    # Debugging: Print the middle values
    print(f"Middle values: {middle_values}")

    # Map middle values to symbols
    symbol_sequence = map_to_symbols(middle_values)
    # Debugging: Check the length of the symbol sequence
    print(f"Number of blocks: {len(blocks)}, Expected symbols: {len(blocks)}, Actual symbols: {len(symbol_sequence)}")

    # Stop at 128 symbols for the first iteration (reference sequence)
    reference_sequence = stop_at_128_symbols(symbol_sequence)
    # Debugging: Print the sample range for the reference sequence
    start_sample = 0
    end_sample = 128 * samples_per_symbol - 1
    print(f"Reference sequence sample range: {start_sample} to {end_sample}")
    # Debugging: Check the length of the reference sequence
    print(f"Reference sequence length: {len(reference_sequence)}")

    # Initialize BER tracking
    ber_results = []

    # Process subsequent 128-symbol sections
    for i in range(128, len(symbol_sequence), 128):  # Step through 128-symbol blocks
        received_sequence = symbol_sequence[i:i + 128]
        if len(received_sequence) < 128:  # Stop if the block is incomplete
            print(f"Skipping block with incorrect length: {len(received_sequence)}")
            break

        # Debugging: Print the sample range for the received sequence
        start_sample = i * samples_per_symbol
        end_sample = (i + 128) * samples_per_symbol
        print(f"Received sequence sample range: {start_sample} to {end_sample}")
        
        # Debugging: Print the received sequence and its length
        print(f"Processing received sequence starting at index {i}:")
        print(f"Received sequence: {received_sequence}")
        print(f"Length of received sequence: {len(received_sequence)}")

        ber = calculate_ber(received_sequence, reference_sequence)
        ber_results.append(ber)

    return reference_sequence, ber_results

# Example usage
if __name__ == "__main__":
    # File path to the CSV file
    file_path = "/Users/omagg/Downloads/Newfile5.csv"

    # User-defined clock frequency (10 MHz)
    clock_frequency = 10e6

    # Process the signal and calculate BER
    reference_sequence, ber_results = process_signal(file_path, clock_frequency)

    # Print the reference sequence and BER results
    print("Reference Sequence (First 128 Symbols):", reference_sequence)
    print("BER Results for Subsequent Sections:", ber_results)
