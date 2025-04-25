import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Import CSV file and extract voltage values and time increment
def load_csv(file_path):
    data = pd.read_csv(file_path)
    ch1 = pd.to_numeric(data['CH1'], errors='coerce').values
    ch2 = pd.to_numeric(data['CH2'], errors='coerce').values
    time_increment = float(data['Increment'].iloc[0])
    return ch1, ch2, time_increment

# Step 2: User-defined clock frequency
def calculate_clock_period(clock_frequency):
    return 1 / clock_frequency

# Step 3: Calculate the number of voltage values per symbol
def calculate_samples_per_symbol(clock_period, time_increment):
    print(clock_period / time_increment)
    return int(clock_period / time_increment)

# Step 4: Compute the differential signal
def compute_differential_signal(ch1, ch2):
    return ch1 - ch2

# Step 5 (Updated): Estimate middle values using timing guess and local stability check
def estimate_middle_values(signal, samples_per_symbol, max_symbols=None, stability_window = any, variance_threshold = any):
    middle_values = []
    middle_indices = []
    current_index = 0
    count = 0
    variance_threshold = 0.006
    stability_window = 2

    while current_index + stability_window < len(signal):
        region = signal[max(0, current_index - stability_window):current_index + stability_window + 1]
        local_std = np.std(region)
        print(f"Standard Deviation {local_std}")

        if local_std < variance_threshold:
            middle_values.append(signal[current_index])
            middle_indices.append(current_index)
            count += 1
        print(f"Count {count}")
        if max_symbols and count >= max_symbols:
                break
        current_index += samples_per_symbol

    return middle_values, middle_indices

# Step 6: Map middle values to bits based on thresholds
def map_to_bits(middle_values):
    bit_sequence = ""
    for value in middle_values:
        if -0.02 < value < -0.01:
            bits = "00"
        elif -0.011 < value < 0.0:
            bits = "10"
        elif 0.0 < value < 0.0112:
            bits = "01"
        elif 0.0114 < value < 0.0252:
            bits = "11"
        else:
            print(f"Warning: Middle value {value} is out of range and will not be mapped.")
            continue
        print(f"Middle value: {value}, Mapped bits: {bits}")
        bit_sequence += bits
    return bit_sequence

# Step 7: Extract the first 127 bits for reference
def stop_at_127_bits(bit_sequence):
    return bit_sequence[:127]

# Step 8: Calculate BER bit-by-bit
def calculate_ber(received_sequence, reference_sequence):
    total_bits = ((len(received_sequence) +len(reference_sequence))*2)
    errors = sum(1 for r, ref in zip(received_sequence, reference_sequence) if r != ref)
    return errors / total_bits

# Step 9: Plotting the results
def plot_results(ch1, ch2, time_increment, middle_indices, middle_values):
    time_vector = np.arange(len(ch1)) * time_increment
    differential_signal = ch1 - ch2

    plt.figure()
    plt.plot(time_vector, differential_signal, label="Raw Differential Signal", alpha=1)
    middle_times = [idx * time_increment for idx in middle_indices]
    plt.plot(middle_times, middle_values, 'ro', markersize=4, label="Middle Values")
    plt.title("Raw Signal with Middle Values")
    plt.xlabel("Time (s)")
    plt.ylabel("Voltage (V)")
    plt.legend()
    plt.grid(True)

    plt.figure()
    plt.hist(differential_signal, bins=90, color='blue', alpha=1)
    plt.title("Histogram of Raw Differential Signal")
    plt.xlabel("Voltage (V)")
    plt.ylabel("Count")
    plt.grid(True)

    plt.figure()
    plt.hist(middle_values, bins=90, color='red', alpha=1)
    plt.title("Histogram of Middle Values")
    plt.xlabel("Voltage (V)")
    plt.ylabel("Count")
    plt.grid(True)

    plt.show()

# Main function to process the signal and calculate BER
def process_signal(file_path, clock_frequency, plot_enabled):
    ch1, ch2, time_increment = load_csv(file_path)
    clock_period = calculate_clock_period(clock_frequency)
    samples_per_symbol = calculate_samples_per_symbol(clock_period, time_increment)
    print(f"samples_per_symbol: {samples_per_symbol}")

    differential_signal = compute_differential_signal(ch1, ch2)
    middle_values, middle_indices = estimate_middle_values(
        differential_signal,
        samples_per_symbol,
        max_symbols=None,
        stability_window=2,
        variance_threshold=0.005
    )

    print(f"Middle values: {middle_values}")
    bit_sequence = map_to_bits(middle_values)
    print(f"Total bitstream length: {len(bit_sequence)} bits")

    reference_sequence = stop_at_127_bits(bit_sequence)
    print(f"Reference sequence length (bits): {len(reference_sequence)}")
    ref_start_sample = middle_indices[0] if middle_indices else 0
    ref_end_sample = middle_indices[63] if len(middle_indices) > 63 else ref_start_sample
    print(f"Reference sequence sample range: {ref_start_sample} to {ref_end_sample}")

    ber_results = []
    for i in range(127, len(bit_sequence), 128):
        received_bits = bit_sequence[i:i + 127]
        if len(received_bits) < 127:
            print(f"Skipping short bit block at bit index {i}, length: {len(received_bits)}")
            break

        start_block_index = i // 2
        end_block_index = (i + 127) // 2
        start_sample = middle_indices[start_block_index] if start_block_index < len(middle_indices) else 0
        end_sample = middle_indices[end_block_index] if end_block_index < len(middle_indices) else start_sample
        print(f"Received bit block from bit index {i} to {i + len(received_bits)}, sample range: {start_sample} to {end_sample}")
        print(f"Processing received sequence starting at index {i}:")
        print(f"Received sequence: {received_bits}")
        print(f"Length of received sequence (bits): {len(received_bits)}")
        ber = calculate_ber(received_bits, reference_sequence)
        ber_results.append(ber)

    if plot_enabled:
        plot_results(ch1, ch2, time_increment, middle_indices, middle_values)

    return reference_sequence, ber_results

# Example usage
if __name__ == "__main__":
    file_path = "/Users/omagg/Downloads/Newfile5.csv"
    clock_frequency = 10e6
    plot_enabled = True
    reference_sequence, ber_results = process_signal(file_path, clock_frequency, plot_enabled)
    print("Reference Sequence (First 127 Bits):", reference_sequence)
    print("BER Results for Subsequent Sections:", ber_results)
