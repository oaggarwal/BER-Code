# BER-Code

Before we get started with the description, I want to note that currently the code is just working off of 1 CSV file that contained 2 repetitions of the 128 bit PAM4 PRBS7 signal and the first 128 bit pattern is used as a "refrence" to calculate errors from this "refrence" to the next 128 bit pattern which is called "recived". The plan is to somehow import the waveform data directly from the scope into the python code but I need to figure that out...

1. Load Oscilloscope Data
Function: load_csv(file_path)
Reads a CSV file containing two waveform channels (CH1, CH2) and a time increment field.
Extracts and converts CH1/CH2 values into floating-point arrays.
Retrieves the time_increment for use in sampling rate calculations.

2. Clock Configuration
Functions:
calculate_clock_period(clock_frequency) — Inverts the clock frequency to get the period of one symbol.
calculate_samples_per_symbol(clock_period, time_increment) — Divides the symbol period by the time increment to determine how many samples represent one symbol.

4. Differential Signal Conversion
Function: compute_differential_signal(ch1, ch2)
Computes the difference between CH1 and CH2 to create a differential waveform

4. Symbol Block Segmentation
Function: divide_into_blocks(voltages, samples_per_symbol)
Splits the signal into fixed-length blocks, one per symbol.

5. Middle Value Extraction
Function: extract_middle_values(blocks)
Extracts the center value of each symbol block. This approximates the sampling point in a real receiver and reduces sensitivity to edge noise.

6. Symbol Decoding
Function: map_to_symbols(middle_values)
Uses absolute values of the middle voltage to classify each sample into one of four PAM4 levels.
Mapping thresholds (calculated through making histogram of voltage values in CSV file and anayzing to pick best thresholds per symbol):

00: 0.0000 – 0.0054 V

10: 0.0054 – 0.0128 V

01: 0.0128 – 0.0162 V

11: 0.0162 – 0.0252 V

Out-of-range samples are flagged and skipped with a warning.
Debug output shows each voltage and its mapped symbol.

7. Reference Sequence Extraction
Function: stop_at_128_symbols(symbol_sequence)
Takes the first 128 decoded symbols as a reference pattern (assumed to be correct).

8. Bit Error Rate (BER) Calculation
Function: calculate_ber(received_sequence, reference_sequence)
Compares subsequent 128-symbol chunks to the initial reference.
Counts the number of symbol mismatches.
BER is computed as:

python
Copy
Edit
BER = errors / ((len(reference) + len(received)) * 2)

Multiplied by 2 because PAM4 symbols represent 2 bits each.
Both reference and received lengths are used as the denominator because they’re treated as co-equal in this measurement and are both being sent to the reciver.

9. Main Driver Function
Function: process_signal(file_path, clock_frequency)
Integrates all prior steps:
Loads data, calculates sampling structure.
Converts to differential waveform and segments into blocks.
Extracts middle values, decodes them into symbols.
Uses the first 128 symbols as the reference.
Iteratively computes BER across all remaining blocks (in chunks of 128 symbols).

10. Execution and Output
The if __name__ == "__main__" block runs the full process on a CSV file at a specified clock frequency (10 MHz in this case).
It prints the reference symbol sequence and the BER results for each subsequent block.


