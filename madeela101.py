import numpy as np
import mne
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import os

edf_file = "C:\\project\\signal\\DS-blink-control\\S16-B4-O3_INSIGHT2_49865_2023.11.06T15.27.43+07.00.edf"

channel_names = ['AF3', 'AF4']
start_time = 5 
end_time = 10  
prominence = 50e-4

raw = mne.io.read_raw_edf(edf_file, preload=True)
raw.pick_channels(channel_names)
raw.filter(0.1, 4, fir_design='firwin')

eeg_data = raw.get_data(picks=channel_names)
combined_signal = np.mean(eeg_data, axis=0)
scaled_signal = combined_signal * 1000
sfreq = raw.info['sfreq']
start_sample = int(start_time * sfreq)
end_sample = int(end_time * sfreq)
time_range_signal = scaled_signal[start_sample:end_sample]
time_range_times = raw.times[start_sample:end_sample]

max_amplitude = np.max(np.abs(time_range_signal))
threshold = max_amplitude * 0.5

peaks, properties = find_peaks(time_range_signal, height=threshold, distance=int(sfreq * 0.35), width=0.2, prominence=prominence)
num_blinks = len(peaks)
print(f"Number of detected blinks: {num_blinks}")

if num_blinks == 1:
    print("select device 1")
    selected_device = 1
elif num_blinks == 2:
    print("select device 2")
    selected_device = 2
elif num_blinks == 3:
    print("select device 3")
    selected_device = 3
elif num_blinks == 4:
    print("select device 4")
    selected_device = 4
else:
    print("not select")
    selected_device = None

plt.figure(figsize=(12, 6))
plt.plot(time_range_times, time_range_signal, label='EEG Signal')
plt.plot(time_range_times[peaks], time_range_signal[peaks], 'rx', label='Detected Blinks')
plt.axhline(y=threshold, color='g', linestyle='--', label='Threshold')
plt.title('Blink Detection in EEG Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()
plt.show()

no_imagination_start = 0
no_imagination_end = 5
imagination_start = 11
imagination_end = 15

segment_length = min(no_imagination_end - no_imagination_start, imagination_end - imagination_start)

no_imagination_start_sample = int(no_imagination_start * sfreq)
no_imagination_end_sample = int(no_imagination_end * sfreq)
imagination_start_sample = int(imagination_start * sfreq)
imagination_end_sample = int(imagination_end * sfreq)

no_imagination_signal = eeg_data[:, no_imagination_start_sample:no_imagination_end_sample]
imagination_signal = eeg_data[:, imagination_start_sample:imagination_end_sample]

# Ensure same length for FFT
min_samples = min(no_imagination_signal.shape[1], imagination_signal.shape[1])
no_imagination_signal = no_imagination_signal[:, :min_samples]
imagination_signal = imagination_signal[:, :min_samples]

no_imagination_fft = np.fft.rfft(no_imagination_signal, axis=1)
imagination_fft = np.fft.rfft(imagination_signal, axis=1)

fft_freqs = np.fft.rfftfreq(min_samples, 1 / sfreq)

no_imagination_psd = np.abs(no_imagination_fft) ** 2
imagination_psd = np.abs(imagination_fft) ** 2

beta_band = (8, 30)

# Match length of boolean mask to PSD arrays
beta_mask = (fft_freqs >= beta_band[0]) & (fft_freqs < beta_band[1])
beta_mask = beta_mask[:no_imagination_psd.shape[1]]  # Trim to match PSD length

no_imagination_beta_power = np.sum(no_imagination_psd[:, beta_mask], axis=1) * 100000
imagination_beta_power = np.sum(imagination_psd[:, beta_mask], axis=1) * 100000

for i, channel in enumerate(channel_names):
    print(f"{channel} Channel:")
    print(f"  No Imagination Beta Power: {no_imagination_beta_power[i]:.4f}")
    print(f"  Imagination Beta Power: {imagination_beta_power[i]:.4f}")
    print(f"  Increase: {imagination_beta_power[i] > no_imagination_beta_power[i]}")

state_files = ["device1_state.txt", "device2_state.txt", "device3_state.txt", "device4_state.txt"]

def toggle_state(current_state):
    return "closed" if current_state == "open" else "open"

if selected_device:
    state_file = state_files[selected_device - 1]
    if os.path.exists(state_file):
        with open(state_file, "r") as file:
            state = file.read().strip()
    else:
        state = "open"

    if all(imagination_beta_power[i] > no_imagination_beta_power[i] for i in range(len(imagination_beta_power))):
        state = toggle_state(state)

    with open(state_file, "w") as file:
        file.write(state)

    print(f"Device {selected_device} state: {state}")
