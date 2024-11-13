import numpy as np
import mne
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import os
import json

edf_file = "C:\\project\\signal\\DS-blink-control\\S4-B4-O2_INSIGHT2_49865_2023.11.01T18.10.57+07.00.edf"

channel_names = ['AF3', 'AF4']
start_time = 5
end_time = 10

raw1b = mne.io.read_raw_edf(edf_file, preload=True)
raw1b.pick_channels(channel_names)
raw2i = raw1b.copy()

raw1b.filter(1, 4, fir_design='firwin')  # For blink detection
raw2i.filter(8, 30, fir_design='firwin')  # For beta power analysis

eeg_data = raw1b.get_data(picks=channel_names)
combined_signal = np.mean(eeg_data, axis=0)
scaled_signal = combined_signal * 1000  # Scale to microvolts
sfreq = raw1b.info['sfreq']

start_sample = int(start_time * sfreq)
end_sample = int(end_time * sfreq)
time_range_signal = scaled_signal[start_sample:end_sample]
time_range_times = raw1b.times[start_sample:end_sample]

max_amplitude = np.max(np.abs(time_range_signal))
threshold = max_amplitude * 0.5  # Increase the threshold


peaks, properties = find_peaks(time_range_signal, height=threshold, distance=int(sfreq * 0.38),width=int(sfreq * 0.1), prominence=max_amplitude * 0.3)  # Increase prominence
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
elif num_blinks > 4:
    print("Detected more than 4 blinks, no device selected.")
    selected_device = None
else:
    print("not select")
    selected_device = None

plt.plot(time_range_times, time_range_signal, label='EEG Signal')
plt.plot(time_range_times[peaks], time_range_signal[peaks], 'rx', label='Detected Blinks')
plt.axhline(threshold, color='g', linestyle='--', label='Threshold')
plt.title('Blink Detection in EEG Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()
plt.show()


no_imagination_start = 0
no_imagination_end = 5
imagination_start = 10
imagination_end = 15

no_imagination_start_sample = int(no_imagination_start * sfreq)
no_imagination_end_sample = int(no_imagination_end * sfreq)
imagination_start_sample = int(imagination_start * sfreq)
imagination_end_sample = int(imagination_end  * sfreq)


beta_band_signal = raw2i.copy().filter(8, 30, fir_design='firwin')

no_imagination_signal_beta = beta_band_signal.get_data(picks=channel_names)[:, no_imagination_start_sample:no_imagination_end_sample]
imagination_signal_beta = beta_band_signal.get_data(picks=channel_names)[:, imagination_start_sample:imagination_end_sample]

min_samples = min(no_imagination_signal_beta.shape[1], imagination_signal_beta.shape[1])
no_imagination_signal = no_imagination_signal_beta[:, :min_samples]
imagination_signal = imagination_signal_beta[:, :min_samples]

fft_freqs = np.fft.rfftfreq(min_samples, 1 / sfreq)

# Proceed with the FFT and power calculation as in your current code
no_imagination_fft_beta = np.fft.rfft(no_imagination_signal_beta, axis=1)
imagination_fft_beta = np.fft.rfft(imagination_signal_beta, axis=1)

# PSD calculation for the beta band
no_imagination_psd_beta = np.abs(no_imagination_fft_beta) ** 2 / min_samples
imagination_psd_beta = np.abs(imagination_fft_beta) ** 2 / min_samples

# The rest of the beta power calculations and state file toggling remain the same.


no_imagination_beta_power = np.sum(no_imagination_psd_beta, axis=1) * 10e+6
imagination_beta_power = np.sum(imagination_psd_beta, axis=1) * 10e+6

# Plotting the comparison of beta power between no imagination and imagination states
plt.figure(figsize=(10, 6))

# Bar plot to show beta power in both conditions
bar_width = 0.35
index = np.arange(len(channel_names))

# Plot the beta power for no imagination and imagination states
plt.bar(index, no_imagination_beta_power, bar_width, label='No Imagination Beta Power', color='blue')
plt.bar(index + bar_width, imagination_beta_power, bar_width, label='Imagination Beta Power', color='green')

# Add labels and title
plt.xlabel('Channels')
plt.ylabel('Beta Power')
plt.title('Comparison of Beta Power in No Imagination and Imagination States')

# Add ticks for channel names
plt.xticks(index + bar_width / 2, channel_names)

# Add legend
plt.legend()

# Show the plot
plt.tight_layout()
plt.show()

# Plotting the beta power for no imagination and imagination states
for i, channel in enumerate(channel_names):
    print(f"{channel} Channel:")
    print(f"  No Imagination Beta Power: {no_imagination_beta_power[i]:.4f}")
    print(f"  Imagination Beta Power: {imagination_beta_power[i]:.4f}")
    print(f"  Increase: {imagination_beta_power[i] > no_imagination_beta_power[i]}")

state_file = "device_states.json"

if os.path.exists(state_file):
    with open(state_file,"r") as file:
        device_states = json.load(file)
else:
    device_state = ["open","open","open","open"]

def toggle_state(current_state):
    return "closed" if current_state == "open" else "open"

if selected_device:
    state = device_states[selected_device - 1]

    if all(imagination_beta_power[i] > no_imagination_beta_power[i] for i in range(len(imagination_beta_power))):
        state = toggle_state(state)
        device_states[selected_device - 1] = state

    with open(state_file, "w") as file:
        json.dump(device_states, file)

    print(f"Device {selected_device} state: {state}")
