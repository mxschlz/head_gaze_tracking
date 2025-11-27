import mne
import pathlib
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import minmax_scale

custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", palette="colorblind", context="talk", font_scale=1.0, rc=custom_params)

# --- Import necessary functions from your existing project ---
from HeadGazeTracker import get_data_path

plt.ion()

# Define the channels and time window of interest for the Nc component
NC_CHANNELS = ['F3', 'Fz', 'F4', 'C3', 'Cz', 'C4']
NC_TIME_WINDOW = (0.4, 0.8)  # 400ms to 800ms

# --- Configuration ---
# CHOOSE YOUR METRIC: 'fuzziness' or 'head_pose_mean'
WEIGHTING_METRIC = 'head_pose_mean'

# Define which subjects to process. Copied from preprocess_eeg_with_fuzziness.py
subjects_to_include = [
    'SCS048',
    'SMM049',
    'SMM050',
    'SMM054',
    'SMS056',
]

if WEIGHTING_METRIC not in ['fuzziness', 'head_pose_mean']:
    raise ValueError("Invalid WEIGHTING_METRIC. Choose 'fuzziness' or 'head_pose_mean'.")

print(f"--- Running analysis with WEIGHTING_METRIC: '{WEIGHTING_METRIC}' ---")
# Define paths using the central get_data_path() function
base_data_path = get_data_path()
preprocessed_dir = pathlib.Path(f"{base_data_path}output/preprocessed_eeg")

# --- Main Execution ---
# Find all preprocessed epoch files
epoch_files = sorted(list(preprocessed_dir.glob('*-epo.fif')))

# Filter for the subjects we want to process
if subjects_to_include:
    epoch_files = [f for f in epoch_files if f.name.split('_')[0] in subjects_to_include]

# --- Lists to store data for Grand Average ---
all_unweighted_evokeds = []
all_weighted_evokeds = []

for file_path in epoch_files:
    # --- Start of Analysis for a single file ---
    print(f"\n--- Processing file: {file_path.name} ---")

    # --- 1. Load the preprocessed epochs ---
    try:
        epochs = mne.read_epochs(file_path, preload=True)
    except Exception as e:
        print(f"  - ERROR: Could not load epochs file. Skipping. Error: {e}")
        continue

    epochs.apply_baseline()  # Apply baseline correction

    # --- 3. Calculate the standard (unweighted) ERP ---
    unweighted_evoked = epochs.average()
    unweighted_evoked.comment = 'Unweighted Average'
    print(f"  - Calculated unweighted ERP from {len(epochs)} epochs.")

    # --- 4. Calculate the weighted ERP ---
    if WEIGHTING_METRIC == 'fuzziness':
        # --- MODIFIED: Use the raw fuzziness score and normalize it here for robustness ---
        weighting_col = 'fuzziness_score' # Use the raw fuzziness score
        if weighting_col not in epochs.metadata.columns:
            print(f"  - ERROR: '{weighting_col}' column not found. Skipping.")
            continue
        # A higher fuzziness score is worse. We normalize it to a 0-1 scale,
        # where 1 represents the highest fuzziness in the dataset.
        scores = minmax_scale(epochs.metadata[weighting_col].values)
        weights = 1.0 - scores
        weighted_comment = 'Fuzziness-Weighted Average'

    elif WEIGHTING_METRIC == 'head_pose_mean':
        weighting_col = 'head_pose_mean'
        if weighting_col not in epochs.metadata.columns:
            print(f"  - ERROR: '{weighting_col}' column not found. Skipping.")
            continue
        # For mean pose, a larger absolute deviation from zero is worse.
        # We normalize the absolute value to a 0-1 scale.
        abs_pose = abs(epochs.metadata[weighting_col].values)
        scores = minmax_scale(abs_pose)
        weights = 1.0 - scores  # Large deviation from center -> low weight
        weighted_comment = 'Head-Pose-Weighted Average'

    # Ensure weights are not negative if fuzziness > 1 for some reason
    weights[weights < 0] = 0

    # The `weights` argument for epochs.average() was added in MNE v0.21.
    # The following is a manual implementation for compatibility with older versions.
    try:
        # Try the modern, simple way first
        weighted_evoked = epochs.average(weights=weights)
    except TypeError:
        print("  - MNE version is < 0.21. Calculating weighted average manually.")
        # Get data for EEG channels only to ensure consistency with epochs.average()
        data = epochs.get_data(picks='eeg')  # Shape: (n_epochs, n_eeg_channels, n_times)

        # Create a new info object with only EEG channels
        info = mne.pick_info(epochs.info, mne.pick_types(epochs.info, eeg=True))

        # Reshape weights for broadcasting
        weights_reshaped = weights[:, np.newaxis, np.newaxis]
        # Manually compute the weighted average
        weighted_data = (data * weights_reshaped).sum(axis=0) / weights.sum()
        # Create a new Evoked object from the result
        weighted_evoked = mne.EvokedArray(weighted_data, info, tmin=epochs.tmin, nave=len(epochs))

    weighted_evoked.comment = weighted_comment
    print(f"  - Calculated fuzziness-weighted ERP.")

    # --- Append to lists for grand average ---
    all_unweighted_evokeds.append(unweighted_evoked)
    all_weighted_evokeds.append(weighted_evoked)

    # --- 5. Plot the comparison for the specific Nc channels ---
    # Find which of our desired channels are available in the data
    available_channels = [ch for ch in NC_CHANNELS if ch in epochs.ch_names]
    missing_channels = [ch for ch in NC_CHANNELS if ch not in epochs.ch_names]
    if missing_channels:
        print(f"  - Note: The following requested channels were not found and will be ignored: {missing_channels}")

    picks = available_channels
    if not picks:
        print(f"  - ERROR: None of the desired channels {NC_CHANNELS} were found. Skipping plot.")
        continue

    # Define colors for the plot
    colors = {"Unweighted": "steelblue", "Weighted": "red"}
    linestyles = {"Unweighted": "-", "Weighted": "--"}

    # Create a joint plot: ERP waveform on the left, topomap on the right
    # We use GridSpec for more control over the layout, especially for the colorbar.
    fig = plt.figure(figsize=(13, 5))
    gs = fig.add_gridspec(1, 3, width_ratios=[2.5, 2, 0.2])

    ax_erp = fig.add_subplot(gs[0, 0])
    ax_topo = fig.add_subplot(gs[0, 1])
    ax_cbar = fig.add_subplot(gs[0, 2])

    # Plot the ERP waveforms averaged over the selected channels
    mne.viz.plot_compare_evokeds(
        {'Unweighted': unweighted_evoked, 'Weighted': weighted_evoked},
        picks=picks,
        combine='mean',  # Plot the mean of the channels, not the GFP
        axes=ax_erp,
        colors=colors,
        linestyles=linestyles,
        show=False,
        legend='upper right',
        title=f'ERP over Frontocentral Channels ({", ".join(picks)})'
    )
    # Highlight the Nc time window
    ax_erp.axvspan(NC_TIME_WINDOW[0], NC_TIME_WINDOW[1], color='gray', alpha=0.2, label='Nc Window (400-800ms)')
    ax_erp.legend()

    # --- 6. Plot the topography of the difference ---
    # Calculate the difference wave (Weighted - Unweighted)
    diff_evoked = mne.combine_evoked([weighted_evoked, unweighted_evoked], weights=[1, -1])
    diff_evoked.comment = "Difference (Weighted - Unweighted)"

    # Plot the topography of the mean difference in the Nc time window
    diff_evoked.plot_topomap(
        # To plot the average over a time window, we can manually average the data
        # or specify the center of the time window and the duration.
        # Let's use the more explicit manual average method.
        # We will create a new Evoked object with the mean data in the time window.
        times=(NC_TIME_WINDOW[0] + NC_TIME_WINDOW[1]) / 2,  # Center of the window
        average=NC_TIME_WINDOW[1] - NC_TIME_WINDOW[0],  # Duration of the window
        axes=[ax_topo, ax_cbar],  # Pass both the topo and colorbar axes
        show=False,
        colorbar=True
    )

    fig.suptitle(
        f"Unweighted vs. {WEIGHTING_METRIC.replace('_', ' ').title()}-Weighted Nc Analysis\n({file_path.name})",
        fontsize=16)
    fig.tight_layout()  # Adjust layout to make room for suptitle

    # --- End of Analysis for a single file ---

# --- Grand Average Plotting ---
if all_unweighted_evokeds:
    print("\n\n--- Calculating and plotting Grand Average ERPs ---")

    # --- 1. Compute Grand Averages ---
    grand_avg_unweighted = mne.grand_average(all_unweighted_evokeds)
    grand_avg_unweighted.comment = 'Grand Average (Unweighted)'

    grand_avg_weighted = mne.grand_average(all_weighted_evokeds)
    grand_avg_weighted.comment = f"Grand Average ({WEIGHTING_METRIC.replace('_', ' ').title()}-Weighted)"

    print(f"  - Calculated Grand Average from {len(all_unweighted_evokeds)} subjects.")

    # --- 2. Plot the Grand Average comparison ---
    # Define colors and styles
    colors = {"Unweighted": "steelblue", "Weighted": "red"}
    linestyles = {"Unweighted": "-", "Weighted": "--"}

    # We can reuse the picks from the last subject, as they should be consistent
    # Or, more robustly, get them from the grand average object itself
    picks = [ch for ch in NC_CHANNELS if ch in grand_avg_unweighted.ch_names]

    # Create a joint plot
    fig = plt.figure(figsize=(13, 5))
    gs = fig.add_gridspec(1, 3, width_ratios=[2.5, 2, 0.2])
    ax_erp = fig.add_subplot(gs[0, 0])
    ax_topo = fig.add_subplot(gs[0, 1])
    ax_cbar = fig.add_subplot(gs[0, 2])

    # Plot the ERP waveforms
    mne.viz.plot_compare_evokeds(
        {'Unweighted': grand_avg_unweighted, 'Weighted': grand_avg_weighted},
        picks=picks,
        combine='mean',
        axes=ax_erp,
        colors=colors,
        linestyles=linestyles,
        show=False,
        legend='upper right',
        title=f'Grand Average ERP over Frontocentral Channels'
    )
    ax_erp.axvspan(NC_TIME_WINDOW[0], NC_TIME_WINDOW[1], color='gray', alpha=0.2, label='Nc Window (400-800ms)')
    ax_erp.legend()

    # Plot the topography of the difference
    diff_evoked = mne.combine_evoked([grand_avg_weighted, grand_avg_unweighted], weights=[1, -1])
    diff_evoked.comment = "Grand Average Difference"

    diff_evoked.plot_topomap(
        times=(NC_TIME_WINDOW[0] + NC_TIME_WINDOW[1]) / 2,
        average=NC_TIME_WINDOW[1] - NC_TIME_WINDOW[0],
        axes=[ax_topo, ax_cbar],
        show=False,
        colorbar=True
    )

    fig.suptitle(f"Grand Average Nc Analysis (N={len(all_unweighted_evokeds)})\nWeighted by {WEIGHTING_METRIC}",
                 fontsize=16)
    fig.tight_layout()
    plt.savefig(f"{get_data_path()}output/plots/grand_average_{WEIGHTING_METRIC}.svg")

else:
    print("\nNo data was processed, skipping grand average plot.")
