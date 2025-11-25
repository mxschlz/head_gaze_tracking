import pandas as pd
import pathlib
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
from merge_tracking_and_trial_data import merge_tracking_and_trial_data
from inter_rater_reliability import calculate_cohens_kappa
from HeadGazeTracker.HeadGazeTracker import HeadGazeTracker # Import the class to use its static method
from HeadGazeTracker import get_data_path
plt.ion()

# --- NEW: Set a consistent and beautiful theme for all plots ---
custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", palette="colorblind", context="talk", font_scale=1.0, rc=custom_params)

plots_dir = pathlib.Path(f"{get_data_path()}output\plots")

log_dir = pathlib.Path(f"{get_data_path()}output\logs")
rater1_dir = pathlib.Path(f"{get_data_path()}output\human_coding\\rater_1")
rater2_dir = pathlib.Path(f"{get_data_path()}output\human_coding\\rater_2")
eye_tracking_file_patttern = "*eye_tracking_log*"

# --- NEW: Subject Selection ---
# Add subject IDs to this list to run the analysis only for them.
# If the list is empty, all subjects found in the log_dir will be processed.
subjects_to_include = [
    'SCS048',
    'SMM049',
    'SMM050',
    'SMM054',
    'SMS056',
    #'SCS063',
    #'SMS057',
    #'SMS059'
]


def analyze_attention_getter_effect(df, header_file, marker_file, eeg_sync_info):
    """
    Analyzes the re-orienting effect of attention getters on head pose.

    Args:
        df (pd.DataFrame): The merged dataframe containing tracking data.
        header_file (str): Path to the EEG .vhdr file.
        marker_file (str): Path to the EEG .vmrk file.
        eeg_sync_info (dict): Dictionary containing 'eeg_sampling_rate' and 'sample_offset'.

    Returns:
        pd.DataFrame: A dataframe with the analysis results for each attention getter event,
                      or None if no events are found.
    """
    # --- 1. Find Attention Getter Events in EEG data ---
    # Events S70-S79 are the attention getters.
    ag_events = [f"S{i}" for i in range(70, 80)]
    try:
        # We can reuse the static method from HeadGazeTracker to parse the EEG files
        _, ag_samples_raw = HeadGazeTracker.get_eeg_stimulus_times(
            header_file, marker_file, stimulus_description="Stimulus", events_of_interest=ag_events
        )
    except (ValueError, FileNotFoundError) as e:
        print(f"      - Could not process attention getters: {e}")
        return None

    # --- 2. Convert EEG samples to synchronized video milliseconds ---
    sampling_rate = eeg_sync_info['eeg_sampling_rate']
    sample_offset = eeg_sync_info['sample_offset']
    ag_onsets_ms = [int(((s - sample_offset) / sampling_rate) * 1000) for s in ag_samples_raw]

    # --- DEBUGGING: Check for timestamp mismatch ---
    # If the windows are empty, it's likely the EEG and video timestamps are not aligned.
    # Let's print the ranges to see if they overlap.
    min_video_ts, max_video_ts = df['Timestamp (ms)'].min(), df['Timestamp (ms)'].max()
    print(f"      - DEBUG: Video timestamp range: [{min_video_ts:.0f} ms, {max_video_ts:.0f} ms]")
    print(f"      - DEBUG: Calculated EEG event onsets: {ag_onsets_ms} ms")

    # --- 3. Analyze head pose around each event ---
    # Assume a fixed duration for the attention getter, e.g., 2000 ms.
    # This should match the experiment design.
    results = []
    for i, onset_ms in enumerate(ag_onsets_ms):
        # --- MODIFIED LOGIC ---
        # Instead of comparing pre-onset to post-offset, we will compare the head pose
        # from a baseline window just BEFORE the onset to a response window immediately AFTER the onset.
        # This is more robust and directly measures the re-orienting effect.
        # Baseline window: 1 second before the event.
        pre_onset_window = df[(df['Timestamp (ms)'] >= onset_ms - 100) & (df['Timestamp (ms)'] < onset_ms)]
        # Response window: 5 seconds immediately after the event starts.
        post_onset_window = df[(df['Timestamp (ms)'] > onset_ms) & (df['Timestamp (ms)'] <= onset_ms + 5000)]

        if not pre_onset_window.empty and not post_onset_window.empty:
            pre_pitch, pre_yaw = pre_onset_window[['Pitch', 'Yaw']].mean()
            post_pitch, post_yaw = post_onset_window[['Pitch', 'Yaw']].mean()
            # --- MODIFIED: Store pre and post values for better plotting ---
            results.append({
                'ag_id': i + 1,
                'onset_ms': onset_ms,
                'pre_pitch': pre_pitch,
                'pre_yaw': pre_yaw,
                'post_pitch': post_pitch,
                'post_yaw': post_yaw,
                'delta_pitch': post_pitch - pre_pitch,
                'delta_yaw': post_yaw - pre_yaw,
            })

    if not results:
        return None

    return pd.DataFrame(results)


def count_excluded_trials(file_path, column_index, valid_labels):
    """Counts total and excluded trials from a rater file."""
    try:
        if str(file_path).endswith('.csv'):
            df = pd.read_csv(file_path, header=None)
        else: # Assume Excel
            df = pd.read_excel(file_path, header=None)

        if column_index >= len(df.columns):
            return 0, 0 # Column doesn't exist

        # Ensure the column is numeric, coercing errors to NaN
        series = pd.to_numeric(df.iloc[:, column_index], errors='coerce')

        total_trials = len(series)
        # An excluded trial is one that is not in the list of valid labels.
        # This includes NaNs from empty cells or non-numeric values.
        excluded_trials = total_trials - series.isin(valid_labels).sum()

        return total_trials, excluded_trials
    except Exception as e:
        print(f"      - Warning: Could not read or process file for exclusion count: {file_path.name} ({e})")
        return 0, 0

# Find all eye tracking log files
eye_tracking_files = list(log_dir.glob(eye_tracking_file_patttern))

# --- NEW: Filter files based on subjects_to_include list ---
if subjects_to_include:
    print(f"\nFiltering analysis for {len(subjects_to_include)} specific subject(s): {', '.join(subjects_to_include)}")

    filtered_files = []
    for f in eye_tracking_files:
        # Extract subject ID from filename (e.g., 'SUBJ001_session1_...')
        subject_id_from_file = f.name.split('_')[0]
        if subject_id_from_file in subjects_to_include:
            filtered_files.append(f)

    eye_tracking_files = filtered_files
    print(f"Found {len(eye_tracking_files)} files matching the specified subjects.")
# --- END NEW ---

# List to hold individual dataframes
df_list = []
# List to hold reliability results
reliability_results = []
# List to hold attention getter analysis results
attention_getter_results = []
# List to hold excluded trial counts
excluded_trials_results = []

print(f"Found {len(eye_tracking_files)} eye tracking files. Processing...")
for file_path in eye_tracking_files:
    # Construct the path for the corresponding trial summary file
    # The trial summary filename doesn't contain the FPS info, so we need to reconstruct it.
    # Example: 'SUBJ_SESS_eye_tracking_log_60.0fps_DATE_TIME.csv' -> 'SUBJ_SESS_trial_summary_DATE_TIME.csv'
    name_parts = file_path.name.split('_')
    if 'eye' in name_parts:
        # Rebuild the name, replacing 'eye_tracking_log' and removing the fps part
        date_time_part = '_'.join(name_parts[-2:]) # e.g., '20251121_012515.csv'
        summary_file_name = f"{name_parts[0]}_{name_parts[1]}_trial_summary_{date_time_part}"
        trial_summary_path = file_path.with_name(summary_file_name)
    else:
        # Fallback for unexpected filenames
        print(f"  - Skipping {file_path.name}, filename does not match expected pattern.")
        continue

    if not trial_summary_path.exists():
        print(f"  - Skipping {file_path.name}, no matching trial summary file found.")
        continue

    # Merge the two files to get trial information
    merged_df = merge_tracking_and_trial_data(str(file_path), str(trial_summary_path))

    if merged_df is not None:
        merged_df['Timestamp (ms)'] = pd.to_numeric(merged_df['Timestamp (ms)'], errors='coerce')

        # --- FIX: Normalize video timestamps to be relative ---
        # The debug output shows video timestamps are absolute (e.g., from date/time),
        # while EEG timestamps are relative (ms from start). We must align them.
        # We do this by subtracting the first timestamp from the entire column.
        # --- FIX 3: Convert from microseconds to milliseconds ---
        # The scale of the video timestamps suggests they are in microseconds (not nanoseconds).
        if not merged_df.empty:
            start_time_ns = merged_df['Timestamp (ms)'].iloc[0]
            # 1. Subtract the start time to make them relative (still in ns).
            # 2. Divide by 1,000 to convert from microseconds to milliseconds.
            merged_df['Timestamp (ms)'] = (merged_df['Timestamp (ms)'] - start_time_ns) / 1_000

        # Extract subject ID and session from the filename
        parts = file_path.stem.split('_')
        subject_id = parts[0]
        session = parts[1]

        # Add the new identifiers to the dataframe
        merged_df['subject_id'] = subject_id
        merged_df['session'] = session

        # --- Inter-Rater Reliability Calculation ---
        # The algorithm's trial summary is what we compare against human coders.
        algo_summary_path = trial_summary_path

        # Find corresponding human-coded files
        file_prefix = f"{subject_id}_{session}"
        try:
            rater1_file = next(rater1_dir.glob(f"{file_prefix}*"))
            rater2_file = next(rater2_dir.glob(f"{file_prefix}*"))
        except StopIteration:
            print(f"  - WARNING: Could not find matching human-coded files for {file_prefix}. Skipping reliability check.")
            rater1_file, rater2_file = None, None

        if rater1_file and rater2_file:
            print(f"  - Found human coding files. Running reliability analysis for {file_prefix}...")
            # --- New: Count Excluded Trials per Rater ---
            raters_to_check = {
                "Algorithm": (algo_summary_path, 6),
                "Rater 1": (rater1_file, 0),
                "Rater 2": (rater2_file, 0)
            }
            for rater_name, (path, col_idx) in raters_to_check.items():
                total, excluded = count_excluded_trials(path, col_idx, valid_labels=[1])
                if total > 0:
                    excluded_trials_results.append({
                        "subject_id": subject_id,
                        "session": session,
                        "rater": rater_name,
                        "total_trials": total,
                        "excluded_trials": excluded,
                        "excluded_proportion": excluded / total if total > 0 else 0
                    })
                    print(f"    - {rater_name}: {excluded} of {total} trials excluded.")
            # --- End New ---

            # Define column indices.
            # Algorithm: 'looked_at_stimulus' is the 7th column (index 6).
            # Human: Assume it's the 1st column (index 0) in their Excel/CSV files.
            # The labels [1, 2] correspond to 'looked_at_stimulus' values.
            kappa_pairs = {
                "Rater1_vs_Rater2": (rater1_file, 0, rater2_file, 0),
                #"Algo_vs_Rater1": (algo_summary_path, 6, rater1_file, 0),
                #"Algo_vs_Rater2": (algo_summary_path, 6, rater2_file, 0),
            }

            for name, (file1, col1, file2, col2) in kappa_pairs.items():
                kappa, summary = calculate_cohens_kappa(
                    str(file1), str(file2),
                    column_index1=col1, column_index2=col2,
                    labels=[1, 2]
                )
                if kappa is not None:
                    reliability_results.append({
                        "subject_id": subject_id,
                        "session": session,
                        "comparison": name,
                        "kappa": kappa,
                        "agreement_proportion": summary.get("observed_agreement_proportion", 0)
                    })
                    # --- Moved: Add more detailed logging for diagnosis ---
                    print(f"    - {name}: Kappa={kappa:.3f}, Agreement={summary.get('observed_agreement_proportion', 0):.3f}")
                    # If Kappa is low despite high agreement, print the confusion matrix details
                    if kappa < 0.4 and summary.get('observed_agreement_proportion', 0) > 0.75:
                        print(f"      -> Low Kappa Warning: This may be due to category imbalance.")
                        print(f"         Agreement on '1' (True): {summary.get('agreement_both_1', 'N/A')}")
                        print(f"         Agreement on '2' (False): {summary.get('agreement_both_2', 'N/A')}")
                        print(f"         Total Disagreements: {summary.get('total_disagreements', 'N/A')}")
                else:
                    print(f"    - Could not calculate Kappa for {name}")
        # --- End of Reliability Calculation ---

        df_list.append(merged_df)
        print(f"  + Successfully processed and merged data for {subject_id} session {session}.")

        # --- New: Attention Getter Analysis ---
        # We need the path to the original EEG files and the sync info file.
        # Let's assume the sync info file is in the same log directory.
        sync_info_path = file_path.with_name(f"{subject_id}_{session}_eeg_sync_info.json")
        
        # --- MODIFIED: Use glob to find EEG files more robustly ---
        # The base directory for raw data seems to be 'input'
        eeg_base_dir = pathlib.Path(f"{get_data_path()}input")
        
        # Recursively search within the subject's directory for any .vhdr/.vmrk file
        # that contains both the subject ID and the session string.
        # This is more flexible than assuming a specific folder structure.
        # e.g., it will find '.../input/SMS056/C/SMS056_C_eeg.vhdr'
        # when subject is 'SMS056' and session is 'C'.
        search_pattern = f"**/{subject_id}*{session}*.vhdr"
        header_files = list(eeg_base_dir.glob(search_pattern))
        
        marker_files = [p.with_suffix('.vmrk') for p in header_files]

        if sync_info_path.exists() and header_files and marker_files:
            if len(header_files) > 1 or len(marker_files) > 1:
                print(f"  - WARNING: Found multiple possible EEG files for {file_prefix}. Using the first one found.")
            
            header_file = header_files[0]
            marker_file = marker_files[0]

            print(f"  - Found EEG sync info. Running attention getter analysis for {file_prefix}...")
            import json
            with open(sync_info_path, 'r') as f:
                eeg_sync_info = json.load(f)

            ag_df = analyze_attention_getter_effect(merged_df, str(header_file), str(marker_file), eeg_sync_info)

            if ag_df is not None:
                ag_df['subject_id'] = subject_id
                ag_df['session'] = session
                attention_getter_results.append(ag_df)
        else:
            print(f"  - WARNING: Could not find EEG files or sync info for {file_prefix}. Skipping attention getter analysis.")

# Concatenate all dataframes into a single one
df = pd.concat(df_list, ignore_index=True)

# --- New Plot: Angle Distribution Across Trials ---
# Filter out the inter-trial data to focus only on frames within a trial
trials_only_df = df[df['Trial ID'] > 0].copy()

# --- New Plot: Distribution of Trial-Averaged Angles ---
# Group by subject, session, and trial to calculate the average angle for each trial
print("\nCalculating trial-averaged angles...")
averaged_trials_df = trials_only_df.groupby(['subject_id', 'session', 'Trial ID']).agg(
    Pitch=('Pitch', 'mean'),
    Yaw=('Yaw', 'mean')
).reset_index()

# Melt the averaged data into a long format for plotting
averaged_long_df = averaged_trials_df.melt(id_vars=['subject_id', 'Trial ID'], value_vars=['Pitch', 'Yaw'],
                                           var_name='angle_type', value_name='average_angle_value')

# --- MODIFIED: Plot individual subjects AND the grand average ---
# 1. Plot each individual subject's KDE with low alpha
g = sns.displot(
    data=averaged_long_df, x="average_angle_value", col="angle_type",
    kind="kde", hue="subject_id", common_norm=False,
    fill=False, legend=False,  # Disable default legend, we create a custom one
    alpha=0.7, linewidth=2,
    facet_kws=dict(sharey=True, sharex=False), height=7, aspect=1.1
)

# 2. Overlay the grand average KDE on each facet
for i, angle_type in enumerate(['Pitch', 'Yaw']):
    ax = g.axes.flat[i]
    # Filter data for the current angle type
    subset_df = averaged_long_df[averaged_long_df['angle_type'] == angle_type]
    # Plot the grand average KDE on the axis
    sns.kdeplot(data=subset_df, x='average_angle_value', color='black', linewidth=4, ax=ax, label='Grand Average')
    # Add a vertical reference line at 0
    ax.axvline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)

# 3. Create a custom legend
g.fig.suptitle("Distribution of Mean Head Angles per Trial")
g.set_axis_labels("Mean Angle per Trial (degrees)", "Density")
g.set_titles("{col_name} Angle")
plt.tight_layout()
g.savefig(plots_dir / "distribution_mean_head_angles.svg")

# --- New: Calculate and Plot "fuzziness Score" ---
# The fuzziness score is the standard deviation of the trial-averaged angles for each subject/session.
# A higher score means more variability in head position across trials.
print("\nCalculating fuzziness scores (std dev of trial-averaged angles)...")
fuzziness_df = averaged_trials_df.groupby(['subject_id', 'session']).agg(
    Pitch_std=('Pitch', 'std'),
    Yaw_std=('Yaw', 'std')
).reset_index()

# Calculate a single composite score by averaging the std deviations.
fuzziness_df['fuzziness'] = fuzziness_df[['Pitch_std', 'Yaw_std']].mean(axis=1)

# Create a bar plot to visualize the fuzziness scores
plt.figure(figsize=(18, 9))
sns.barplot(data=fuzziness_df, x='subject_id', y='fuzziness', hue="session", dodge=True, palette="mako")
plt.title('Overall Head Pose Variability (fuzziness)')
plt.ylabel('Mean SD of Head Angles (degrees)')
plt.xlabel('Subject ID')
plt.legend(title="Session")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(plots_dir / "fuzziness_score_variability.svg")

# --- New: Temporal Evolution of Intra-Trial Head Pose Variability ---
# To test the hypothesis that movement increases over time, we calculate the
# standard deviation of head pose *within* each trial.
print("\nCalculating intra-trial head pose variability to see temporal evolution...")
intra_trial_variability_df = trials_only_df.groupby(['subject_id', 'session', 'Trial ID']).agg(
    Pitch_std=('Pitch', 'std'),
    Yaw_std=('Yaw', 'std')
).reset_index()

# Melt the data for plotting
intra_trial_variability_long_df = intra_trial_variability_df.melt(
    id_vars=['subject_id', 'session', 'Trial ID'],
    value_vars=['Pitch_std', 'Yaw_std'],
    var_name='Angle_Component',
    value_name='Intra_Trial_Std_Dev'
)

# Create a line plot showing the average trend of variability over trials
# --- MODIFIED: Plot individual subjects AND the grand average ---
# 1. Plot each individual subject's line with low alpha
g = sns.relplot(data=intra_trial_variability_long_df, x='Trial ID', y='Intra_Trial_Std_Dev',
                col='Angle_Component', kind='line', hue='subject_id',
                aspect=1.4, height=6,
                legend=False, alpha=0.4, linewidth=1.5, errorbar=None)

# 2. Overlay the grand average line on each facet
for i, angle_type in enumerate(['Pitch_std', 'Yaw_std']):
    ax = g.axes.flat[i]
    sns.lineplot(data=intra_trial_variability_long_df[intra_trial_variability_long_df['Angle_Component'] == angle_type],
                 x='Trial ID', y='Intra_Trial_Std_Dev',
                 color='black', linewidth=4, errorbar=None, ax=ax, label='Grand Average')

g.fig.suptitle('Intra-Trial Head Pose Variability Over Trials (fuzziness)')
g.set_axis_labels("Trial ID", "Intra-Trial fuzziness (degrees)")
g.set_titles("{col_name} Angle")
g.map(plt.axhline, y=0, color='black', linestyle='--', linewidth=1.5, alpha=0.7) # Add reference line
plt.tight_layout()
g.savefig(plots_dir / "intra_trial_variability_evolution.svg")

# --- New: Temporal Evolution of Mean Head Pose per Subject ---
# To see how each subject's average head pose changes over the experiment.
# We can reuse the 'averaged_long_df' which is already calculated.
print("\nPlotting temporal evolution of mean head pose per subject...")

# --- MODIFIED: Plot individual subjects AND the grand average ---
# 1. Plot each individual subject's line with low alpha
g = sns.relplot(data=averaged_long_df, x='Trial ID', y='average_angle_value', col='angle_type',
                kind='line', hue='subject_id', errorbar=None,
                facet_kws=dict(sharey=True), aspect=1.5, height=5,
                legend=False, alpha=0.4, linewidth=1.5)

# 2. Overlay the grand average line on each facet
for i, angle_type in enumerate(['Pitch', 'Yaw']):
    ax = g.axes.flat[i]
    sns.lineplot(data=averaged_long_df[averaged_long_df['angle_type'] == angle_type],
                 x='Trial ID', y='average_angle_value',
                 color='black', linewidth=4, errorbar=None, ax=ax, label='Grand Average')

g.fig.suptitle('Mean Head Pose Over Trials')
g.set_axis_labels("Trial ID", "Mean Angle (degrees)")
g.set_titles("{col_name} Angle")
g.map(plt.axhline, y=0, color='black', linestyle='--', linewidth=1.5, alpha=0.7) # Add reference line

# 3. Add a clear legend
plt.tight_layout()
g.savefig(plots_dir / "mean_head_pose_evolution.svg")

# --- New: Display Reliability Results ---
if reliability_results:
    print("\n\n--- Inter-Rater Reliability Analysis ---")
    reliability_df = pd.DataFrame(reliability_results)

    # Display a summary table
    reliability_summary = reliability_df.groupby('comparison').agg(
        mean_kappa=('kappa', 'mean'),
        std_kappa=('kappa', 'std'),
        mean_agreement=('agreement_proportion', 'mean')
    ).reset_index()

    print("\nAverage Reliability Scores Across All Subjects:")
    print(reliability_summary.to_string(index=False))

    # Create a bar plot to visualize the kappa scores
    plt.figure(figsize=(18, 9))
    sns.barplot(data=reliability_df, x='subject_id', y='kappa', hue='comparison', dodge=True, palette="rocket")
    plt.title('Cohen\'s Kappa: Algorithm vs. Human Raters')
    plt.ylabel('Cohen\'s Kappa Score')
    plt.xlabel('Subject ID')
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.7, label='Chance Agreement (0.0)')
    plt.axhline(y=0.6, color='grey', linestyle='--', label='Substantial Agreement (0.6)')
    plt.axhline(y=0.8, color='black', linestyle='--', label='Almost Perfect Agreement (0.8)')
    plt.tight_layout()
    plt.savefig(plots_dir / "cohens_kappa_reliability.svg")
else:
    print("\nNo reliability analysis was performed (no matching human-coded files found).")

# --- New: Display and Plot Attention Getter Results ---
if attention_getter_results:
    print("\n\n--- Attention Getter Re-orientation Analysis ---")
    ag_results_df = pd.concat(attention_getter_results, ignore_index=True)

    # Display a summary table
    # We now average the pre and post values for the summary
    ag_summary = ag_results_df.groupby(['subject_id', 'session']).agg(
        mean_pre_pitch=('pre_pitch', 'mean'),
        mean_post_pitch=('post_pitch', 'mean'),
        mean_pre_yaw=('pre_yaw', 'mean'),
        mean_post_yaw=('post_yaw', 'mean')
    ).reset_index()

    print("\nAverage Head Pose Change After Attention Getter (Post-Pre):")
    print(ag_summary.to_string(index=False))

    # --- MODIFIED PLOT: Create a dumbbell plot to show the "before vs. after" effect ---
    # We need to melt the data to have a 'condition' (pre/post) column for plotting
    ag_long_df = ag_summary.melt(
        id_vars=['subject_id', 'session'],
        value_vars=['mean_pre_pitch', 'mean_post_pitch', 'mean_pre_yaw', 'mean_post_yaw'],
        var_name='measurement',
        value_name='angle'
    )
    ag_long_df['condition'] = ag_long_df['measurement'].apply(lambda x: 'Pre-Onset' if 'pre' in x else 'Post-Onset')
    ag_long_df['angle_type'] = ag_long_df['measurement'].apply(lambda x: 'Pitch' if 'pitch' in x else 'Yaw')

    # --- NEW: Create a unique ID for each subject-session pair for plotting ---
    ag_long_df['subject_session'] = ag_long_df['subject_id'] + " (" + ag_long_df['session'] + ")"

    # Create a FacetGrid to have separate plots for Pitch and Yaw
    # Hue is now mapped to the unique session to give each a distinct color.
    g = sns.FacetGrid(ag_long_df, col="angle_type", hue="subject_session",
                      sharex=True, sharey=False, height=8, aspect=1.2)

    # Map the line plot to draw the "dumbbell" lines
    g.map(sns.lineplot, "angle", "subject_session", marker="", sort=False, estimator=None, lw=3, alpha=0.8)

    # Map the scatter plot to draw the "before" and "after" points
    g.map_dataframe(sns.scatterplot, x="angle", y="subject_session", style="condition",
                    markers={'Pre-Onset': 'X', 'Post-Onset': 'o'}, s=250, legend=False, zorder=5)

    # Add a vertical line at 0 to represent the baseline
    g.map(plt.axvline, x=0, color='black', linestyle='--', lw=2, alpha=0.7)

    g.set_titles("{col_name} Angle Re-orientation", size=18)
    g.set_axis_labels("Head Angle (degrees)", "Subject (Session)")

    # --- FIX: Manually create the legend to resolve the TypeError ---
    # The hue legend is too busy, so we only add the marker legend.
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], marker='X', color='w', label='Pre-Onset', markerfacecolor='k', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Post-Onset', markerfacecolor='k', markersize=10)
    ]
    g.add_legend(handles=legend_handles, title="Condition", label_order=['Pre-Onset', 'Post-Onset'])

    g.fig.suptitle("Head Pose Re-orientation Following Attention Getters", fontsize=22, y=1.03)
    plt.tight_layout(rect=[0, 0, 1, 0.98])

    g.savefig(plots_dir / "attention_getter_reorientation_effect.svg")

    # --- NEW: Bar plot showing average pre vs. post absolute head pose ---
    # This provides a clear summary of the overall effect.
    print("\nPlotting summary bar plot for attention getter effect...")
    bar_plot_df = ag_long_df.copy()
    bar_plot_df['absolute_angle'] = bar_plot_df['angle'].abs()

    # --- MODIFIED: Use catplot to facet by subject ---
    g = sns.catplot(
        data=bar_plot_df,
        x='angle_type',
        y='absolute_angle',
        hue='condition',
        col='subject_id',
        kind='bar',
        errorbar='se',  # Show standard error of the mean
        col_wrap=4,  # Wrap plots after 4 columns
        height=5,
        aspect=1.1
    )
    g.fig.suptitle('Absolute Head Pose Before vs. After Attention Getter (by Subject)', y=1.03)
    g.set_axis_labels("Angle Component", "Average Absolute Angle (degrees)")
    g.set_titles("Subject: {col_name}")
    g.tight_layout(rect=[0, 0, 1, 1])
    g.savefig(plots_dir / "attention_getter_pre_vs_post_barplot_by_subject.svg")

# --- New: Plot Excluded Trial Proportions ---
if excluded_trials_results:
    print("\n\n--- Excluded Trials Analysis ---")
    excluded_df = pd.DataFrame(excluded_trials_results)

    # Display a summary table
    excluded_summary = excluded_df.groupby('rater').agg(
        mean_excluded_proportion=('excluded_proportion', 'mean'),
        mean_excluded_count=('excluded_trials', 'mean')
    ).reset_index()

    print("\nAverage Proportion of Excluded Trials by Rater:")
    print(excluded_summary.to_string(index=False))

    plt.figure(figsize=(12, 8))
    sns.barplot(data=excluded_df, x='rater', y='excluded_proportion', errorbar='se', palette="crest")
    plt.title('Proportion of Excluded Trials')
    plt.ylabel('Proportion of Trials Excluded')
    plt.xlabel('Coder')
    plt.tight_layout()
    plt.savefig(plots_dir / "excluded_trials_proportion.svg")

# --- New: Average Head Pose by Session ---
print("\nCalculating average head pose per session...")
# We can reuse averaged_trials_df which has the mean pose for each trial
session_avg_pose_df = averaged_trials_df.groupby(['subject_id', 'session']).agg(
    Pitch=('Pitch', 'mean'),
    Yaw=('Yaw', 'mean')
).reset_index()

# Melt for plotting
session_avg_pose_long_df = session_avg_pose_df.melt(
    id_vars=['subject_id', 'session'],
    value_vars=['Pitch', 'Yaw'],
    var_name='Angle Component',
    value_name='Mean Angle (degrees)'
)

print("\nAverage Head Pose per Subject and Session:")
print(session_avg_pose_df.to_string())

# --- MODIFIED: Plot the overall absolute deviation from baseline, independent of subject ---
# Calculate the absolute mean angle to capture the magnitude of deviation.
session_avg_pose_long_df['Absolute Mean Angle (degrees)'] = session_avg_pose_long_df['Mean Angle (degrees)'].abs()

# Create a categorical plot to visualize the results, with session on the x-axis.
g = sns.catplot(data=session_avg_pose_long_df, x='session', y='Absolute Mean Angle (degrees)',
                hue='session', col='Angle Component',
                kind='bar', palette='deep', height=7, aspect=1.2,
                legend=False)

# Overlay a stripplot to show the individual subject data points.
g.map_dataframe(sns.stripplot, x='session', y='Absolute Mean Angle (degrees)',
                hue='session', palette=['#444444'], dodge=False,
                edgecolor='lightgrey', linewidth=1, jitter=0.15)

g.fig.suptitle('Overall Head Pose Deviation by Session')
g.set_axis_labels("Session", "Absolute Mean Angle (degrees)")
g.set_titles("{col_name} Angle")
g.tight_layout()
g.savefig(plots_dir / "mean_head_pose_by_session.svg")
