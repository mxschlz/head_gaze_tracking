import mne
import pandas as pd
import pathlib
import matplotlib
matplotlib.use("TkAgg")
import mne_icalabel
import autoreject

# --- Import necessary functions from your existing project ---
# We need these to locate data and merge tracking/trial info.
from HeadGazeTracker import get_data_path
from merge_tracking_and_trial_data import merge_tracking_and_trial_data

preproc_params = dict(
    highpass=0.2,
    lowpass=20.0,
    ica_reject_threshold=0.8,
    montage='easycap-M1',
    reref_chs=["TP9", "TP10"],
    tmin=-0.2,
    tmax=0.8,
    reject=200e-6,
    n_jobs=-1
)

def calculate_intra_trial_fuzziness(log_dir, subjects_to_include):
    """
    Calculates the intra-trial head pose variability ('fuzziness') for each trial.

    This function processes head tracking logs to compute the standard deviation of
    Pitch and Yaw within each trial for the specified subjects.

    Args:
        log_dir (pathlib.Path): Directory containing the head tracking logs.
        subjects_to_include (list): A list of subject IDs to process.

    Returns:
        pd.DataFrame: A DataFrame with columns ['subject_id', 'session', 'Trial ID',
                      'Pitch_std', 'Yaw_std', 'fuzziness_score'].
    """
    print("--- Step 1: Calculating Intra-Trial Fuzziness from Head Tracking Data ---")
    eye_tracking_files = list(log_dir.glob("*eye_tracking_log*"))

    # Filter for the subjects we want to process
    if subjects_to_include:
        filtered_files = [
            f for f in eye_tracking_files
            if f.name.split('_')[0] in subjects_to_include
        ]
        eye_tracking_files = filtered_files

    all_trials_df_list = []
    for file_path in eye_tracking_files:
        # Reconstruct trial summary path from the tracking log path
        name_parts = file_path.name.split('_')
        date_time_part = '_'.join(name_parts[-2:])
        summary_file_name = f"{name_parts[0]}_{name_parts[1]}_trial_summary_{date_time_part}"
        trial_summary_path = file_path.with_name(summary_file_name)

        if not trial_summary_path.exists():
            print(f"  - Skipping {file_path.name}, no matching trial summary file found.")
            continue

        # Merge tracking data with trial info
        merged_df = merge_tracking_and_trial_data(str(file_path), str(trial_summary_path))
        if merged_df is None or merged_df.empty:
            continue

        # Extract subject and session IDs
        subject_id, session = name_parts[0], name_parts[1]
        merged_df['subject_id'] = subject_id
        merged_df['session'] = session

        # Filter for frames that are part of a trial
        trials_only_df = merged_df[merged_df['Trial ID'] > 0].copy()
        all_trials_df_list.append(trials_only_df)

    if not all_trials_df_list:
        print("  - No valid tracking data found. Aborting.")
        return None

    # Combine all data and calculate fuzziness
    full_df = pd.concat(all_trials_df_list, ignore_index=True)
    fuzziness_df = full_df.groupby(['subject_id', 'session', 'Trial ID']).agg(
        Pitch_std=('Pitch', 'std'),
        Yaw_std=('Yaw', 'std'),
        Pitch_mean=('Pitch', 'mean'),
        Yaw_mean=('Yaw', 'mean')
    ).reset_index()

    # Create a single composite fuzziness score (average of Pitch and Yaw std dev)
    fuzziness_df['fuzziness_score'] = fuzziness_df[['Pitch_std', 'Yaw_std']].mean(axis=1)
    # Create a single composite mean head pose score
    fuzziness_df['head_pose_mean'] = fuzziness_df[['Pitch_mean', 'Yaw_mean']].mean(axis=1)

    print(f"  - Calculated fuzziness for {len(fuzziness_df)} trials across {len(subjects_to_include)} subjects.")
    return fuzziness_df


def preprocess_eeg_and_add_metadata(subject_id, session, fuzziness_df, eeg_base_dir, output_dir):
    """
    Loads raw EEG, preprocesses it, creates epochs, and injects fuzziness metadata.

    Args:
        subject_id (str): The subject ID (e.g., 'SMS056').
        session (str): The session ID (e.g., 'C').
        fuzziness_df (pd.DataFrame): DataFrame containing fuzziness scores for all trials.
        eeg_base_dir (pathlib.Path): The base directory for raw EEG data (e.g., '.../input').
        output_dir (pathlib.Path): Directory to save the processed MNE Epochs file.
    """
    print(f"\n--- Step 2: Processing EEG for {subject_id} Session {session} ---")
    # --- 1. Find EEG data file ---
    session_dir = eeg_base_dir / subject_id / session
    if not session_dir.is_dir():
        print(f"  - EEG directory not found at {session_dir}. Skipping.")
        return

    # Use glob to find the .vhdr file, making the filename more flexible.
    # It looks for any file ending in .vhdr inside the session folder.
    possible_files = list(session_dir.glob('*.vhdr'))

    if not possible_files:
        print(f"  - No EEG header file (.vhdr) found in {session_dir}. Skipping.")
        return
    elif len(possible_files) > 1:
        print(f"  - WARNING: Found multiple .vhdr files in {session_dir}. Skipping to avoid ambiguity.")
        print(f"    Files found: {[f.name for f in possible_files]}")
        return
    else:
        eeg_file_path = possible_files[0]

    # --- 2. Load Raw EEG Data ---
    # We exclude the EOG channels from the main data for now
    try:
        raw = mne.io.read_raw_brainvision(eeg_file_path, eog=('VEOG', 'HEOG'), preload=True)
        print(f"  - Loaded raw data from {eeg_file_path.name}: {raw.info}")
    except Exception as e:
        print(f"  - ERROR: Failed to load or process {eeg_file_path.name}. Error: {e}")
        return

    # --- 3. Basic Preprocessing ---
    # Set standard montage (channel locations)
    # First, we correct for common case-sensitivity issues in channel names (e.g., 'PZ' -> 'Pz')
    # before setting the montage.
    channel_mapping = {
        'PZ': 'Pz',
    }
    # We use a try-except block in case some files don't have these channels
    try:
        raw.rename_channels(channel_mapping)
        print(f"  - Renamed channels to match standard montage (e.g., PZ -> Pz).")
    except Exception:
        print(f"  - No channels needed renaming.")
    raw.set_montage(preproc_params["montage"])

    raw.set_eeg_reference("average", projection=False)
    #raw.apply_proj()

    # Filter the data. These values are needed for the CNN to label the ICs effectively
    raw_filt = raw.copy().filter(1, 100)
    # apply ICA
    ica = mne.preprocessing.ICA(method="infomax", fit_params=dict(extended=True), random_state=42)
    ica.fit(raw_filt)
    ic_labels = mne_icalabel.label_components(raw_filt, ica, method="iclabel")
    exclude_idx = [idx for idx, (label, prob) in enumerate(zip(ic_labels["labels"], ic_labels["y_pred_proba"])) if label not in ["brain", "other"] and prob > preproc_params["ica_reject_threshold"]]
    print(f"Excluding these ICA components: {exclude_idx}")
    # ica.plot_properties(raw_filt, picks=exclude_idx)  # inspect the identified IC
    # ica.apply() changes the Raw object in-place, so let's make a copy first:
    reconst_raw = raw.copy()
    ica.apply(reconst_raw, exclude=exclude_idx)
    # band pass filter
    reconst_raw_filt = reconst_raw.copy().filter(preproc_params["highpass"], preproc_params["lowpass"])

    reconst_raw_filt.set_eeg_reference(preproc_params["reref_chs"])

    # --- 5. Find Trial Events from Markers ---
    # This regex captures the specific trial onset markers "S 21", "S 22", "S 23", and "S 24".
    # The pattern '^Stimulus/S 2[1-4]$' matches the full annotation description.
    events, event_id = mne.events_from_annotations(reconst_raw_filt, regexp='^Stimulus/S 2[1-4]$') # type: ignore

    if len(events) == 0:
        print("  - No trial onset events found in the EEG data. Skipping.")
        return

    print(f"  - Found {len(events)} trial onset events in the EEG data.")
    # Clean up event_id keys from numpy.str_ to standard str for clarity
    event_id = {str(key): val for key, val in event_id.items()}
    print(f"  - Mapped event IDs: {event_id}")

    # --- 6. Align Fuzziness Data with EEG Events ---
    # Get the fuzziness data for this specific subject and session
    session_fuzziness = fuzziness_df[
        (fuzziness_df['subject_id'] == subject_id) &
        (fuzziness_df['session'] == session)
    ].copy()

    # The 'Trial ID' from tracking data should correspond to the order of EEG events.
    # We assume Trial ID 1 corresponds to the 1st event, Trial ID 2 to the 2nd, etc.
    # Let's check if the counts match.
    if len(session_fuzziness) != len(events):
        print(f"  - WARNING: Mismatch between number of trials in tracking data ({len(session_fuzziness)}) "
              f"and EEG events ({len(events)}). Using the minimum of the two.")
        min_trials = min(len(session_fuzziness), len(events))
        session_fuzziness = session_fuzziness.iloc[:min_trials]
        events = events[:min_trials]

    # The MNE metadata must have the same number of rows as the events array.
    # We can directly use our `session_fuzziness` DataFrame.
    metadata = session_fuzziness.reset_index(drop=True)

    # --- 7. Create Epochs with Metadata & Artifact Rejection ---
    # Define the time window for each epoch (e.g., -200ms to +1500ms around trial onset)
    tmin, tmax = preproc_params["tmin"], preproc_params["tmax"]

    # OPTION 2 (Advanced): Use ICA (done in Section 4) and no rejection here.
    epochs = mne.Epochs(
        reconst_raw_filt,
        events=events,
        event_id=event_id,
        tmin=tmin,
        tmax=tmax,
        metadata=metadata,
        baseline=None,  # Baseline correct from start of epoch to event onset
        preload=True,
        # To use simple rejection (Option 1), uncomment the line below.
        reject=None
    )

    # run AutoReject
    ar = autoreject.AutoReject(n_jobs=preproc_params["n_jobs"], random_state=42)
    epochs_ar, log = ar.fit_transform(epochs, return_log=True)
    # If you used reject_criteria, you can see how many were dropped.
    # print(epochs.drop_log)

    print(f"  - Created {len(epochs)} clean epochs with fuzziness metadata.")
    print("  - Example of the first 5 rows of metadata:")
    print(epochs.metadata.head().to_string())

    # --- 8. Save the Processed Epochs ---
    output_path = output_dir / f"{subject_id}_{session}-epo.fif"
    epochs_ar.save(output_path, overwrite=True)
    print(f"  - Saved preprocessed epochs to: {output_path}")


if __name__ == '__main__':
    # --- Configuration ---
    # Define which subjects to process.
    subjects_to_include = [
        'SCS048',
        'SMM049',
        'SMM050',
        'SMM054',
        'SMS056',
    ]

    # Define paths using the central get_data_path() function
    base_data_path = get_data_path()
    log_dir = pathlib.Path(f"{base_data_path}output/logs")
    eeg_base_dir = pathlib.Path(f"{base_data_path}input")
    output_dir = pathlib.Path(f"{base_data_path}output/preprocessed_eeg")

    # Create the output directory if it doesn't exist
    output_dir.mkdir(exist_ok=True)

    # --- Main Execution ---
    # 1. Calculate fuzziness scores for all trials and subjects first.
    fuzziness_df = calculate_intra_trial_fuzziness(log_dir, subjects_to_include)

    if fuzziness_df is not None:
        # 2. Iterate through each unique subject/session and process their EEG data.
        # We group by subject and session from the fuzziness dataframe to find what to process.
        sessions_to_process = fuzziness_df[['subject_id', 'session']].drop_duplicates()

        for _, row in sessions_to_process.iterrows():
            subject_id = row['subject_id']
            session = row['session']
            preprocess_eeg_and_add_metadata(
                subject_id=subject_id,
                session=session,
                fuzziness_df=fuzziness_df,
                eeg_base_dir=eeg_base_dir,
                output_dir=output_dir
            )

        print("\n\nPreprocessing complete.")
        print("You can now load the '-epo.fif' files for further analysis.")
        print("Example: epochs = mne.read_epochs('path/to/your/output/SCS048_A-epo.fif')")
        print("Access metadata via: epochs.metadata")
        print("Use weights in analysis: your_model.fit(..., sample_weight=epochs.metadata['fuzziness_normalized'])")