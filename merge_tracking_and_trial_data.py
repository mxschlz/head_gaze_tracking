import pandas as pd
import os
import re


def merge_tracking_and_trial_data(tracking_log_path, trial_summary_path, output_path=None):
    """
    Merges an eye tracking log with a trial summary file by aligning relative timestamps.

    This function reads the frame-by-frame data from the tracking log and the
    trial-by-trial data from the summary. It then adds 'Trial ID' and 'Trial Phase'
    columns to the tracking data based on the timestamp of each frame.

    Args:
        tracking_log_path (str): Path to the main eye_tracking_log.csv file.
        trial_summary_path (str): Path to the trial_summary.csv file.
        output_path (str): Path to save the new, merged CSV file.
    """
    try:
        # 1. Load the datasets
        print(f"Loading tracking data from: {tracking_log_path}")
        tracking_df = pd.read_csv(tracking_log_path)
        print(f"Loading trial summary from: {trial_summary_path}")
        trials_df = pd.read_csv(trial_summary_path)
        print("Files loaded successfully.")

        # --- FIX: Use Frame Nr for alignment instead of absolute timestamp ---
        # 1. Validate required columns
        if 'Frame Nr' not in tracking_df.columns:
            raise ValueError("Tracking log must contain a 'Frame Nr' column.")

        required_trial_cols = {'trial_id', 'start_time_ms', 'stimulus_end_time_ms', 'trial_end_time_ms'}
        if not required_trial_cols.issubset(trials_df.columns):
            raise ValueError(f"Trial summary is missing one of the required columns: {required_trial_cols}")

        # 2. Get FPS from the log file name (e.g., ..._30.0fps.csv) or default to 30.0
        fps_match = re.search(r'_(\d+\.?\d*)fps', tracking_log_path)
        if fps_match:
            fps = float(fps_match.group(1))
            print(f"Extracted FPS from filename: {fps}")
        else:
            fps = 60.0
            print(f"Could not extract FPS from filename, defaulting to {fps} FPS.")

        # 3. Create a relative timestamp column in the main tracking dataframe
        tracking_df['Relative Timestamp (ms)'] = (tracking_df['Frame Nr'] * (1000.0 / fps)).astype(int)
        print("Created 'Relative Timestamp (ms)' column for alignment.")

        # 4. Convert trial summary timestamps to a numeric type for comparison
        for col in required_trial_cols:
            # trial_id might not always be numeric, so we skip it.
            if 'ms' in col:
                trials_df[col] = pd.to_numeric(trials_df[col])
        print("Trial summary timestamp columns converted to numeric type.")
        # --- END FIX ---

        # 2. Initialize new columns in the main tracking dataframe
        tracking_df['Trial ID'] = 0
        tracking_df['Trial Phase'] = 'inter-trial'

        # 3. Iterate through each trial from the summary and label the frames
        print(f"Processing {len(trials_df)} trials...")
        for _, trial in trials_df.iterrows():
            trial_id = trial['trial_id']
            start_ms = trial['start_time_ms']
            stim_end_ms = trial['stimulus_end_time_ms']
            trial_end_ms = trial['trial_end_time_ms']

            # Find frames within the stimulus period
            stim_mask = (tracking_df['Relative Timestamp (ms)'] >= start_ms) & (tracking_df['Relative Timestamp (ms)'] < stim_end_ms)
            tracking_df.loc[stim_mask, 'Trial ID'] = trial_id
            tracking_df.loc[stim_mask, 'Trial Phase'] = 'stimulus'

            # Find frames within the post-stimulus period
            post_stim_mask = (tracking_df['Relative Timestamp (ms)'] >= stim_end_ms) & (
                        tracking_df['Relative Timestamp (ms)'] < trial_end_ms)
            tracking_df.loc[post_stim_mask, 'Trial ID'] = trial_id
            tracking_df.loc[post_stim_mask, 'Trial Phase'] = 'post-stimulus'

        print("All trials processed.")

        # 4. Save the merged dataframe
        if output_path:
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)

            # Drop the temporary column before saving
            if 'Relative Timestamp (ms)' in tracking_df.columns:
                tracking_df = tracking_df.drop(columns=['Relative Timestamp (ms)'])

            tracking_df.to_csv(output_path, index=False)
            print(f"Successfully saved merged file to: {output_path}")

        return tracking_df

    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == '__main__':
    import pathlib
    # --- PLEASE UPDATE THESE PATHS ---
    # Path to your existing main data log
    tracking_log_path = 'G:\\Meine Ablage\\PhD\\data\\BABYGAZE\\output\\logs\\SMS059_C_eye_tracking_log_20251117_075725.csv'

    # Path to your existing trial summary log
    trial_summary_path = 'G:\\Meine Ablage\\PhD\\data\\BABYGAZE\\output\\logs\\SMS059_C_trial_summary_20251117_075725.csv'

    # Path where the new, combined file will be saved
    output_path = 'test.csv'
    # --- ------------------------- ---

    if 'path/to/your' in tracking_log_path:
        print("Please update the file paths in the script before running!")
    else:
        df = merge_tracking_and_trial_data(tracking_log_path, trial_summary_path, output_path)
