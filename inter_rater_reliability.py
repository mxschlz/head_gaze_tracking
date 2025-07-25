import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix
import os


def calculate_cohens_kappa(file_path1, file_path2, column_index1=0, column_index2=0, labels=[1, 2]):
    """
    Calculates Cohen's Kappa and a summary of agreements/disagreements
    for two data files (CSV or Excel).

    Args:
        file_path1 (str): Path to the first data file (e.g., rater 1).
        file_path2 (str): Path to the second data file (e.g., rater 2).
        column_index1 (int): The index of the column to use for the first file.
        column_index2 (int): The index of the column to use for the second file.
        labels (list, optional): The list of labels to compare. Must contain two values.
                                 Defaults to [1, 2].

    Returns:
        tuple: (Cohen's Kappa score (float), summary_dict (dict)) or (None, None) if an error occurs.
               The summary_dict contains counts of agreements and disagreements.
    """
    try:
        if len(labels) != 2:
            print("Error: The 'labels' parameter must contain exactly two values.")
            return None, None

        rater_data_list = []
        # Use a tuple to associate files with their respective column indices
        files_and_cols = [(file_path1, column_index1), (file_path2, column_index2)]

        for file_path, column_index in files_and_cols:
            _, file_extension = os.path.splitext(file_path.lower())

            if file_extension == '.csv':
                # Assume the generated CSV has a header, so use header=0
                data = pd.read_csv(file_path, header=0, usecols=[column_index]).iloc[:, 0]
            elif file_extension in ['.xlsx', '.xls']:
                # Assume Excel file might not have a header row
                data = pd.read_excel(file_path, header=None, usecols=[column_index]).iloc[:, 0]
            else:
                print(f"Error: Unsupported file type for {file_path}. Please use .csv, .xlsx, or .xls.")
                return None, None
            rater_data_list.append(data)

        rater1_data, rater2_data = rater_data_list[0], rater_data_list[1]

        if len(rater1_data) != len(rater2_data):
            # Try to align by trimming the longer one, which can happen if one file has an extra empty row etc.
            min_len = min(len(rater1_data), len(rater2_data))
            print(f"Warning: Files have different lengths ({len(rater1_data)} vs {len(rater2_data)}). Trimming to {min_len} rows.")
            rater1_data = rater1_data.head(min_len)
            rater2_data = rater2_data.head(min_len)


        if len(rater1_data) == 0:
            print("Error: Files are empty or the specified column could not be read.")
            return None, None

        kappa = cohen_kappa_score(rater1_data, rater2_data, labels=labels)

        cm = confusion_matrix(rater1_data, rater2_data, labels=labels)

        # Unpack the confusion matrix based on the provided labels
        label1, label2 = labels[0], labels[1]

        agreement_both_l1 = int(cm[0, 0])
        disagreement_r1_l1_r2_l2 = int(cm[0, 1])
        disagreement_r1_l2_r2_l1 = int(cm[1, 0])
        agreement_both_l2 = int(cm[1, 1])

        total_observations = len(rater1_data)
        observed_agreement_count = agreement_both_l1 + agreement_both_l2
        observed_agreement_proportion = observed_agreement_count / total_observations if total_observations > 0 else 0

        summary = {
            "cohen_kappa": kappa,
            "total_observations": total_observations,
            f"agreement_both_{label1}": agreement_both_l1,
            f"agreement_both_{label2}": agreement_both_l2,
            f"disagreement_rater1_{label1}_rater2_{label2}": disagreement_r1_l1_r2_l2,
            f"disagreement_rater1_{label2}_rater2_{label1}": disagreement_r1_l2_r2_l1,
            "total_agreements": observed_agreement_count,
            "total_disagreements": disagreement_r1_l1_r2_l2 + disagreement_r1_l2_r2_l1,
            "observed_agreement_proportion": observed_agreement_proportion,
        }

        return kappa, summary

    except FileNotFoundError:
        print("Error: One or both files not found.")
        return None, None
    except pd.errors.EmptyDataError:
        print("Error: One or both files are empty.")
        return None, None
    except (IndexError, ValueError) as e:
        print(f"Error reading file or processing data: {e}. Check column indices and file formats.")
        return None, None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None, None


if __name__ == "__main__":
    # Example usage of the improved function
    file_path1 = "/home/max/Insync/schulz.max5@gmail.com/GoogleDrive/PhD/data/baby_vids/input/SMS014_A_VideoCoding.xlsx"
    file_path2 = "/home/max/Insync/schulz.max5@gmail.com/GoogleDrive/PhD/data/baby_vids/logs/SMS019_A_baby_gaze_trial_summary_part1_20250623_075239.csv"

    # NOTE: Adjust column indices as needed.
    # For the generated summary, 'looked_at_stimulus' is the 7th column (index 6).
    # For the human-coded Excel file, it might be a different column.

    # The function now defaults to comparing labels [1, 2].
    # You can override this by passing the `labels` argument, e.g., labels=[0, 1].
    kappa_result, summary_details = calculate_cohens_kappa(
        file_path1,
        file_path2,
        column_index1=0,  # Example: 1st column for the Excel file
        column_index2=6,  # 7th column for the generated CSV
        labels=[1, 2]      # Explicitly comparing labels 1 and 2
    )

    if kappa_result is not None and summary_details is not None:
        print(f"\nCohen's Kappa: {kappa_result:.4f}")
        print("\nDetailed Summary of (Rater1 vs Rater2):")
        for key, value in summary_details.items():
            if isinstance(value, float):
                print(f"  {key.replace('_', ' ').title()}: {value:.4f}")
            else:
                print(f"  {key.replace('_', ' ').title()}: {value}")
    else:
        print("\nCould not calculate Cohen's Kappa or summary due to errors.")
