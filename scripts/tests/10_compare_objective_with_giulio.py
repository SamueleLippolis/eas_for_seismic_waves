# scripts/tests/10_compare_objective_with_giulio.py

from pathlib import Path
import sys
import csv
import math

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

OUR_PATH = ROOT / "reports" / "our_objective_check.csv"
GIULIO_PATH = ROOT / "reports" / "giulio_objective_check.csv"


COLUMNS_TO_COMPARE = [
    "C_true",
    "phi_percent",
    "C_percent",
    "S_b_percent",
    "sigma_b_inv",
    "xi",
    "Vp_obs",
    "Vs_obs",
    "sigma_obs",
    "objective",
]


def read_csv(path):
    rows = []

    with open(path, "r") as f:
        reader = csv.DictReader(f)

        for row in reader:
            converted = {}

            for key, value in row.items():
                if key == "candidate_name":
                    converted[key] = value
                else:
                    converted[key] = float(value)

            rows.append(converted)

    return rows


def main():
    our_rows = read_csv(OUR_PATH)
    giulio_rows = read_csv(GIULIO_PATH)

    if len(our_rows) != len(giulio_rows):
        raise ValueError(
            f"Different number of rows: ours={len(our_rows)}, "
            f"giulio={len(giulio_rows)}"
        )

    global_max_abs_diff = 0.0

    for idx, (our_row, giulio_row) in enumerate(zip(our_rows, giulio_rows)):
        if our_row["candidate_name"] != giulio_row["candidate_name"]:
            raise ValueError(
                f"Candidate mismatch at row {idx}: "
                f"ours={our_row['candidate_name']}, "
                f"giulio={giulio_row['candidate_name']}"
            )

        print(
            f"Row {idx} | C_true={our_row['C_true']} | "
            f"candidate={our_row['candidate_name']}"
        )

        for column in COLUMNS_TO_COMPARE:
            our_value = our_row[column]
            giulio_value = giulio_row[column]
            abs_diff = abs(our_value - giulio_value)

            global_max_abs_diff = max(global_max_abs_diff, abs_diff)

            print(
                f"  {column:12s} | "
                f"ours={our_value:.12g} | "
                f"giulio={giulio_value:.12g} | "
                f"abs_diff={abs_diff:.3e}"
            )

        print()

    print("Global max absolute difference:")
    print(f"  {global_max_abs_diff:.3e}")

    if math.isclose(global_max_abs_diff, 0.0, abs_tol=1e-9):
        print("\nResult: OK — objective matches Giulio.")
    else:
        print("\nResult: NOT IDENTICAL — inspect objective or input values.")


if __name__ == "__main__":
    main()