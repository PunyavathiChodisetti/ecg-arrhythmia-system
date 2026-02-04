import pandas as pd
import ast

BASE_PATH = r"data/ptb-xl"

def load_and_filter():
    df = pd.read_csv(f"{BASE_PATH}/ptbxl_database.csv")
    df["scp_codes"] = df["scp_codes"].apply(ast.literal_eval)

    scp_df = pd.read_csv(f"{BASE_PATH}/scp_statements.csv", index_col=0)
    scp_df = scp_df[scp_df["diagnostic"] == 1]
    code_to_superclass = scp_df["diagnostic_class"].to_dict()

    def extract_superclass(scp_codes):
        classes = set()
        for code in scp_codes.keys():
            if code in code_to_superclass:
                classes.add(code_to_superclass[code])
        return list(classes)

    df["superdiagnostic_class"] = df["scp_codes"].apply(extract_superclass)

    # Keep only single-label ECGs
    df = df[df["superdiagnostic_class"].apply(len) == 1]

    # Convert list → single value
    df["label"] = df["superdiagnostic_class"].apply(lambda x: x[0])

    print("Remaining samples:", len(df))
    print(df[["ecg_id", "filename_lr", "label"]].head())

    return df

if __name__ == "__main__":
    load_and_filter()
