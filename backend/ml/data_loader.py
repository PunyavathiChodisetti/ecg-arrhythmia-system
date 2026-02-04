import pandas as pd
import ast

# CHANGE THIS PATH to your actual PTB-XL folder
BASE_PATH = r"data/ptb-xl"

def load_metadata():
    # Load main metadata
    df = pd.read_csv(f"{BASE_PATH}/ptbxl_database.csv")

    # Convert scp_codes from string → dict
    df["scp_codes"] = df["scp_codes"].apply(ast.literal_eval)

    # Load SCP statements
    scp_df = pd.read_csv(f"{BASE_PATH}/scp_statements.csv", index_col=0)

    # Keep only diagnostic statements
    scp_df = scp_df[scp_df["diagnostic"] == 1]

    # Map scp code → superdiagnostic class
    code_to_superclass = scp_df["diagnostic_class"].to_dict()

    def extract_superclass(scp_codes):
        classes = set()
        for code in scp_codes.keys():
            if code in code_to_superclass:
                classes.add(code_to_superclass[code])
        return list(classes)

    # Create new column
    df["superdiagnostic_class"] = df["scp_codes"].apply(extract_superclass)

    return df

if __name__ == "__main__":
    df = load_metadata()
    print(df[["ecg_id", "filename_lr", "superdiagnostic_class"]].head())