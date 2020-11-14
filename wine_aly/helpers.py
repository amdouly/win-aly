import pandas as pd
import os
import re

CSV_PATH = os.path.join("..", "datasets")


def load_csv(csv_path: str = CSV_PATH, sep: str = ",") -> pd.DataFrame:
    """
    Read a comma-separated values (csv) file into DataFrame."
    :param csv_path: str
        File path.
    :return:
        None
    """
    return pd.read_csv(csv_path,sep)


def save_csv(df: pd.DataFrame, csv_path: str = CSV_PATH, index: bool = False) -> None:
    """
    Write a df to a comma-separated values (csv) file.
    :param df: pd.DataFrame
        a pandas dataframe.
    :param csv_path: str
        File path.
    :param index: bool, default False
        Write row names (index).
    :return:
        None
    """
    return df.to_csv(csv_path, index=index)


def get_corr_with_target(
    df: pd.DataFrame, col: str = "pclass", target: str = "survived"
) -> pd.DataFrame:
    """
    Give the correlation coefficient between a given column and the target variable survived.
    :param df: pd.DataFrame
        the dataset represented
    :param col: str
        the predictor feature
    :param target: str
        the target column
    :return:
    """
    return (
        df[[col, target]]
        .groupby([col], as_index=False)
        .mean()
        .sort_values(by=target, ascending=False)
    )


def add_indicator_col(col: str):
    """
    add a missing value indicator column
    :param col: str
        a pandas DataFrame column
    :return:
    """

    def wrapper(df: pd.DataFrame) -> pd.Series:
        return df[col].isna().astype(int)

    return wrapper


def snake_case_col(name):
    """
    Snakecaseify a string
    :param name: str
        column name
    :return:
    """
    return name.strip().lower().replace(" ", "_")


def get_title(name: str) -> str:
    """
    Give the title from the Titanic passenger name.

    Dans le code suivant, nous extrayons la fonction Title à l'aide d'expressions régulières.
    Le pattern RegEx '([A-Za-z]+)\.' correspond au premier mot qui se termine par un caractère point dans name.
    :param name: str
        name of the passenger
    :return:
    """
    title_search = re.search("([A-Za-z]+)\.", name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""
