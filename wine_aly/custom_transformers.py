"""
The mod:`datamining_ai.custom_transformers` module implements the custom transformers needed
to build a composite estimator, as a chain of transforms and estimators of the titanic dataset.
"""
# Author: Fallou Tall

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


from datamining_ai import helpers


class ColDropper(BaseEstimator, TransformerMixin):
    """
    Drop specified columns from a pandas.DataFrame.

    Remove columns by specifying column names.

    Parameters
    ----------
    drop_col : boolean, optional, default True
    If False, no feature will be dropped.
    cols_to_drop : str or list
        single or list of columns to drop.
    """

    def __init__(self, drop_col=True, cols_to_drop=None):  # no *args or **kargs
        self.drop_col = drop_col
        self.cols_to_drop = cols_to_drop

    def fit(self, X, y=None):
        return self  # nothing else to do

    def transform(self, X):
        if self.drop_col and self.cols_to_drop is not None:
            X = X.drop(self.cols_to_drop, axis=1)
            return X
        else:
            return X


class MissingColIndicatorAdder(BaseEstimator, TransformerMixin):
    """
    Add a missing indicator feature to the Titanic data.

    A add feature that informs about missing status of a given columns or list of columns of Titanic.

    Parameters
    ----------
    add_missing_indicator : boolean, optional, default True
        If False, don't add the indicator feature to the data.
    """

    def __init__(
        self, add_missing_indicator=True, missing_cols=None
    ):  # no *args or **kargs
        self.add_missing_indicator = add_missing_indicator
        self.missing_cols = missing_cols

    def fit(self, X, y=None):
        return self  # nothing else to do

    def transform(self, X):
        if self.add_missing_indicator and self.missing_cols is not None:
            for col in self.missing_cols:
                X = X.assign(missing=helpers.add_indicator_col(col))
                X.rename(columns={"missing": f"{col}_missing"}, inplace=True)
            return X
        else:
            return X


class CustomImputer(BaseEstimator, TransformerMixin):
    """
     Imputation transformer for completing missing values.

    Imputation transformer for completing missing values
    by using the 'median' for numerical features and the
    'mode' for categorical features

     Parameters
     ----------
     impute : boolean, optional, default True
         If False, don't impute the data.
     strategy : str, optional, default "median"
         the strategy to use to impute the data.
         The possible values are 'mode' for categorical
         and 'median' for numerical features.
    """

    def __init__(self, strategy="median", impute=True):  # no *args or **kargs
        self.impute = impute
        self.strategy = strategy

    def fit(self, X, y=None):
        return self  # nothing else to do

    def transform(self, X):
        if self.impute:
            if self.strategy == "median":
                medians = X.median()
                X = X.fillna(medians)
                return X
            elif self.strategy == "mode":
                modes = X.mode()
                X = X.fillna(modes.iloc[0])
                return X
            else:
                raise Exception(
                    "The strategy parameter should be set to 'median' or 'mode'"
                )
        else:
            return X


class TitleAdder(BaseEstimator, TransformerMixin):
    """
    Add the title feature to the Titanic data.

    Extract the title  from the name feature of Titanic.

    Parameters
    ----------
    add_title : boolean, optional, default True
        If False, don't add the title feature to the data.
    """

    def __init__(self, add_title=True):  # no *args or **kargs
        self.add_title = add_title

    def fit(self, X, y=None):
        return self  # nothing else to do

    def transform(self, X):
        if self.add_title:
            X["title"] = X["name"].apply(helpers.get_title)
            return X
        else:
            return X


class TitleCategorizer(BaseEstimator, TransformerMixin):
    """
    Categorize the title feature of the Titanic data.

    Group the title in common people's title.

    Parameters
    ----------
    categorize_title : boolean, optional, default True
        If False, don't categorize the title feature.
    """

    def __init__(self, categorize_title=True):  # no *args or **kargs
        self.categorize_title = categorize_title

    def fit(self, X, y=None):
        return self  # nothing else to do

    def transform(self, X):
        if self.categorize_title:
            X["title"] = X["title"].replace(
                [
                    "Countess",
                    "Capt",
                    "Col",
                    "Don",
                    "Dr",
                    "Major",
                    "Rev",
                    "Sir",
                    "Jonkheer",
                    "Dona",
                ],
                "Rare",
            )
            X["title"] = X["title"].replace("Mlle", "Miss")
            X["title"] = X["title"].replace("Ms", "Miss")
            X["title"] = X["title"].replace("Mme", "Mrs")
            return X
        else:
            return X


class FamilySizeAdder(BaseEstimator, TransformerMixin):
    """
    Add the family_size feature to the Titanic data.

    Combine parch and sibsp to get the  family size of Titanic passengers.
    The family size of a sample `x` is calculated as:

        family_size = sibsp + parch + 1

    Parameters
    ----------
    add_family_size : boolean, optional, default True
        If False, don't add the family_size feature to the data.
    """

    def __init__(self, add_family_size=True):  # no *args or **kargs
        self.add_family_size = add_family_size

    def fit(self, X, y=None):
        return self  # nothing else to do

    def transform(self, X):
        if self.add_family_size:
            X["family_size"] = X["sibsp"] + X["parch"] + 1
            return X
        else:
            return X


class CustomOrdinalEncoder(BaseEstimator, TransformerMixin):
    """
    Encode categorical features as an integer array.

    The input to this transformer should be a pandas.series of integers or
    strings, denoting the values taken on by categorical (discrete) features.
    The features are converted to ordinal integers. This results in
    a single column of integers (0 to n_categories - 1) per feature.

    Parameters
    ----------
    encode_ordinal : boolean, optional, default True
        If False, don't ordinal encode the data.
    """

    def __init__(self, encode_ordinal=True):  # no *args or **kargs
        self.encode_ordinal = encode_ordinal

    def fit(self, X, y=None):
        return self  # nothing else to do

    def transform(self, X):
        if self.encode_ordinal:
            title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
            X["title"] = X["title"].map(title_mapping)
            X["title"] = X["title"].fillna(0)
            X["age"] = X["age"].astype(int)
            X.loc[X["age"] <= 16, "age"] = 0
            X.loc[(X["age"] > 16) & (X["age"] <= 32), "age"] = 1
            X.loc[(X["age"] > 32) & (X["age"] <= 48), "age"] = 2
            X.loc[(X["age"] > 48) & (X["age"] <= 64), "age"] = 3
            X.loc[X["age"] > 64, "age"] = 4
            return X
        else:
            return X


class CustomOneHotEncoder(BaseEstimator, TransformerMixin):
    """
    Encode categorical features as a one-hot numeric pandas.series.

    The input to this transformer should be a pandas.series integers or
    strings, denoting the values taken on by categorical (discrete) features.
    The features are encoded using a one-hot (aka 'one-of-K' or 'dummy')
    encoding scheme. This creates a binary column for each category

    By default, the encoder derives the categories based on the unique values
    in each feature.

    This encoding is needed for feeding categorical data to many scikit-learn
    estimators, notably linear models and SVMs with the standard kernels.

    Note: a one-hot encoding of y labels should use a LabelBinarizer
    instead.

    Parameters
    ----------
    encode_nominal : boolean, optional, default True
        If False, don't one hot encode the data.

    """

    def __init__(self, encode_nominal=True, drop_first=True):  # no *args or **kargs
        self.encode_nominal = encode_nominal
        self.drop_first = drop_first

    def fit(self, X, y=None):
        return self  # nothing else to do

    def transform(self, X):
        if self.encode_nominal:
            X = pd.get_dummies(X, drop_first=self.drop_first)
            return X
        else:
            return X


class CustomStandardScaler(BaseEstimator, TransformerMixin):
    """
    Standardize features by removing the mean and scaling to unit variance

    The standard score of a sample `x` is calculated as:

        z = (x - u) / s

    where `u` is the mean of the training samples,
    and `s` is the standard deviation of the training samples.

    Centering and scaling happen independently on each feature by computing
    the relevant statistics on the samples in the training set. Mean and
    standard deviation are then stored to be used on later data using
    :meth:`transform`.

    Standardization of a dataset is a common requirement for many
    machine learning estimators: they might behave badly if the
    individual features do not more or less look like standard normally
    distributed data (e.g. Gaussian with 0 mean and unit variance).

    For instance many elements used in the objective function of
    a learning algorithm (such as the RBF kernel of Support Vector
    Machines or the L1 and L2 regularizers of linear models) assume that
    all features are centered around 0 and have variance in the same
    order. If a feature has a variance that is orders of magnitude larger
    that others, it might dominate the objective function and make the
    estimator unable to learn from other features correctly as expected.

    Parameters
    ----------
    std_scale : boolean, optional, default True
        If False, dont standardize the data.
    """

    def __init__(self, std_scale=True):  # no *args or **kargs
        self.std_scale = std_scale

    def fit(self, X, y=None):
        return self  # nothing else to do

    def transform(self, X):
        if self.std_scale:
            features_to_scale = ["sibsp", "parch", "fare", "family_size"]
            X_std = X.copy()
            for col in features_to_scale:
                X_std[col] = (X_std[col] - X_std[col].mean()) / X_std[col].std()
            return X_std
        else:
            return X
