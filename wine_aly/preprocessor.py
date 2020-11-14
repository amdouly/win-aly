"""
The mod:`datamining_ai.preprocessor` module implements the preprocess() method to build a composite
estimator, as a chain of transforms and estimators of the titanic dataset.
"""
# Author: Fallou Tall
from sklearn.pipeline import Pipeline


from datamining_ai import custom_transformers as ct


def preprocess() -> Pipeline:
    """
    Pipeline of Titanic transforms with a final estimator.

    Sequentially apply a list of transforms and a final estimator.
    Intermediate steps of the pipeline must be 'transforms', that is, they
    must implement fit and transform methods.
    The final estimator only needs to implement fit.

    The purpose of the pipeline is to assemble several steps that can be
    cross-validated together while setting different parameters.

    :return: Pipeline
        a Scikit-Learn pipeline
    """
    cat_pipeline = Pipeline(
        [
            (
                "missing_col_dropper",
                ct.ColDropper(
                    cols_to_drop=["cabin", "boat", "body", "ticket", "home.dest"]
                ),
            ),
            (
                "missing_col_indicator",
                ct.MissingColIndicatorAdder(missing_cols=["embarked"]),
            ),
            ("cat_feat_imputer", ct.CustomImputer(strategy="mode")),
            ("title_adder", ct.TitleAdder()),
            ("title_categorizer", ct.TitleCategorizer()),
            ("name_dropper", ct.ColDropper(cols_to_drop=["name"])),
        ]
    )

    num_pipeline = Pipeline(
        [
            ("missing_col_dropper", ct.ColDropper()),
            (
                "missing_col_indicator",
                ct.MissingColIndicatorAdder(missing_cols=["age", "fare"]),
            ),
            ("cat_feat_imputer", ct.CustomImputer(strategy="median")),
            ("family_size_adder", ct.FamilySizeAdder()),
        ]
    )

    feature_transformation_pipeline = Pipeline(
        [
            ("ordinal_cat_feat_transformer", ct.CustomOrdinalEncoder()),
            ("nominal_cat_feat_transformer", ct.CustomOneHotEncoder()),
            ("num_feat_scaler", ct.CustomStandardScaler()),
        ]
    )

    preprocessing_pipeline = Pipeline(
        [
            ("cat_pipeline", cat_pipeline),
            ("num_pipeline", num_pipeline),
            ("feature_transformation_pipeline", feature_transformation_pipeline),
        ]
    )

    return preprocessing_pipeline
