# TODOS LOS IMPORTS
# data manipulation and plotting
# for saving the pipeline
from feature_engine.encoding import OneHotEncoder
from feature_engine.imputation import AddMissingIndicator, MeanMedianImputer

# from feature-engine
from feature_engine.selection import DropFeatures
from sklearn.linear_model import LogisticRegression

# from Scikit-learn
from sklearn.pipeline import Pipeline

from my_model.config.core import config

# the model


genero_pipe = Pipeline(
    [
        # ===== IMPUTATION =====
        (
            "drop_features",
            DropFeatures(features_to_drop=config.model_config.drop_features),
        ),
        # add missing indicator
        (
            "missing_indicator",
            AddMissingIndicator(variables=config.model_config.numerical_vars_with_na),
        ),
        # impute numerical variables with the mean
        (
            "mean_imputation",
            MeanMedianImputer(
                imputation_method="mean",
                variables=config.model_config.numerical_vars_with_na,
            ),
        ),
        # convert variables categorical to numerics
        ("encoder", OneHotEncoder(variables=config.model_config.variable_encoder)),
        # modelo
        ("LogisticRegression", LogisticRegression()),
    ]
)
