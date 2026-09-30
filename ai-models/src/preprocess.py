from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

ZERO_AS_MISSING = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']


def build_preprocessor():
    """Build the train-fold-fitted imputer used by the model pipelines.

    Zero values in the listed medical measurements are treated as missing.
    SimpleImputer learns medians only when the surrounding Pipeline is fit,
    which keeps cross-validation folds and the held-out test set isolated.
    """
    return ColumnTransformer(
        [('median', SimpleImputer(strategy='median', missing_values=0), ZERO_AS_MISSING)],
        remainder='passthrough',
    )
