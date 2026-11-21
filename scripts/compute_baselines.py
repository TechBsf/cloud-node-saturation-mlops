import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.data.splitting import split_features_target
from src.schema import FEATURES, TARGET


def main() -> None:
    df = pd.read_csv("data/compute_capacity.csv")
    x_train, _, x_test, y_train, _, y_test = split_features_target(
        df,
        FEATURES,
        TARGET,
    )
    x_train_array = x_train.to_numpy()
    x_test_array = x_test.to_numpy()

    models = {
        "Baseline (mean predictor)": DummyRegressor(strategy="mean").fit(
            x_train_array,
            y_train,
        ),
        "Linear Regression": LinearRegression().fit(x_train_array, y_train),
        "Selected model": joblib.load("model/model.joblib"),
    }

    print("| Model | RMSE | MAE | R2 |")
    print("|---|---|---|---|")
    rmses: dict[str, float] = {}
    for name, m in models.items():
        prediction = m.predict(x_test_array)
        rmse = float(np.sqrt(mean_squared_error(y_test, prediction)))
        rmses[name] = rmse
        print(
            f"| {name} | {rmse:.4f} | "
            f"{mean_absolute_error(y_test, prediction):.4f} | "
            f"{r2_score(y_test, prediction):.4f} |"
        )

    gain = 100 * (1 - rmses["Selected model"] / rmses["Linear Regression"])
    print(f"\nRMSE reduction vs linear: {gain:.1f} %")


if __name__ == "__main__":
    main()
