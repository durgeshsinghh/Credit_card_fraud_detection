import pathlib
import joblib
import sys
import pandas as pd
from sklearn import metrics
from dvclive import Live
from matplotlib import pyplot as plt


def evaluate(model, X, y, split, live):
    predictions_by_class = model.predict_proba(X)
    predictions = predictions_by_class[:, 1]

    avg_prec = metrics.average_precision_score(y, predictions)
    roc_auc = metrics.roc_auc_score(y, predictions)

    if not live.summary:
        live.summary = {"avg_prec": {}, "roc_auc": {}}

    live.summary["avg_prec"][split] = avg_prec
    live.summary["roc_auc"][split] = roc_auc

    live.log_sklearn_plot("roc", y, predictions, name=f"roc/{split}")

    live.log_sklearn_plot(
        "precision_recall",
        y,
        predictions,
        name=f"prc/{split}",
        drop_intermediate=True,
    )

    live.log_sklearn_plot(
        "confusion_matrix",
        y,
        predictions_by_class.argmax(-1),
        name=f"cm/{split}",
    )


def save_importance_plot(live, model, feature_names):
    fig, axes = plt.subplots(dpi=100)
    fig.subplots_adjust(bottom=0.2, top=0.95)

    axes.set_ylabel("Mean decrease in impurity")

    importances = model.feature_importances_
    forest_importances = pd.Series(importances, index=feature_names).nlargest(10)

    forest_importances.plot.bar(ax=axes)

    live.log_image("importance.png", fig)


def main():

    curr_dir = pathlib.Path(__file__)
    home_dir = curr_dir.parent.parent.parent

    input_file = sys.argv[1]

    data_path = home_dir / input_file
    model_file = home_dir / "models" / "model.joblib"

    output_path = home_dir / "dvclive"
    output_path.mkdir(parents=True, exist_ok=True)

    model = joblib.load(model_file)

    TARGET = "Class"

    train_features = pd.read_csv(data_path / "train.csv")
    X_train = train_features.drop(TARGET, axis=1)
    y_train = train_features[TARGET]
    feature_names = X_train.columns.to_list()

    test_features = pd.read_csv(data_path / "test.csv")
    X_test = test_features.drop(TARGET, axis=1)
    y_test = test_features[TARGET]

    with Live(output_path, dvcyaml=False) as live:
        evaluate(model, X_train, y_train, "train", live)
        evaluate(model, X_test, y_test, "test", live)

        save_importance_plot(live, model, feature_names)

    with Live(output_path, dvcyaml=False) as live:
        evaluate(model, X_train, y_train, "train", live)
        evaluate(model, X_test, y_test, "test", live)

        save_importance_plot(live, model, feature_names)


if __name__ == "__main__":
    main()