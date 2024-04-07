"""This tests the CLI functionality of training and predicting a regression model on a multi-molecule.
"""

import pytest
import json

from chemprop.cli.main import main

pytestmark = pytest.mark.CLI


@pytest.fixture
def data_path(data_dir):
    return str(data_dir / "regression" / "mol+mol.csv")


@pytest.fixture
def model_path(data_dir):
    return str(data_dir / "example_model_v2_regression_mol+mol.pt")


def test_train_quick(monkeypatch, data_path):
    args = [
        "chemprop",
        "train",
        "-i",
        data_path,
        "--smiles-columns",
        "smiles",
        "solvent",
        "--epochs",
        "1",
        "--num-workers",
        "0",
    ]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()


def test_predict_quick(monkeypatch, data_path, model_path):
    args = [
        "chemprop",
        "predict",
        "-i",
        data_path,
        "--smiles-columns",
        "smiles",
        "solvent",
        "--model-path",
        model_path,
    ]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()


def test_train_output_structure(monkeypatch, data_path, tmp_path):
    args = [
        "chemprop",
        "train",
        "-i",
        data_path,
        "--smiles-columns",
        "smiles",
        "solvent",
        "--epochs",
        "1",
        "--num-workers",
        "0",
        "--save-dir",
        str(tmp_path),
        "--save-smiles-splits",
        "--save-preds",
    ]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()

    assert (tmp_path / "model_0" / "model.pt").exists()
    assert (tmp_path / "model_0" / "checkpoints" / "last.ckpt").exists()
    assert (tmp_path / "model_0" / "trainer_logs" / "version_0").exists()
    assert (tmp_path / "train_smiles.csv").exists()
    assert (tmp_path / "model_0" / "test_predictions.csv").exists()


def test_train_splits_file(monkeypatch, data_path, tmp_path):
    splits_file = str(tmp_path / "splits.json")
    splits = [
        {"train": [1, 2], "val": "3-5", "test": "6,7"},
        {"val": [1, 2], "test": "3-5", "train": "6,7"},
    ]

    with open(splits_file, "w") as f:
        json.dump(splits, f)

    args = [
        "chemprop",
        "train",
        "-i",
        data_path,
        "--smiles-columns",
        "smiles",
        "solvent",
        "--epochs",
        "1",
        "--num-workers",
        "0",
        "--save-dir",
        str(tmp_path),
        "--splits-file",
        splits_file,
    ]

    with monkeypatch.context() as m:
        m.setattr("sys.argv", args)
        main()
