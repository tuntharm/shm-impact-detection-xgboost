from __future__ import annotations

import pandas as pd

from fyp_impact.data import direct_csv_files, load_feature_data, localisation_feature_columns, prepare_feature_data
from fyp_impact.geometry import DEFAULT_Z_MAX_CM, parse_location_set, top_location_mask


def test_direct_csv_files_does_not_recurse(tmp_path):
    (tmp_path / "nested").mkdir()
    (tmp_path / "root.csv").write_text("Loc,theta,z\n1,0,7.5\n", encoding="utf-8")
    (tmp_path / "nested" / "child.csv").write_text("Loc,theta,z\n2,0,15\n", encoding="utf-8")

    files = direct_csv_files([tmp_path])

    assert [path.name for path in files] == ["root.csv"]


def test_load_feature_data_keeps_source_file(tmp_path):
    (tmp_path / "a.csv").write_text("Loc,theta,z,ToA_S1,Impact_Type\n1,0,7.5,0.1,1\n", encoding="utf-8")

    df = load_feature_data([tmp_path])

    assert df.loc[0, "source_file"] == "a.csv"


def test_prepare_feature_data_adds_sensor_geometry_and_targets():
    df = pd.DataFrame({"Loc": [1], "theta": [0.0], "z": [7.5], "ToA_S1": [0.1], "Impact_Type": [1]})

    prepared = prepare_feature_data(df, z_max=DEFAULT_Z_MAX_CM)

    assert prepared.loc[0, "sin_theta"] == 0.0
    assert prepared.loc[0, "cos_theta"] == 1.0
    assert prepared.loc[0, "S1_theta"] == 0.0
    assert prepared.loc[0, "S8_z"] == DEFAULT_Z_MAX_CM


def test_localisation_feature_columns_exclude_targets_and_labels():
    df = pd.DataFrame(
        {
            "Loc": [1],
            "theta": [0.0],
            "z": [7.5],
            "ToA_S1": [0.1],
            "Amplitude_S1": [2.0],
            "SignalEnergy_S1": [3.0],
            "Force_N": [4.0],
            "Impact_Type": [1],
            "S1_theta": [0.0],
            "S1_z": [0.0],
            "sin_theta": [0.0],
            "cos_theta": [1.0],
        }
    )

    columns = localisation_feature_columns(df)

    assert "Impact_Type" not in columns
    assert "theta" not in columns
    assert {"ToA_S1", "Amplitude_S1", "SignalEnergy_S1", "Force_N", "S1_theta", "S1_z"} <= set(columns)


def test_parse_and_mask_top_locations():
    locations = parse_location_set("1-3,31,40")
    mask = top_location_mask(pd.Series([1, 4, 31, 40]), locations)

    assert locations == (1, 2, 3, 31, 40)
    assert mask.tolist() == [True, False, True, True]
