"""Deprecated compatibility wrappers for frontend data prep.

This module keeps older imports working while delegating to data_prep.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

from src.main.frontend import data_prep

__all__ = [
    "make_radar_scores",
    "update_layout_overview",
    "get_overview_table_data",
    "get_selected_table_data",
    "get_anchor_table_data",
    "prepare_map_data",
]


def make_radar_scores(results_df: pd.DataFrame, axes: List[str]) -> pd.DataFrame:
    return data_prep.make_radar_scores(results_df, axes)


def update_layout_overview(indices, forest_area_3, model_list, precomputed=None) -> dict:
    return data_prep.update_layout_overview(indices, forest_area_3, model_list, precomputed=precomputed)


def get_overview_table_data(forest_area_3, model_list, results_df) -> List[List[Any]]:
    return data_prep.get_overview_table_data(forest_area_3, model_list, results_df)


def get_selected_table_data(
    forest_area_3,
    model_list,
    results_df: pd.DataFrame,
    selected_index: Optional[int],
) -> List[List[Any]]:
    return data_prep.get_selected_table_data(
        forest_area_3,
        model_list,
        results_df,
        selected_index,
    )


def get_anchor_table_data(
    forest_area_3,
    model_list,
    results_df: pd.DataFrame,
    selected_index: Optional[int],
) -> List[List[str]]:
    return data_prep.get_anchor_table_data(
        forest_area_3,
        model_list,
        results_df,
        selected_index,
    )


def prepare_map_data(
    forest_area_3,
    results_df: pd.DataFrame,
    model_list=None,
) -> Dict[str, Any]:
    return data_prep.prepare_map_data(forest_area_3, results_df, model_list=model_list)
