from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple, Optional, Set
from shapely.geometry import Point

import numpy as np
import pandas as pd
import plotly.express as px
from plotly.colors import hex_to_rgb
from pyparsing import line

from src.main import geometry_operations

# ... [Previous helper functions _scale, _convert_hex_to_rgba, _safe_float remain unchanged] ...
def _scale(series: pd.Series, pad_low: float = 0.1) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    minv, maxv = s.min(skipna=True), s.max(skipna=True)
    rng = maxv - minv
    if pd.isna(rng) or rng == 0:
        return pd.Series(0.5, index=s.index, dtype="float64")
    min_adj = minv - pad_low * rng
    return ((s - min_adj) / (maxv - min_adj)).clip(0, 1)


def _convert_hex_to_rgba(hex_color: str, alpha: float = 1.0) -> str:
    r, g, b = hex_to_rgb(hex_color)
    return f"rgba({r}, {g}, {b}, {alpha:.3f})"


def _safe_float(x, default=0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)

def _extract_tree_metadata(tree_obj) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    """Return (x, y, BHD, height) for different pandas/dict representations."""

    def _clean(value):
        if value is None:
            return None
        try:
            if pd.isna(value):  # type: ignore[arg-type]
                return None
        except Exception:
            pass
        return _safe_float(value)

    if tree_obj is None:
        return None, None, None, None

    if isinstance(tree_obj, pd.Series):
        return (
            _clean(tree_obj.get("x")) if "x" in tree_obj else None,
            _clean(tree_obj.get("y")) if "y" in tree_obj else None,
            _clean(tree_obj.get("BHD")) if "BHD" in tree_obj else None,
            _clean(tree_obj.get("h")) if "h" in tree_obj else None,
        )

    if isinstance(tree_obj, pd.DataFrame):
        if tree_obj.empty:
            return None, None, None, None
        return _extract_tree_metadata(tree_obj.iloc[0])

    if isinstance(tree_obj, dict):
        return (
            _clean(tree_obj.get("x")) if "x" in tree_obj else None,
            _clean(tree_obj.get("y")) if "y" in tree_obj else None,
            _clean(tree_obj.get("BHD")) if "BHD" in tree_obj else None,
            _clean(tree_obj.get("h")) if "h" in tree_obj else None,
        )

    return None, None, None, None


def _sample_line_xy(line, min_points: int = 20, step: float = 5.0) -> Tuple[List[float], List[float]]:
    """
    Take a shapely LineString-like geometry and sample along it for plotting.
    This smooths our display of cable corridors.
    """
    length = float(line.length)
    n_points = max(int(length // step) + 2, min_points)
    dists = np.linspace(0.0, length, n_points)
    xs, ys = [], []
    for d in dists:
        p = line.interpolate(d)
        xs.append(float(p.x))
        ys.append(float(p.y))
    return xs, ys


# ... [Previous coloring helpers _labels_to_plotly_colors, _tree_colors_for_indices, etc. remain unchanged] ...
_PLOTLY = px.colors.qualitative.Plotly

def _labels_to_plotly_colors(labels: List[int]) -> List[str]:
    out = []
    n = len(_PLOTLY)
    for lab in labels:
        try:
            out.append(_PLOTLY[int(lab) % n])
        except Exception:
            out.append("green")
    return out


def _colors_for_dtl(
    indices: List[int],
    line_index,
    dtl_full: np.ndarray,
    num_points: int,
) -> List[str]:
    if num_points <= 0:
        return []
    if not indices:
        return ["green"] * num_points
    pos_map: Dict[int, int] = {int(k): i for i, k in enumerate(line_index)}
    valid_cols: List[int] = []
    for ridx in indices:
        ridx_int = int(ridx)
        if ridx_int in pos_map:
            valid_cols.append(pos_map[ridx_int])
    if not valid_cols:
        return ["green"] * num_points
    dtl_slice = dtl_full[:, valid_cols]
    try:
        labels = np.argmin(dtl_slice, axis=1).astype(int).tolist()
    except Exception:
        return ["green"] * num_points
    return _labels_to_plotly_colors(labels)


def _tree_colors_for_indices(indices: List[int], forest_area_3, dtl_full: np.ndarray) -> List[str]:
    num_trees = len(forest_area_3.harvesteable_trees_gdf)
    return _colors_for_dtl(indices, forest_area_3.line_gdf.index, dtl_full, num_trees)

def _labels_to_real_indices(sel_real: List[int], labels: List[int]) -> List[Optional[int]]:
    if not sel_real or labels is None:
        return [None] * len(labels)
    out: List[Optional[int]] = []
    n = len(sel_real)
    for lbl in labels:
        try:
            il = int(lbl)
            if 0 <= il < n:
                out.append(int(sel_real[il]))
            else:
                out.append(None)
        except Exception:
            out.append(None)
    return out

# ... [update_layout_overview, VizData, and wrappers remain unchanged] ...
def update_layout_overview(indices, forest_area_3, model_list, precomputed=None) -> dict:
    line_gdf = forest_area_3.line_gdf
    full_index = line_gdf.index
    pos_map: Dict[int, int] = {int(k): i for i, k in enumerate(full_index)}
    sel_real: List[int] = []
    for i in indices:
        ii = int(i)
        if ii in pos_map:
            sel_real.append(ii)
    if not sel_real:
        return {
            "Wood Volume per Cable Corridor (m3)": [],
            "Total Cable Corridor Costs (€)": 0,
            "Setup and Takedown Costs (€)": 0,
            "Productivity Costs (€)": 0,
            "Setup and Takedown, Prod. Costs (€)": "0 / 0",
            "Ecol. Penalty": 0,
            "Ergon. Penalty": 0,
            "Tree to Cable Corridor Assignment": [0] * len(forest_area_3.harvesteable_trees_gdf),
            "Supports Height (m)": [],
            "Supports Amount": [],
            "Max lateral Yarding Distance (m)": 0,
            "Average lateral Yarding Distance (m)": 0,
            "Cost per m3 (€)": 0.0,
            "Average Tree Height (m)": [],
            "Volume per Meter (m3/m)": 0.0,
            "Max Yarding Distance per Cable Corridor (m)": [],
            "Average Yarding Distance per Cable Corridor (m)": [],
            "Anchor height": [],
            "Anchor BHD": [],
            "Anchor max holding force": [],
            "Anchor x coordinate": [],
            "Anchor y coordinate": [],
            "Corresponding Cable Corridor": [],
            "Road Anchor height": [],
            "Road Anchor BHD": [],
            "Road Anchor max holding force": [],
            "Road Anchor x coordinate": [],
            "Road Anchor y coordinate": [],
            "Road Anchor Angle of Attack": [],
            "Tail Anchor Angle of Attack": [],
        }
    rot_line_gdf = line_gdf.loc[sel_real]
    if precomputed is not None:
        dtl_full, dcs_full = precomputed
        cols = [pos_map[i] for i in sel_real]
        distance_tree_line = dtl_full[:, cols]
        distance_carriage_support = dcs_full[:, cols]
    else:
        distance_tree_line, distance_carriage_support = geometry_operations.compute_distances_facilities_clients(
            forest_area_3.harvesteable_trees_gdf,
            rot_line_gdf
        )
    try:
        tree_to_line_assignment = np.argmin(distance_tree_line, axis=1)
        distance_trees_to_selected_lines = distance_tree_line[
            range(distance_tree_line.shape[0]), tree_to_line_assignment
        ]
    except Exception:
        tree_to_line_assignment = np.zeros((len(forest_area_3.harvesteable_trees_gdf),), dtype=int)
        distance_trees_to_selected_lines = np.zeros_like(tree_to_line_assignment, dtype=float)
    if model_list is not None and hasattr(model_list[0], "productivity_cost"):
        prod = model_list[0].productivity_cost
        sel_cols = [pos_map[i] for i in sel_real]
        selected_prod_cost = prod[:, sel_cols]
    else:
        selected_prod_cost = np.zeros((len(forest_area_3.harvesteable_trees_gdf), len(sel_real)))
    productivity_cost_overall = 0
    for index, val in enumerate(tree_to_line_assignment):
        col = min(int(val), selected_prod_cost.shape[1] - 1)
        productivity_cost_overall += selected_prod_cost[index][col]
    grouped_class_indices = [
        np.nonzero(tree_to_line_assignment == label)[0]
        for label in range(max(1, len(sel_real)))
    ]
    gtrees = forest_area_3.harvesteable_trees_gdf
    wood_volume_per_cr = [
        int(sum(gtrees.iloc[g]["cubic_volume"])) if len(g) else 0
        for g in grouped_class_indices
    ][: len(sel_real)]
    average_tree_size_per_cr = [
        round(float(sum(gtrees.iloc[g]["h"])) / len(g), 2) if len(g) else 0.0
        for g in grouped_class_indices
    ][: len(sel_real)]
    supports_height = [
        (
            [segment.start_support.attachment_height for segment in cr_object.supported_segments[1:]]
            if cr_object.supported_segments else []
        )
        for cr_object in rot_line_gdf["Cable Road Object"]
    ]
    supports_amount = [len(heights) for heights in supports_height]
    max_yarding_distance_per_cr, average_yarding_distance_per_cr = [], []
    for line_idx, g in enumerate(grouped_class_indices[: len(sel_real)]):
        if len(g) == 0:
            max_yarding_distance_per_cr.append(0)
            average_yarding_distance_per_cr.append(0)
        else:
            dists = distance_carriage_support[g, line_idx]
            max_yarding_distance_per_cr.append(int(max(dists)))
            average_yarding_distance_per_cr.append(int(np.mean(dists)))
    endmast_height_list, endmast_BHD_list, endmast_max_holding_force_list = [], [], []
    endmast_x_list, endmast_y_list = [], []
    for _, row in rot_line_gdf.iterrows():
        end_tree = getattr(row, "end_support_tree", getattr(row, "end_anchor_tree", None))
        end_pt = row.geometry.coords[-1]
        ex, ey = float(end_pt[0]), float(end_pt[1])
        eh = eb = emf = 0
        if isinstance(end_tree, dict):
            eh = int(end_tree.get("h", 0))
            eb = int(end_tree.get("BHD", 0))
            emf = int(end_tree.get("max_holding_force", 0))
        cr_object = row.get("Cable Road Object")
        end_support = getattr(cr_object, "end_support", None) if cr_object is not None else None
        support_attach_h = getattr(end_support, "attachment_height", None)
        if support_attach_h is not None and eh <= 1:
            eh = int(support_attach_h)
        endmast_height_list.append(eh)
        endmast_BHD_list.append(eb)
        endmast_max_holding_force_list.append(emf)
        endmast_x_list.append(round(ex, 2))
        endmast_y_list.append(round(ey, 2))
    road_anchor_height_list, road_anchor_BHD_list = [], []
    road_anchor_max_holding_force_list, road_anchor_x_list, road_anchor_y_list = [], [], []
    for _, row in rot_line_gdf.iterrows():
        ra = getattr(row, "road_anchor_tree_series", None)
        if isinstance(ra, dict):
            road_anchor_height_list.append(int(ra.get("h", 0)))
            road_anchor_BHD_list.append(int(ra.get("BHD", 0)))
            road_anchor_max_holding_force_list.append(int(ra.get("max_holding_force", 0)))
            road_anchor_x_list.append(round(_safe_float(ra.get("x", 0.0)), 2))
            road_anchor_y_list.append(round(_safe_float(ra.get("y", 0.0)), 2))
            continue
        if hasattr(ra, "iterrows"):
            try:
                first = next(ra.iterrows())[1]
                road_anchor_height_list.append(int(first.get("h", 0)))
                road_anchor_BHD_list.append(int(first.get("BHD", 0)))
                road_anchor_max_holding_force_list.append(int(first.get("max_holding_force", 0)))
                road_anchor_x_list.append(round(_safe_float(first.get("x", 0.0)), 2))
                road_anchor_y_list.append(round(_safe_float(first.get("y", 0.0)), 2))
            except StopIteration:
                road_anchor_height_list.append(0)
                road_anchor_BHD_list.append(0)
                road_anchor_max_holding_force_list.append(0)
                road_anchor_x_list.append(0.0)
                road_anchor_y_list.append(0.0)
            continue
        road_anchor_height_list.append(0)
        road_anchor_BHD_list.append(0)
        road_anchor_max_holding_force_list.append(0)
        road_anchor_x_list.append(0.0)
        road_anchor_y_list.append(0.0)
    if len(tree_to_line_assignment) > 0 and selected_prod_cost.shape[1] > 0:
        assigned_yarding_distances = distance_carriage_support[
            range(distance_carriage_support.shape[0]), tree_to_line_assignment
        ]
        max_yarding_distance = round(float(np.max(assigned_yarding_distances)), 1)
        average_yarding_distance = round(float(np.mean(assigned_yarding_distances)), 1)
    else:
        max_yarding_distance = 0
        average_yarding_distance = 0
    line_cost_total = int(sum(rot_line_gdf["line_cost"])) if len(rot_line_gdf) else 0
    total_cable_road_costs = int(line_cost_total + productivity_cost_overall)
    denom = max(1, sum(wood_volume_per_cr) if len(wood_volume_per_cr) else 1)
    cost_per_m3 = round(total_cable_road_costs / denom, 2)
    if len(sel_real) > 0:
        threshold_eco = 10
        eco_penalty_lateral = np.where(
            distance_tree_line > threshold_eco,
            distance_tree_line - threshold_eco,
            0,
        )
        sum_eco_distances = int(
            sum(eco_penalty_lateral[j][i] for i, j in zip(tree_to_line_assignment, range(len(eco_penalty_lateral))))
        )
    else:
        sum_eco_distances = 0
    if len(sel_real) > 0:
        threshold_ergo = 15
        ergo_penalty_lateral = np.where(
            distance_tree_line > threshold_ergo,
            (distance_tree_line - threshold_ergo) * 2,
            0,
        )
        sum_ergo_distances = int(
            sum(ergo_penalty_lateral[j][i] for i, j in zip(tree_to_line_assignment, range(len(ergo_penalty_lateral))))
        )
    else:
        sum_ergo_distances = 0
    total_len = float(sum(rot_line_gdf["line_length"])) if len(rot_line_gdf) else 0.0
    volume_per_meter = round((sum(wood_volume_per_cr) / total_len) if total_len else 0.0, 2)
    return {
        "Wood Volume per Cable Corridor (m3)": wood_volume_per_cr,
        "Total Cable Corridor Costs (€)": total_cable_road_costs,
        "Setup and Takedown Costs (€)": line_cost_total,
        "Productivity Costs (€)": int(productivity_cost_overall),
        "Setup and Takedown, Prod. Costs (€)": f"{line_cost_total} / {int(productivity_cost_overall)}",
        "Ecol. Penalty": sum_eco_distances,
        "Ergon. Penalty": sum_ergo_distances,
        "Tree to Cable Corridor Assignment": tree_to_line_assignment
        if len(sel_real) > 0
        else [0] * len(forest_area_3.harvesteable_trees_gdf),
        "Supports Height (m)": supports_height,
        "Supports Amount": supports_amount,
        "Max lateral Yarding Distance (m)": max_yarding_distance,
        "Average lateral Yarding Distance (m)": average_yarding_distance,
        "Cost per m3 (€)": cost_per_m3,
        "Average Tree Height (m)": average_tree_size_per_cr,
        "Volume per Meter (m3/m)": volume_per_meter,
        "Max Yarding Distance per Cable Corridor (m)": max_yarding_distance_per_cr,
        "Average Yarding Distance per Cable Corridor (m)": average_yarding_distance_per_cr,
        "Anchor height": endmast_height_list,
        "Anchor BHD": endmast_BHD_list,
        "Anchor max holding force": endmast_max_holding_force_list,
        "Anchor x coordinate": endmast_x_list,
        "Anchor y coordinate": endmast_y_list,
        "Corresponding Cable Corridor": sel_real,
        "Road Anchor height": road_anchor_height_list,
        "Road Anchor BHD": road_anchor_BHD_list,
        "Road Anchor max holding force": road_anchor_max_holding_force_list,
        "Road Anchor x coordinate": road_anchor_x_list,
        "Road Anchor y coordinate": road_anchor_y_list,
        "Road Anchor Angle of Attack": rot_line_gdf["angle_between_start_support_and_cr"] if len(rot_line_gdf) else [],
        "Tail Anchor Angle of Attack": rot_line_gdf["angle_between_end_support_and_cr"] if len(rot_line_gdf) else [],
    }

@dataclass
class VizData:
    forest_area_3: Any
    model_list: Any
    results_df: pd.DataFrame
    indices_to_show: List[int] = field(init=False)
    display_to_real: Dict[int, int] = field(init=False)
    real_to_display: Dict[int, int] = field(init=False)
    dtl_full: np.ndarray = field(init=False)
    dcs_full: np.ndarray = field(init=False)
    palette: List[str] = field(init=False)
    layout_union: Dict[str, Any] = field(init=False)
    layout_by_model: Dict[int, Dict[str, Any]] = field(init=False)
    map: Dict[str, Any] = field(init=False)
    overview_rows: List[List[Any]] = field(init=False)
    def __post_init__(self):
        self._build_core()
        self._precompute_all_layouts()
        self._build_map_payload()
        self._build_overview_rows()
    def _build_core(self) -> None:
        valid_line_ids = set(map(int, self.forest_area_3.line_gdf.index))
        flat_ids: List[int] = []
        for row in self.results_df["selected_lines"]:
            for ridx in row:
                ii = int(ridx)
                if ii in valid_line_ids:
                    flat_ids.append(ii)
        self.indices_to_show = sorted(set(flat_ids))
        display_names = list(range(1, len(self.indices_to_show) + 1))
        self.display_to_real = dict(zip(display_names, self.indices_to_show))
        self.real_to_display = dict(zip(self.indices_to_show, display_names))
        self.dtl_full, self.dcs_full = geometry_operations.compute_distances_facilities_clients(
            self.forest_area_3.harvesteable_trees_gdf,
            self.forest_area_3.line_gdf,
        )
        self.palette = list(px.colors.qualitative.Plotly)
    def _precompute_all_layouts(self) -> None:
        self.layout_by_model = {}
        if self.indices_to_show:
            self.layout_union = update_layout_overview(
                self.indices_to_show,
                self.forest_area_3,
                self.model_list,
                precomputed=(self.dtl_full, self.dcs_full),
            )
        else:
            self.layout_union = {}
        valid_line_ids = set(map(int, self.forest_area_3.line_gdf.index))
        for i, res in self.results_df.iterrows():
            sel_real = [int(x) for x in res["selected_lines"] if int(x) in valid_line_ids]
            self.layout_by_model[i] = update_layout_overview(
                sel_real,
                self.forest_area_3,
                self.model_list,
                precomputed=(self.dtl_full, self.dcs_full),
            )
    def _compute_fixed_volumes_for_map(self) -> Dict[int, float]:
        line = self.forest_area_3.line_gdf
        for col in ("wood_volume", "volume_m3", "volumen_m3", "volume"):
            if col in line.columns:
                return {
                    int(i): _safe_float(line.loc[int(i), col])
                    for i in self.indices_to_show
                }
        if self.indices_to_show and self.layout_union:
            sel_real = list(self.layout_union.get("Corresponding Cable Corridor", self.indices_to_show))
            per_cr = self.layout_union.get("Wood Volume per Cable Corridor (m3)", [])
            return {
                int(r): _safe_float(v)
                for r, v in zip(sel_real, per_cr)
            }
        return {int(i): 0.0 for i in self.indices_to_show}
    def _build_map_payload(self) -> None:
        fa = self.forest_area_3
        idx_all = self.indices_to_show
        gtrees = fa.harvesteable_trees_gdf
        tree_x = [float(geom.x) for geom in gtrees.geometry]
        tree_y = [float(geom.y) for geom in gtrees.geometry]
        tree_coords = None
        if tree_x and tree_y:
            tree_coords = np.column_stack((tree_x, tree_y))
        bhd_series = gtrees.get("BHD", pd.Series([None] * len(gtrees)))
        tree_bhd_cm = [None if pd.isna(b) else float(b) for b in bhd_series]
        street_anchor_x: List[float] = []
        street_anchor_y: List[float] = []
        street_anchor_bhd_cm: List[Optional[float]] = []
        anchor_gdf = getattr(fa, "anchor_trees_gdf", None)
        if anchor_gdf is not None:
            try:
                if hasattr(anchor_gdf, "geometry"):
                    street_anchor_x = [float(geom.x) for geom in anchor_gdf.geometry]
                    street_anchor_y = [float(geom.y) for geom in anchor_gdf.geometry]
                elif hasattr(anchor_gdf, "columns") and "x" in anchor_gdf and "y" in anchor_gdf:
                    street_anchor_x = [float(v) for v in anchor_gdf["x"]]
                    street_anchor_y = [float(v) for v in anchor_gdf["y"]]
                elif isinstance(anchor_gdf, (list, tuple)):
                    for entry in anchor_gdf:
                        ax, ay, abhd, _ = _extract_tree_metadata(entry)
                        if ax is None or ay is None:
                            continue
                        street_anchor_x.append(float(ax))
                        street_anchor_y.append(float(ay))
                        street_anchor_bhd_cm.append(abhd)
            except Exception:
                street_anchor_x = []
                street_anchor_y = []
        if street_anchor_x and not street_anchor_bhd_cm:
            if hasattr(anchor_gdf, "get"):
                bhd_series = anchor_gdf.get("BHD", pd.Series([None] * len(street_anchor_x)))
                street_anchor_bhd_cm = [
                    None if pd.isna(b) else float(b)
                    for b in bhd_series
                ]
            else:
                street_anchor_bhd_cm = [None] * len(street_anchor_x)
        street_anchor_color_default = ["green"] * len(street_anchor_x)
        street_anchor_colors_by_union = street_anchor_color_default
        street_anchor_colors_by_model: Dict[int, List[str]] = {}
        street_anchor_colors_by_selection: Dict[Tuple[int, ...], List[str]] = {}
        anchor_dtl_full: Optional[np.ndarray] = None
        if anchor_gdf is not None and hasattr(anchor_gdf, "geometry") and street_anchor_x:
            try:
                anchor_dtl_full, _ = geometry_operations.compute_distances_facilities_clients(
                    anchor_gdf,
                    fa.line_gdf,
                )
            except Exception:
                anchor_dtl_full = None
        if anchor_dtl_full is not None and len(street_anchor_x) == anchor_dtl_full.shape[0]:
            street_anchor_colors_by_union = _colors_for_dtl(
                self.indices_to_show,
                fa.line_gdf.index,
                anchor_dtl_full,
                len(street_anchor_x),
            )
            for i, res in self.results_df.iterrows():
                sel_real_all = [int(x) for x in res["selected_lines"]]
                colors_i = _colors_for_dtl(
                    sel_real_all,
                    fa.line_gdf.index,
                    anchor_dtl_full,
                    len(street_anchor_x),
                )
                street_anchor_colors_by_model[int(i)] = colors_i
                street_anchor_colors_by_selection[tuple(sel_real_all)] = colors_i
        volumes_by_idx = self._compute_fixed_volumes_for_map()
        corridors: Dict[int, Dict[str, Any]] = {}
        subset = fa.line_gdf.loc[idx_all] if len(idx_all) else fa.line_gdf.iloc[[]]
        display_lookup = {
            int(real): int(self.real_to_display.get(int(real), int(real)))
            for real in self.indices_to_show
        }
        support_coords: Set[Tuple[float, float]] = set()
        support_indices_by_corridor: Dict[int, Set[int]] = defaultdict(set)
        for real_idx, row in subset.iterrows():
            line = row.geometry
            xs, ys = _sample_line_xy(line)
            start_pt, end_pt = line.coords[0], line.coords[-1]
            length_m = _safe_float(row.get("line_length", 0.0))
            volume_m3 = _safe_float(volumes_by_idx.get(int(real_idx), 0.0))
            display_id = display_lookup.get(int(real_idx), int(real_idx))
            tail_tree = getattr(row, "end_anchor_tree", None)
            tail_x, tail_y, tail_bhd, tail_h = _extract_tree_metadata(tail_tree)
            if tail_x is None or tail_y is None:
                tail_x, tail_y = float(end_pt[0]), float(end_pt[1])
            endmast_tree = getattr(row, "end_support_tree", None)
            if endmast_tree is None:
                endmast_tree = tail_tree
            endmast_x, endmast_y, endmast_bhd, endmast_h = _extract_tree_metadata(endmast_tree)
            if endmast_x is None or endmast_y is None:
                endmast_x, endmast_y = float(end_pt[0]), float(end_pt[1])
            road_anchors_list: List[dict] = []
            ra_src = getattr(row, "road_anchor_tree_series", None)
            if isinstance(ra_src, pd.DataFrame) and not ra_src.empty:
                for _, r in ra_src.iterrows():
                    road_anchors_list.append(
                        dict(
                            x=_safe_float(r.get("x", 0)),
                            y=_safe_float(r.get("y", 0)),
                            BHD=_safe_float(r.get("BHD", 0)),
                        )
                    )
            elif isinstance(ra_src, dict) and "features" in ra_src:
                for feat in ra_src["features"]:
                    props = feat.get("properties", {})
                    road_anchors_list.append(
                        dict(
                            x=_safe_float(props.get("x", 0)),
                            y=_safe_float(props.get("y", 0)),
                            BHD=_safe_float(props.get("BHD", 0)),
                        )
                    )
            elif isinstance(ra_src, dict):
                road_anchors_list.append(
                    dict(
                        x=_safe_float(ra_src.get("x", 0)),
                        y=_safe_float(ra_src.get("y", 0)),
                        BHD=_safe_float(ra_src.get("BHD", 0)),
                    )
                )
            tail_anchor_info = dict(x=tail_x, y=tail_y)
            if tail_bhd is not None:
                tail_anchor_info["BHD"] = tail_bhd
            if tail_h is not None:
                tail_anchor_info["h"] = tail_h
            endmast_info = dict(x=endmast_x, y=endmast_y)
            if endmast_bhd is not None:
                endmast_info["BHD"] = endmast_bhd
            if endmast_h is not None:
                endmast_info["h"] = endmast_h
            corridors[int(real_idx)] = dict(
                xs=xs,
                ys=ys,
                start=(float(start_pt[0]), float(start_pt[1])),
                end=(float(end_pt[0]), float(end_pt[1])),
                tail_anchor=tail_anchor_info,
                endmast=endmast_info,
                road_anchors=road_anchors_list,
                length_m=length_m,
                volume_m3=volume_m3,
                display_id=display_id,
            )
            cr_object = row.get("Cable Road Object")
            segments = getattr(cr_object, "supported_segments", None)
            if not segments:
                continue
            for segment in segments[1:]:
                support = getattr(segment, "start_support", None)
                xy_location = getattr(support, "xy_location", None)
                if xy_location is None:
                    continue
                x_val = round(float(xy_location.x), 4)
                y_val = round(float(xy_location.y), 4)
                coord_key = (x_val, y_val)
                support_coords.add(coord_key)
                if tree_coords is None:
                    continue
                diffs = tree_coords - np.array([x_val, y_val])
                dist_sq = np.sum(diffs ** 2, axis=1)
                nearest_idx = int(np.argmin(dist_sq))
                if dist_sq[nearest_idx] <= 4.0:
                    support_indices_by_corridor[int(real_idx)].add(nearest_idx)
        x_vals = []
        y_vals = []
        for geom in fa.line_gdf.geometry:
            try:
                xs_geom, ys_geom = geom.xy
                x_vals.extend(xs_geom)
                y_vals.extend(ys_geom)
            except Exception:
                pass
        if hasattr(fa.line_gdf, "end_support_tree"):
            for tail in fa.line_gdf.end_support_tree:
                if isinstance(tail, pd.DataFrame) and not tail.empty:
                    x_vals.extend(list(tail["x"].astype(float)))
                    y_vals.extend(list(tail["y"].astype(float)))
                elif isinstance(tail, dict) and "features" in tail:
                    for f in tail["features"]:
                        props = f.get("properties", f)
                        x_vals.append(float(props.get("x", 0)))
                        y_vals.append(float(props.get("y", 0)))
                elif isinstance(tail, dict):
                    if "x" in tail and "y" in tail:
                        x_vals.append(float(tail["x"]))
                        y_vals.append(float(tail["y"]))
        if hasattr(fa.line_gdf, "end_anchor_tree"):
            for tail in fa.line_gdf.end_anchor_tree:
                if isinstance(tail, pd.DataFrame) and not tail.empty:
                    x_vals.extend(list(tail["x"].astype(float)))
                    y_vals.extend(list(tail["y"].astype(float)))
                elif isinstance(tail, dict) and "features" in tail:
                    for f in tail["features"]:
                        props = f.get("properties", f)
                        x_vals.append(float(props.get("x", 0)))
                        y_vals.append(float(props.get("y", 0)))
                elif isinstance(tail, dict):
                    if "x" in tail and "y" in tail:
                        x_vals.append(float(tail["x"]))
                        y_vals.append(float(tail["y"]))
        if hasattr(fa.line_gdf, "road_anchor_tree_series"):
            for ra in fa.line_gdf.road_anchor_tree_series:
                if isinstance(ra, pd.DataFrame) and not ra.empty:
                    x_vals.extend(list(ra["x"].astype(float)))
                    y_vals.extend(list(ra["y"].astype(float)))
                elif isinstance(ra, dict) and "features" in ra:
                    for f in ra["features"]:
                        props = f.get("properties", f)
                        x_vals.append(float(props.get("x", 0)))
                        y_vals.append(float(props.get("y", 0)))
                elif isinstance(ra, dict):
                    if "x" in ra and "y" in ra:
                        x_vals.append(float(ra["x"]))
                        y_vals.append(float(ra["y"]))
        if street_anchor_x and street_anchor_y:
            x_vals.extend(street_anchor_x)
            y_vals.extend(street_anchor_y)
        if x_vals and y_vals:
            pad = 10.0
            minx, maxx = min(x_vals), max(x_vals)
            miny, maxy = min(y_vals), max(y_vals)
            x_range = (minx - pad, maxx + pad)
            y_range = (miny - pad, maxy + pad)
        else:
            if tree_x and tree_y:
                pad = 10.0
                minx, maxx = min(tree_x), max(tree_x)
                miny, maxy = min(tree_y), max(tree_y)
                x_range = (minx - pad, maxx + pad)
                y_range = (miny - pad, maxy + pad)
            else:
                x_range = (-10, 10)
                y_range = (-10, 10)
        tree_color_default = ["green"] * len(tree_x)
        if self.indices_to_show:
            tree_colors_by_union = _tree_colors_for_indices(
                self.indices_to_show,
                self.forest_area_3,
                self.dtl_full,
            )
        else:
            tree_colors_by_union = tree_color_default
        tree_colors_by_model: Dict[int, List[str]] = {}
        tree_colors_by_selection: Dict[Tuple[int, ...], List[str]] = {}
        for i, res in self.results_df.iterrows():
            sel_real_all = [int(x) for x in res["selected_lines"]]
            colors_i = _tree_colors_for_indices(sel_real_all, self.forest_area_3, self.dtl_full)
            tree_colors_by_model[int(i)] = colors_i
            tree_colors_by_selection[tuple(sel_real_all)] = colors_i
        color_map = {
            rid: self.palette[j % len(self.palette)]
            for j, rid in enumerate(self.indices_to_show)
        }
        if tree_x and tree_y and support_coords:
            support_tree_indices: Set[int] = set()
            for sx, sy in support_coords:
                diffs = tree_coords - np.array([sx, sy])
                dist_sq = np.sum(diffs ** 2, axis=1)
                nearest_idx = int(np.argmin(dist_sq))
                if dist_sq[nearest_idx] <= 4.0:
                    support_tree_indices.add(nearest_idx)
            support_tree_mask = [i in support_tree_indices for i in range(len(tree_x))]
        else:
            support_tree_mask = [False] * len(tree_x)
        support_trees_by_corridor = {
            int(rid): sorted(list(idxs))
            for rid, idxs in support_indices_by_corridor.items()
            if idxs
        }
        self.map = dict(
            tree_x=tree_x,
            tree_y=tree_y,
            tree_bhd_cm=tree_bhd_cm,
            street_anchor_x=street_anchor_x,
            street_anchor_y=street_anchor_y,
            street_anchor_bhd_cm=street_anchor_bhd_cm,
            street_anchor_color_default=street_anchor_color_default,
            street_anchor_colors_by_union=street_anchor_colors_by_union,
            street_anchor_colors_by_model=street_anchor_colors_by_model,
            street_anchor_colors_by_selection=street_anchor_colors_by_selection,
            support_tree_mask=support_tree_mask,
            support_trees_by_corridor=support_trees_by_corridor,
            tree_color_default=tree_color_default,
            tree_colors_by_union=tree_colors_by_union,
            tree_colors_by_model=tree_colors_by_model,
            tree_colors_by_selection=tree_colors_by_selection,
            corridors=corridors,
            color_map=color_map,
            palette=self.palette,
            display_lookup=display_lookup,
            x_range=x_range,
            y_range=y_range,
            indices_to_show=list(self.indices_to_show),
        )
    def _build_overview_rows(self) -> None:
        rows: List[List[Any]] = []
        for i, res in self.results_df.iterrows():
            sel_real = [int(x) for x in res["selected_lines"] if int(x) in self.forest_area_3.line_gdf.index]
            layout = self.layout_by_model[i]
            rows.append([
                i + 1,
                layout.get("Total Cable Corridor Costs (€)"),
                layout.get("Setup and Takedown Costs (€)"),
                layout.get("Ecol. Penalty"),
                layout.get("Ergon. Penalty"),
                str([self.real_to_display.get(int(idx), int(idx)) for idx in sel_real])[1:-1],
                layout.get("Max lateral Yarding Distance (m)"),
                layout.get("Average lateral Yarding Distance (m)"),
                (
                    round(float(np.mean(layout["Supports Amount"])), 2)
                    if layout.get("Supports Amount")
                    else 0.0
                ),
                layout.get("Cost per m3 (€)"),
                layout.get("Volume per Meter (m3/m)"),
            ])
        self.overview_rows = rows
    def selected_rows(self, selected_index: int) -> List[List[str]]:
        if selected_index < 0 or selected_index >= len(self.results_df):
            return []
        valid_line_ids = set(map(int, self.forest_area_3.line_gdf.index))
        sel_real = [int(x) for x in self.results_df.iloc[selected_index]["selected_lines"]
                    if int(x) in valid_line_ids]
        layout = self.layout_by_model[selected_index]
        vols = layout.get("Wood Volume per Cable Corridor (m3)", [])
        sup_count = layout.get("Supports Amount", [])
        sup_heights = layout.get("Supports Height (m)", [])
        avg_tree_h = layout.get("Average Tree Height (m)", [])
        max_yard = layout.get("Max Yarding Distance per Cable Corridor (m)", [])
        avg_yard = layout.get("Average Yarding Distance per Cable Corridor (m)", [])
        fa = self.forest_area_3
        subset = fa.line_gdf.loc[fa.line_gdf.index.isin(sel_real)].loc[sel_real]
        rows: List[List[str]] = []
        for i, real_idx in enumerate(sel_real):
            disp_id = self.real_to_display.get(int(real_idx), int(real_idx))
            line_cost = int(subset.loc[real_idx, "line_cost"]) if "line_cost" in subset.columns else 0
            line_length = int(subset.loc[real_idx, "line_length"]) if "line_length" in subset.columns else 0
            vol = int(vols[i]) if i < len(vols) else 0
            s_cnt = int(sup_count[i]) if i < len(sup_count) else 0
            s_hlst = sup_heights[i] if i < len(sup_heights) and isinstance(sup_heights[i], list) else []
            s_hstr = "/" if not s_hlst else ", ".join(str(int(h)) for h in s_hlst)
            avg_h = float(avg_tree_h[i]) if i < len(avg_tree_h) else 0.0
            max_y = int(max_yard[i]) if i < len(max_yard) else 0
            avg_y = int(avg_yard[i]) if i < len(avg_yard) else 0
            rows.append([
                str(disp_id),
                str(line_cost),
                str(line_length),
                str(vol),
                str(s_cnt),
                s_hstr,
                f"{avg_h:.2f}",
                str(max_y),
                str(avg_y),
            ])
        return rows
    def anchor_rows(self, selected_index: int) -> List[List[str]]:
        import pandas as pd
        if selected_index < 0 or selected_index >= len(self.results_df):
            return []
        valid_line_ids = set(map(int, self.forest_area_3.line_gdf.index))
        sel_real = [
            int(x) for x in self.results_df.iloc[selected_index]["selected_lines"]
            if int(x) in valid_line_ids
        ]
        fa = self.forest_area_3
        subset = fa.line_gdf.loc[fa.line_gdf.index.isin(sel_real)].loc[sel_real]
        out_rows: List[List[str]] = []
        for real_idx, row in subset.iterrows():
            disp_id = self.real_to_display.get(int(real_idx), int(real_idx))
            ta = getattr(row, "end_support_tree", None)
            if ta is None:
                ta = getattr(row, "end_anchor_tree", None)
            cr_object = subset.loc[real_idx, "Cable Road Object"] if "Cable Road Object" in subset.columns else None
            end_support = getattr(cr_object, "end_support", None) if cr_object is not None else None
            support_attach_h = getattr(end_support, "attachment_height", None)
            bhd = h = x = y = None
            if isinstance(ta, pd.Series):
                bhd = ta.get("BHD", None)
                h   = ta.get("h",   None)
                x   = ta.get("x",   None)
                y   = ta.get("y",   None)
            elif isinstance(ta, pd.DataFrame) and not ta.empty:
                first = ta.iloc[0]
                bhd = first.get("BHD", None)
                h   = first.get("h",   None)
                x   = first.get("x",   None)
                y   = first.get("y",   None)
            elif isinstance(ta, dict):
                if "features" in ta:
                    try:
                        props = ta["features"][0].get("properties", ta["features"][0])
                    except Exception:
                        props = {}
                    bhd = props.get("BHD", None)
                    h   = props.get("h",   None)
                    x   = props.get("x",   None)
                    y   = props.get("y",   None)
                else:
                    bhd = ta.get("BHD", None)
                    h   = ta.get("h",   None)
                    x   = ta.get("x",   None)
                    y   = ta.get("y",   None)
            def _to_int(v):
                try:
                    return int(v)
                except Exception:
                    return None
            def _to_coord(v):
                try:
                    return round(float(v), 2)
                except Exception:
                    return None
            bhd_val = _to_int(bhd)
            h_val   = _to_int(h)
            if support_attach_h is not None and (h_val is None or h_val <= 1):
                h_val = int(support_attach_h)
            x_val   = _to_coord(x)
            y_val   = _to_coord(y)
            out_rows.append([
                str(disp_id),
                "" if bhd_val is None else str(bhd_val),
                "" if h_val   is None else str(h_val),
                "" if x_val   is None else str(x_val),
                "" if y_val   is None else str(y_val),
            ])
        return out_rows
    def make_radar_scores(self, axes: List[str]) -> pd.DataFrame:
        df = self.results_df.copy()
        eco = _scale(df["ecological_distances_RNI"])
        ergo = _scale(df["ergonomics_distances_RNI"])
        cost = _scale(df["cost_objective_RNI"])
        scores = pd.DataFrame({
            "Name": [f"{i+1}" for i in df.index],
            "Ökologische Optimierung": eco,
            "Ergonomische Optimierung": ergo,
            "Kosten Optimierung": cost,
        }, index=df.index)
        colors = [
            px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)]
            for i, _ in enumerate(scores.index)
        ]
        scores["color"] = [_convert_hex_to_rgba(c) for c in colors]
        scores["fill_color"] = [_convert_hex_to_rgba(c, 0.18) for c in colors]
        scores["raw_eco"] = self.results_df.loc[df.index, "ecological_distances_RNI"]
        scores["raw_ergo"] = self.results_df.loc[df.index, "ergonomics_distances_RNI"]
        scores["raw_cost"] = self.results_df.loc[df.index, "cost_objective_RNI"]
        angles = np.array([0, 2 * np.pi / 3, 4 * np.pi / 3])
        def _tri_area(row):
            r = np.array([row[axes[0]], row[axes[1]], row[axes[2]]], dtype=float)
            x = r * np.cos(angles)
            y = r * np.sin(angles)
            return 0.5 * abs(
                x[0] * y[1] + x[1] * y[2] + x[2] * y[0]
                - y[0] * x[1] - y[1] * x[2] - y[2] * x[0]
            )
        scores["triangle_area"] = scores.apply(_tri_area, axis=1)
        return scores
    def to_string(self, full: bool = False) -> str:
        def fmt_any(v) -> str:
            try:
                if isinstance(v, np.ndarray):
                    return f"ndarray(shape={v.shape}, dtype={v.dtype})"
            except Exception:
                pass
            if isinstance(v, pd.DataFrame):
                return f"DataFrame(shape={v.shape}, cols={list(v.columns)})"
            if isinstance(v, pd.Series):
                return f"Series(len={len(v)}, name={v.name})"
            return repr(v)
        lines = []
        lines.append(f"{self.__class__.__name__}" + "{{")
        lines.append("  # Core")
        lines.append(f"  indices_to_show: {self.indices_to_show}")
        lines.append(f"  display_to_real: {self.display_to_real}")
        lines.append(f"  real_to_display: {self.real_to_display}")
        lines.append(f"  dtl_full: {fmt_any(self.dtl_full)}")
        lines.append(f"  dcs_full: {fmt_any(self.dcs_full)}")
        lines.append(f"  palette: {px.colors.qualitative.Plotly}")
        lines.append("\n  # Layouts")
        lines.append("  layout_union: " + fmt_any(self.layout_union))
        lines.append(f"  layout_by_model: dict(len={len(self.layout_by_model)})")
        lines.append("\n  # Map")
        lines.append("  map keys: " + ", ".join(list(self.map.keys())))
        lines.append("\n  # Overview rows")
        lines.append(f"  rows={len(self.overview_rows)}")
        lines.append("}")
        return "\n".join(lines)
    def __str__(self) -> str:
        return self.to_string(full=False)

def build_viz_data(forest_area_3, model_list, results_df: pd.DataFrame) -> VizData:
    return VizData(forest_area_3=forest_area_3, model_list=model_list, results_df=results_df)

def prepare_map_data(forest_area_3, results_df: pd.DataFrame, model_list=None) -> Dict[str, Any]:
    return build_viz_data(forest_area_3, model_list, results_df).map

def get_overview_table_data(forest_area_3, model_list, results_df: pd.DataFrame) -> List[List[Any]]:
    return build_viz_data(forest_area_3, model_list, results_df).overview_rows

def get_selected_table_data(forest_area_3, model_list, results_df: pd.DataFrame, selected_index: int) -> List[List[Any]]:
    return build_viz_data(forest_area_3, model_list, results_df).selected_rows(selected_index)

def get_anchor_table_data(forest_area_3, model_list, results_df: pd.DataFrame, selected_index: int) -> List[List[str]]:
    return build_viz_data(forest_area_3, model_list, results_df).anchor_rows(selected_index)

def make_radar_scores(results_df: pd.DataFrame, axes: List[str]) -> pd.DataFrame:
    dummy = VizData(forest_area_3=None, model_list=None, results_df=results_df)
    return dummy.make_radar_scores(axes)

def _build_cable_profile(cr_object, line_geom) -> Dict[str, List[float]]:
    """Build loaded/unloaded cable profiles along the corridor distance."""
    if cr_object is None or line_geom is None:
        return {}

    def _collect_profile(cable) -> List[tuple[float, float, float]]:
        points = getattr(cable, "points_along_line", None)
        if not points:
            return []
        try:
            loaded = np.asarray(cable.absolute_loaded_line_height, dtype=float)
            unloaded = np.asarray(cable.absolute_unloaded_line_height, dtype=float)
        except Exception:
            return []
        if len(points) != len(loaded) or len(points) != len(unloaded):
            return []
        entries: List[tuple[float, float, float]] = []
        for p, lz, uz in zip(points, loaded, unloaded):
            try:
                px, py = float(p.x), float(p.y)
            except Exception:
                try:
                    px, py = float(p[0]), float(p[1])
                except Exception:
                    continue
            dist = float(line_geom.project(Point(px, py)))
            entries.append((dist, float(lz), float(uz)))
        return entries

    cables = []
    segments = getattr(cr_object, "supported_segments", None)
    if segments:
        if hasattr(cr_object, "get_all_subsegments"):
            segments_list = list(cr_object.get_all_subsegments())
        else:
            segments_list = list(segments)
        for segment in segments_list:
            cable = getattr(segment, "cable_road", None)
            if cable is not None:
                cables.append(cable)
    if not cables:
        cables = [cr_object]

    def _start_distance(cable) -> float:
        start_support = getattr(cable, "start_support", None)
        start_point = getattr(start_support, "xy_location", None)
        if start_point is None:
            return 0.0
        try:
            return float(line_geom.project(start_point))
        except Exception:
            return 0.0

    cables = sorted(cables, key=_start_distance)

    entries: List[tuple[float, float, float]] = []
    for cable in cables:
        entries.extend(_collect_profile(cable))
    if not entries:
        return {}

    entries.sort(key=lambda x: x[0])
    deduped: List[tuple[float, float, float]] = []
    tol = 1e-3
    for dist, lz, uz in entries:
        if not deduped or abs(dist - deduped[-1][0]) > tol:
            deduped.append((dist, lz, uz))
        else:
            deduped[-1] = (dist, lz, uz)

    return {
        "x": [d for d, _, _ in deduped],
        "loaded_y": [lz for _, lz, _ in deduped],
        "unloaded_y": [uz for _, _, uz in deduped],
    }

def get_side_profile_data(forest_area_3, corridor_real_index: int) -> dict:
    line_gdf = forest_area_3.line_gdf
    if corridor_real_index not in line_gdf.index:
        return {}
    row = line_gdf.loc[corridor_real_index]
    line_geom = row.geometry
    total_length = line_geom.length
    def get_terrain_heights(geometry, num_samples=100):
        if not hasattr(forest_area_3, 'height_gdf'):
            return None, None
        dists = np.linspace(0, geometry.length, num_samples)
        sample_points = [geometry.interpolate(d) for d in dists]
        sample_x = [p.x for p in sample_points]
        sample_y = [p.y for p in sample_points]
        pad = 20
        minx, miny, maxx, maxy = geometry.bounds
        df = forest_area_3.height_gdf
        subset = df[
            (df.x >= minx - pad) & (df.x <= maxx + pad) &
            (df.y >= miny - pad) & (df.y <= maxy + pad)
        ]
        if subset.empty:
            return None, None
        ground_pts = subset[['x', 'y']].to_numpy()
        ground_z = subset['elev'].to_numpy()
        z_values = []
        for sx, sy in zip(sample_x, sample_y):
            d = np.sqrt((ground_pts[:,0] - sx)**2 + (ground_pts[:,1] - sy)**2)
            idx = np.argmin(d)
            z_values.append(ground_z[idx])
        return dists, z_values
    terr_dists, terr_zs = get_terrain_heights(line_geom)
    if terr_zs is None:
        start_z_default = getattr(row, "start_z", 1000)
        end_z_default = getattr(row, "end_z", 1100)
        terr_dists = [0, total_length]
        terr_zs = [start_z_default, end_z_default]
    else:
        terr_dists = np.asarray(terr_dists, dtype=float)
        terr_zs = np.asarray(terr_zs, dtype=float)
    start_z = terr_zs[0]
    end_z = terr_zs[-1]
    
    # --- Extract Cable Road Info ---
    # Try different keys for volume/cost
    vol = _safe_float(row.get("wood_volume", row.get("volume_m3", row.get("volumen_m3", 0))))
    cost = _safe_float(row.get("line_cost", 0))
    # Calculate approx gradient in %
    if total_length > 0:
        gradient_pct = ((end_z - start_z) / total_length) * 100
    else:
        gradient_pct = 0.0

    cr_object = row.get("Cable Road Object")
    supports = []
    tree_coords = None
    tree_bhd_arr = None
    if hasattr(forest_area_3, "harvesteable_trees_gdf"):
        tgdf = forest_area_3.harvesteable_trees_gdf
        tree_coords = np.column_stack((tgdf.geometry.x, tgdf.geometry.y))
        tree_bhd_arr = tgdf.get("BHD", pd.Series([0]*len(tgdf))).to_numpy()
    if cr_object and hasattr(cr_object, "supported_segments"):
        for i, segment in enumerate(cr_object.supported_segments):
            if i == 0: continue 
            sup = segment.start_support
            loc = getattr(sup, "xy_location", None)
            if loc:
                dist = line_geom.project(Point(loc.x, loc.y))
            else:
                dist = 0.0
            z_ground = np.interp(dist, terr_dists, terr_zs)
            h_attach = getattr(sup, "attachment_height", 0)
            sup_bhd = None
            if loc and tree_coords is not None:
                dists_sq = np.sum((tree_coords - np.array([loc.x, loc.y]))**2, axis=1)
                nearest = np.argmin(dists_sq)
                if dists_sq[nearest] <= 4.0:
                    sup_bhd = float(tree_bhd_arr[nearest])
            supports.append({
                "x": dist, 
                "y_ground": z_ground,
                "height": h_attach,
                "attachment_height": h_attach,
                "type": "Support",
                "BHD": sup_bhd
            })
    end_tree_obj = getattr(row, "end_support_tree", getattr(row, "end_anchor_tree", None))
    _, _, tail_bhd, tail_h = _extract_tree_metadata(end_tree_obj)
    if tail_h is None:
        tail_h = 10
    tail_attach_h = None
    if cr_object is not None:
        end_support = getattr(cr_object, "end_support", None)
        if end_support is not None:
            try:
                tail_attach_h = float(end_support.attachment_height)
            except Exception:
                tail_attach_h = None
    if tail_attach_h is None:
        tail_attach_h = tail_h
    tail_tree_data = {
        "x": total_length,
        "y": end_z,
        "height": tail_attach_h,
    }
    if tail_bhd is not None:
        tail_tree_data["BHD"] = tail_bhd
    road_anchors_list = []
    ra_src = getattr(row, "road_anchor_tree_series", None)
    if isinstance(ra_src, pd.DataFrame) and not ra_src.empty:
         for _, r in ra_src.iterrows():
             _, _, rbhd, rh = _extract_tree_metadata(r)
             road_anchors_list.append({"BHD": rbhd, "height": rh})
    elif isinstance(ra_src, dict):
         if "features" in ra_src:
             for feat in ra_src["features"]:
                  props = feat.get("properties", {})
                  _, _, rbhd, rh = _extract_tree_metadata(props)
                  road_anchors_list.append({"BHD": rbhd, "height": rh})
         else:
             _, _, rbhd, rh = _extract_tree_metadata(ra_src)
             road_anchors_list.append({"BHD": rbhd, "height": rh})
    elif isinstance(ra_src, pd.Series):
         _, _, rbhd, rh = _extract_tree_metadata(ra_src)
         road_anchors_list.append({"BHD": rbhd, "height": rh})
    tail_anchor_count = 1 if end_tree_obj is not None else 0
    tail_anchor_list = []
    if tail_bhd is not None:
         tail_anchor_list.append({"BHD": tail_bhd})
    final_terrain_x = list(terr_dists)
    final_terrain_y = list(terr_zs)
    sorted_pairs = sorted(zip(final_terrain_x, final_terrain_y))
    tx, ty = zip(*sorted_pairs)
    cable_profile = _build_cable_profile(cr_object, line_geom)
    return {
        "terrain_x": tx,
        "terrain_y": ty,
        "yarder": {"x": 0, "y": start_z, "height": 12}, 
        "tail_tree": tail_tree_data,
        "supports": supports,
        "road_anchors": road_anchors_list,
        "road_anchor_count": len(road_anchors_list) if road_anchors_list else (1 if ra_src is not None else 0),
        "tail_anchors": tail_anchor_list,
        "tail_anchor_count": len(tail_anchor_list) if tail_anchor_list else 1,
        "corridor_id": corridor_real_index,
        "length_m": total_length,
        "volume_m3": vol,
        "cost": cost,
        "gradient": gradient_pct,
        "cable_profile": cable_profile,
    }