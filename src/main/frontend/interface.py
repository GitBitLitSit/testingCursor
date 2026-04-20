# interface.py
import pandas as pd
import ipywidgets as w

from src.main.frontend.map import Map
from src.main.frontend.result_selector import ResultSelector
from src.main.frontend.radar_chart import build_radar_dashboard
from src.main.frontend.table import Table
from src.main.frontend.data_prep import build_viz_data
from src.main.frontend.side_profile import ProfileChart
from src.main.frontend.data_prep import get_side_profile_data

_NAMES = {
    "axes": ["Ergonomische Optimierung", "Ökologische Optimierung", "Kosten Optimierung"],
    "title": "Vergleich der Seiltrassenmodelle",
    "table_overview_headers": [
        "Modell",
        "Gesamt Kosten [€]",
        "Auf- und Abbau Kosten [€]",
        "Ökologische Penalty",
        "Ergonomische Penalty",
        "Verwendete Seillinien",
        "Max Zugseillänge [m]",
        "⌀ Zugseillänge [m]",
        "⌀ Stützbaum Anzahl",
        "Kosten pro Vfm [€/m³]",
        "Vfm pro m Seillänge [m³/m]",
    ],
    "table_selected_headers": [
        "Seiltrassen Nummer",
        "Aufbaukosten [€]",
        "Seillänge [m]",
        "Vfm pro Seiltrasse [m³]",
        "Stützbaum Anzahl",
        "Tragseilhöhe Stütze [m]",
        "⌀ Baumhöhe [m]",
        "Max Zugseillänge [m]",
        "⌀ Zugseillänge [m]",
    ],
    "table_anchor_headers": [
        "Seiltrassen Nummer",
        "BHD [cm]",
        "Höhe [m]",
        "X-Koordinate",
        "Y-Koordinate",
    ],
}

_SELECTED_TABLE_WIDTH = 1050


def build_interface(forest_area_3, model_list, results_df: pd.DataFrame) -> w.VBox:
    vd = build_viz_data(forest_area_3, model_list, results_df)

    map_component = Map(vd.map, "Seiltrassen Karte")
    
    selected_table_width = _SELECTED_TABLE_WIDTH
    profile_chart = ProfileChart(base_width=selected_table_width)
    profile_chart.container.layout.display = "none"

    scores = vd.make_radar_scores(_NAMES["axes"])
    radar_chart = build_radar_dashboard(
        scores, 600, 738, _NAMES["axes"],
        on_toggle=lambda idx, active: overview_table.unmute_row(idx) if active else overview_table.mute_row(idx)
    )

    # 1. Overview Table (Selects Model)
    overview_table = Table(
        _NAMES["table_overview_headers"],
        vd.overview_rows,
        1500,
        title=None,
        action_label="Auswählen",
        on_action=lambda idx: model_selector.set_value(idx),
    )

    # 2. Selected Table (Selects Corridor Profile)
    #    Added action_label="Profil" and on_action callback
    selected_table = Table(
        _NAMES["table_selected_headers"], 
        [], 
        selected_table_width,
        is_visible=False, 
        title="Aktivierte Seiltrassen",
        action_label="Auswählen", 
        on_action=lambda idx: corridor_selector.set_value(idx if idx != -1 else None)
    )
    
    anchor_table = Table(_NAMES["table_anchor_headers"], [], 422, is_visible=False, title="Endmast Informationen")

    # --- Logic ---

    def _set_profile_visibility(is_visible: bool) -> None:
        profile_chart.container.layout.display = "flex" if is_visible else "none"

    def on_corridor_select(corridor_idx_in_list: int | None):
        """Called when a 'Seiltrasse' is selected (via Dropdown OR Table Button)."""
        
        # 1. Sync Table Highlight (Two-way binding)
        #    If called from dropdown, ensure table row is highlighted.
        #    If called from table, this is redundant but harmless.
        if corridor_idx_in_list is None:
            selected_table.clear_highlight()
            _set_profile_visibility(False)
            return
        else:
            selected_table.highlight_row(corridor_idx_in_list)

        # 2. Logic to update Chart
        model_idx = model_selector._dropdown.value 
        if model_idx == -1: return

        real_ids = results_df.iloc[model_idx]["selected_lines"]
        
        if 0 <= corridor_idx_in_list < len(real_ids):
            real_id = int(real_ids[corridor_idx_in_list])
            
            display_id = vd.real_to_display.get(real_id, real_id)
            p_data = get_side_profile_data(vd.forest_area_3, real_id)
            p_data['display_id'] = display_id
            
            profile_chart.update(p_data)
            _set_profile_visibility(True)

    def on_model_select(idx: int | None):
        """Called when 'Modell' is changed."""
        
        _on_select_with_vd(
            idx, vd, results_df, selected_table, anchor_table, map_component, overview_table
        )

        if idx is None:
            corridor_selector.set_options(0)
            _set_profile_visibility(False)
            selected_table.clear_highlight() # Clear old selections
        else:
            real_ids = results_df.iloc[idx]["selected_lines"]
            
            custom_labels = []
            for rid in real_ids:
                did = vd.real_to_display.get(int(rid), int(rid))
                custom_labels.append(f"Seiltrasse {did}")
            
            corridor_selector.set_options(len(real_ids), custom_labels=custom_labels)
            
            # Hide profile and clear table highlight when switching models
            _set_profile_visibility(False)
            selected_table.clear_highlight()

    # --- Selectors ---

    model_selector = ResultSelector(
        num_results=len(results_df),
        label="Modell",
        prefix="Optimierung",
        on_select=on_model_select
    )

    corridor_selector = ResultSelector(
        num_results=0, 
        label="Seiltrasse",
        prefix="Seiltrasse",
        on_select=on_corridor_select
    )

    # --- Layout ---

    toolbar = w.HBox(
        [model_selector.get_widget(), corridor_selector.get_widget()],
        layout=w.Layout(width="100%", max_width="1500px", align_items="center", margin="5px 0"),
    )

    sel_widget  = selected_table.getWidget()
    anch_widget = anchor_table.getWidget()
    # Remove margin from sel_widget since flex gap handles spacing better now
    sel_widget.layout.margin = "0 24px 24px 0"

    details_row = w.Box(
        [sel_widget, anch_widget],
        layout=w.Layout(
            width="100%",
            align_items="flex-start",
            margin="20px 0",
            overflow="visible",
            display="flex",
            flex_flow="row wrap",
            justify_content="flex-start",
        ),
    )
    details_row.add_class("details-row")
    details_row.add_class("section-block")

    radar_chart.layout.width = "100%"
    radar_chart.layout.align_items = "center"
    radar_chart.add_class("section-block")

    ui = w.VBox(
        [
            map_component.get_map_widget(),
            toolbar,
            radar_chart,
            overview_table.getWidget(),
            details_row,
            profile_chart.get_widget()
        ],
        layout=w.Layout(width="100%", align_items="stretch", gap="20px"),
    )
    ui.add_class("app-shell")

    return ui

# ... (Keep _on_select_with_vd as is) ...
def _on_select_with_vd(
    selected_index: int | None,
    vd,
    results_df: pd.DataFrame,
    selected_table: Table,
    anchor_table: Table,
    map_component: Map,
    overview_table: Table,
) -> None:
    if selected_index is None:
        selected_table.update_data([])
        selected_table.set_visibility(False)
        anchor_table.update_data([])
        anchor_table.set_visibility(False)
        map_component.update_map()
        overview_table.highlight_row(-1)
        return

    overview_table.clear_button_selection()
    overview_table.highlight_row(selected_index)

    sel_rows  = vd.selected_rows(selected_index)
    anch_rows = vd.anchor_rows(selected_index)

    selected_lines = results_df.iloc[selected_index]["selected_lines"]
    map_component.update_map(selected_index, selected_lines)

    if sel_rows:
        selected_table.update_data(sel_rows)
        selected_table.set_visibility(True)
    else:
        selected_table.update_data([])
        selected_table.set_visibility(False)

    if anch_rows:
        anchor_table.update_data(anch_rows)
        anchor_table.set_visibility(True)
    else:
        anchor_table.update_data([])
        anchor_table.set_visibility(False)