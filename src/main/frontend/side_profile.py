import ipywidgets as w
import plotly.graph_objects as go
from typing import Dict, Any, Optional
import numpy as np

class ProfileChart:
    """
    Visualizes the Side Profile of a single Cable Corridor.
    """
    def __init__(self, base_width: int = 1050):
        self._base_width = int(base_width)
        self.fig = go.FigureWidget()
        self._setup_layout()
        self._last_data: Optional[Dict[str, Any]] = None
        
        # 1. Title Widget
        self._title_html = w.HTML(
            value="<div style='font-weight:800; font-size:18px; margin-bottom:6px;'>Seiltrassen Profilansicht</div>",
            layout=w.Layout(width="100%")
        )

        # 2. Inner wrapper (Holds the Plotly Figure)
        #    Fixed width to match base_width exactly (1050px).
        self.chart_wrapper = w.Box(
            [self.fig],
            layout=w.Layout(
                width=f"{self._base_width}px",
                min_width=f"{self._base_width}px",
                max_width=f"{self._base_width}px",
                display="inline-block",
                flex="0 0 auto",
                border="2px solid #94b48a", 
                background_color="rgb(241, 248, 241)", 
                padding="0px",       
                overflow="hidden" 
            )
        )
        self.chart_wrapper.add_class("border-radius")

        # CSS helper
        self._css = w.HTML(
            "<style>"
            ".border-radius { border-radius: 12px; box-sizing: border-box; }"
            "</style>"
        )

        # 3. Scroll container
        #    Handles the scrollbar if the screen is narrower than the fixed chart_wrapper.
        self.scroll_container = w.Box(
            [self.chart_wrapper],
            layout=w.Layout(
                width="100%",          
                min_width="0",         # Important: allows flex item to shrink below content size
                overflow_x="auto",     # Scrollbar appears here if needed
                overflow_y="hidden",
                display="block"
            )
        )
        
        # 4. Main Container
        #    Width is 100% (responsive) but capped at base_width (1050px) to match the table.
        #    min_width is removed so it doesn't force page scrolling.
        self.container = w.VBox(
            [self._css, self._title_html, self.scroll_container],
            layout=w.Layout(
                width="100%",
                max_width=f"{self._base_width}px", 
                padding="0px",
                background_color="rgb(241, 248, 241)",
                overflow="hidden",      
                align_items="stretch",
                gap="10px"
            )
        )

    def _setup_layout(self):
        self.fig.update_layout(
            paper_bgcolor="rgb(241, 248, 241)", 
            plot_bgcolor="rgb(241, 248, 241)",
            margin=dict(l=30, r=20, t=20, b=110),
            xaxis=dict(title="Distanz (m)", showgrid=False),
            yaxis=dict(title="Höhe (m)", showgrid=True, gridcolor="#d0d0d0"),
            hovermode="closest",
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="top",
                y=-0.15,
                xanchor="left",
                x=0
            ),
            height=450,
            width=self._base_width,
            autosize=False 
        )

    def update(self, data: Dict[str, Any]):
        """
        Expects data dict from get_side_profile_data()
        """
        self._last_data = data or {}
        self._render(self._last_data)

    def _render(self, data: Dict[str, Any]) -> None:
        self.fig.data = [] 
        self.fig.layout.shapes = []
        self.fig.layout.annotations = [] 
        
        if not data:
            self._title_html.value = "<div style='font-weight:800; font-size:18px;'>Keine Seiltrasse ausgewählt</div>"
            return

        # Title
        c_id = data.get('display_id', data.get('corridor_id', '?'))
        self._title_html.value = f"<div style='font-weight:800; font-size:18px;'>Seiltrasse {c_id}</div>"

        cable_profile = data.get("cable_profile") or {}
        has_cable_profile = bool(cable_profile.get("x"))

        # Extracted Stats
        c_len = data.get("length_m", 0.0)
        c_cost = data.get("cost", 0.0)

        # --- 1. Terrain Construction ---
        yx, yy = data["yarder"]["x"], data["yarder"]["y"]
        tx_end, ty_end = data["tail_tree"]["x"], data["tail_tree"]["y"]
        road_anchors = data.get("road_anchors", [])
        tail_anchors = data.get("tail_anchors", [])
        ra_count = data.get("road_anchor_count", len(road_anchors))
        ta_count = data.get("tail_anchor_count", len(tail_anchors))
        tx = np.array(data["terrain_x"])
        ty = np.array(data["terrain_y"])
        
        base_left = float(tx[0])
        base_right = float(tx[-1])
        left_anchor_x = yx - 8 - max(0, ra_count - 1) * 6 if ra_count else base_left
        right_anchor_x = tx_end + 8 + max(0, ta_count - 1) * 6 if ta_count else base_right
        left_edge = min(base_left, left_anchor_x)
        right_edge = max(base_right, right_anchor_x)
        span = max(1.0, base_right - base_left)
        pad = max(5.0, min(15.0, span * 0.03))
        
        # --- SIZE ADJUSTMENT ---
        # Force graph to stay at base_width. 
        # The container will scroll if the screen is smaller.
        fig_width = self._base_width
        
        self.fig.update_layout(width=fig_width)
        self.chart_wrapper.layout.width = f"{fig_width}px"
        self.chart_wrapper.layout.min_width = f"{fig_width}px"
        self.chart_wrapper.layout.max_width = f"{fig_width}px"

        slope_start = (ty[1] - ty[0]) / (tx[1] - tx[0]) if len(tx) > 1 else 0
        ext_x_left = np.array([left_edge - pad, left_edge - pad * 0.5])
        ext_y_left = ty[0] + slope_start * (ext_x_left - base_left) + 2.0

        slope_end = (ty[-1] - ty[-2]) / (tx[-1] - tx[-2]) if len(tx) > 1 else 0
        ext_x_right = np.array([right_edge + pad * 0.5, right_edge + pad])
        ext_y_right = ty[-1] + slope_end * (ext_x_right - base_right) - 2.0

        full_tx = np.concatenate([ext_x_left, tx, ext_x_right])
        full_ty = np.concatenate([ext_y_left, ty, ext_y_right])
        
        idx_sort = np.argsort(full_tx)
        final_tx = full_tx[idx_sort]
        final_ty = full_ty[idx_sort]

        min_y = np.min(final_ty)
        bottom_limit = min_y - 10
        max_y_candidates = [float(np.max(final_ty))]

        poly_x = np.concatenate([final_tx, [final_tx[-1], final_tx[0]]])
        poly_y = np.concatenate([final_ty, [bottom_limit, bottom_limit]])

        # Trace 1: Fill
        self.fig.add_trace(go.Scatter(
            x=poly_x, y=poly_y, fill='toself', mode='lines',
            line=dict(width=0), fillcolor='rgba(92, 64, 51, 0.2)',
            name='Gelände', hoverinfo='skip',
            showlegend=False
        ))

        # Trace 2: Line
        self.fig.add_trace(go.Scatter(
            x=final_tx, y=final_ty, mode='lines',
            line=dict(color='#5c4033', width=2),
            name='Geländekante', hoverinfo='x+y',
            showlegend=True
        ))
        self.fig.update_xaxes(range=[left_edge - pad, right_edge + pad])

        # --- Helper: Draw Tree with Hover ---
        def add_tree_shape(x, y_ground, visual_trunk_height, real_height_for_hover, 
                           color="green", label="", bhd=None,
                           crown_h=10.0, crown_w=6.0, show_height_tooltip=True,
                           height_label="Tragseilhöhe"):
            trunk_width = 1.0       
            trunk_top = y_ground + visual_trunk_height
            
            self.fig.add_shape(type="rect",
                x0=x - trunk_width/2, x1=x + trunk_width/2, 
                y0=y_ground, y1=trunk_top,
                fillcolor="brown", line_width=0, layer="below"
            )
            
            crown_base_y = trunk_top
            crown_tip_y = crown_base_y + crown_h
            
            path = (
                f"M {x - crown_w/2},{crown_base_y} "
                f"L {x},{crown_tip_y} "
                f"L {x + crown_w/2},{crown_base_y} Z"
            )
            
            self.fig.add_shape(type="path", path=path, fillcolor=color, line_color="black", line_width=1, layer="below")
            
            if label:
                 self.fig.add_annotation(
                    x=x, y=crown_tip_y + 5,
                    text=label, showarrow=False, font=dict(size=10, color="#333")
                )
            
            if bhd is not None:
                try:
                    bhd_val = float(bhd)
                    bhd_str = f"{bhd_val:.1f} cm"
                except:
                    bhd_str = str(bhd)
            else:
                bhd_str = "N/A" 

            hover_template = (
                f"<b>{label if label else 'Baum'}</b><br>"
                f"X: {x:.1f} m<br>"
                f"Y: {y_ground:.1f} m<br>"
                f"BHD: %{{customdata[0]}}<br>"
            )
            if show_height_tooltip:
                hover_template += f"{height_label}: %{{customdata[1]:.1f}} m<extra></extra>"
            else:
                hover_template += "<extra></extra>"

            self.fig.add_trace(go.Scatter(
                x=[x],
                y=[y_ground + (visual_trunk_height + crown_h/2)], 
                mode='markers',
                marker=dict(size=15, color='rgba(0,0,0,0)'),
                customdata=[[bhd_str, real_height_for_hover]], 
                hovertemplate=hover_template,
                showlegend=False,
                name=label
            ))

        # --- Helper: Yarder ---
        def add_yarder_shape(x, y, h):
            base_w, base_h = 4.0, 1.2
            cab_w, cab_h = 2.5, 2.5
            mast_top_w = 0.5

            p_tracks = f"M {x - base_w/2},{y} L {x + base_w/2},{y} L {x + base_w/2 - 0.2},{y + base_h} L {x - base_w/2 + 0.2},{y + base_h} Z"
            p_cab = f"M {x - cab_w/2},{y + base_h} L {x + cab_w/2},{y + base_h} L {x + cab_w/2},{y + base_h + cab_h} L {x - cab_w/2},{y + base_h + cab_h} Z"
            p_mast = f"M {x - 0.4},{y + base_h + cab_h} L {x + 0.4},{y + base_h + cab_h} L {x + mast_top_w/2},{y + h} L {x - mast_top_w/2},{y + h} Z"

            full_path = p_tracks + " " + p_cab + " " + p_mast

            self.fig.add_shape(type="path", path=full_path, fillcolor="#333", line_color="black", line_width=1)
            self.fig.add_annotation(x=x, y=y+h+6, text="Seilgerät", showarrow=False)

            self.fig.add_trace(go.Scatter(
                x=[x], y=[y + h/2],
                mode='markers',
                marker=dict(size=20, color='rgba(0,0,0,0)'),
                hovertemplate=f"<b>Seilgerät</b><br>Höhe: {h:.1f} m<extra></extra>",
                showlegend=False
            ))

        # --- 2. Yarder ---
        yh = data["yarder"]["height"]
        max_y_candidates.append(yy + yh + 3.0)
        add_yarder_shape(yx, yy, yh)

        # --- 3. Tail Tree (Endmast) ---
        th = data["tail_tree"]["height"]
        tail_attach_h = data["tail_tree"].get("attachment_height", th)
        dt = data["tail_tree"]
        tbhd = dt.get("BHD") or dt.get("bhd")
        
        visual_trunk_h_end = tail_attach_h + 1.0
        max_y_candidates.append(ty_end + visual_trunk_h_end + 10.0 + 2.0)
        
        add_tree_shape(tx_end, ty_end, 
                       visual_trunk_height=visual_trunk_h_end, 
                       real_height_for_hover=tail_attach_h, 
                       label="Endmast", 
                       bhd=tbhd,
                       crown_h=10.0, crown_w=6.0,
                       show_height_tooltip=True)
        
        # --- 4. Supports ---
        cable_points_x = [yx]
        cable_points_y = [yy + (yh * 1.0)]
        
        for i, sup in enumerate(data["supports"]):
            sx, sy = sup["x"], sup["y_ground"]
            sh = sup.get("attachment_height", sup["height"])
            sbhd = sup.get("BHD") or sup.get("bhd")
            max_y_candidates.append(sy + (sh + 1.0) + 10.0 + 2.0)
            
            add_tree_shape(sx, sy, 
                           visual_trunk_height=sh + 1.0, 
                           real_height_for_hover=sh, 
                           color="#228B22", 
                           label=f"Stütze {i+1}", 
                           bhd=sbhd,
                           crown_h=10.0, crown_w=6.0,
                           show_height_tooltip=True)
            
            cable_points_x.append(sx)
            cable_points_y.append(sy + sh)

        cable_points_x.append(tx_end)
        cable_points_y.append(ty_end + tail_attach_h)

        # --- 5. Skyline ---
        cable_x = cable_points_x
        cable_loaded_y = cable_points_y
        cable_unloaded_y = None
        if has_cable_profile:
            cable_x = cable_profile["x"]
            cable_loaded_y = cable_profile["loaded_y"]
            cable_unloaded_y = cable_profile["unloaded_y"]

        custom_data_skyline = [[c_id, c_len, c_cost] for _ in cable_x]

        self.fig.add_trace(go.Scatter(
            x=cable_x,
            y=cable_loaded_y,
            mode="lines",
            line=dict(color="#94b48a", width=1.5, shape="spline", smoothing=0.4),
            name="Seiltrasse belasteter Zustand",
            customdata=custom_data_skyline,
            hovertemplate=(
                "<b>Seiltrasse %{customdata[0]}</b><br>"
                "Länge: %{customdata[1]:.1f} m<br>"
                "Kosten: %{customdata[2]:.0f} €<extra></extra>"
            ),
            showlegend=True
        ))
        if has_cable_profile and cable_unloaded_y is not None:
            self.fig.add_trace(go.Scatter(
                x=cable_x,
                y=cable_unloaded_y,
                mode="lines",
                line=dict(color="#1f77b4", width=1.5, shape="spline", smoothing=0.4),
                name="Seiltrasse unbelasteter Zustand",
                customdata=custom_data_skyline,
                hovertemplate=(
                    "<b>Seiltrasse %{customdata[0]}</b><br>"
                    "Länge: %{customdata[1]:.1f} m<br>"
                    "Kosten: %{customdata[2]:.0f} €<extra></extra>"
                ),
                showlegend=True
            ))
        if len(cable_loaded_y):
            max_y_candidates.append(float(np.max(cable_loaded_y)))
        if cable_unloaded_y is not None:
            max_y_candidates.append(float(np.max(cable_unloaded_y)))
        self.fig.update_layout(showlegend=True)

        # --- 6. Road Anchors ---
        ra_label = "Ankerbaum" if ra_count == 1 else "Ankerbäume"
        dotted_color = "#7a9c74"
        
        for i in range(ra_count):
            ax = yx - 8 - (i*6)
            ay = np.interp(ax, final_tx, final_ty)
            max_y_candidates.append(ay + 2.0 + 6.0 + 2.0)
            
            abha = None
            if i < len(road_anchors):
                item = road_anchors[i]
                if isinstance(item, dict):
                    abha = item.get("BHD") or item.get("bhd")

            add_tree_shape(ax, ay, 
                           visual_trunk_height=2.0, 
                           real_height_for_hover=0, 
                           color="#556B2F", 
                           label=(ra_label if i==0 else ""), 
                           bhd=abha,
                           crown_h=6.0, crown_w=3.0,
                           show_height_tooltip=False)
            
            self.fig.add_trace(go.Scatter(
                x=[ax, yx], y=[ay+4, yy+(yh*0.8)], 
                mode="lines", line=dict(color=dotted_color, width=1, dash="dot"),
                hoverinfo="skip",
                showlegend=False
            ))

        # --- 7. Tail Anchors ---
        ta_label = "Ankerbaum" if ta_count == 1 else "Ankerbäume"

        for i in range(ta_count):
            ax = tx_end + 8 + (i*6)
            ay = np.interp(ax, final_tx, final_ty)
            max_y_candidates.append(ay + 2.0 + 6.0 + 2.0)
            
            abha = None
            if i < len(tail_anchors):
                item = tail_anchors[i]
                if isinstance(item, dict):
                    abha = item.get("BHD") or item.get("bhd")

            add_tree_shape(ax, ay, 
                           visual_trunk_height=2.0, 
                           real_height_for_hover=0, 
                           color="#556B2F", 
                           label=(ta_label if i==0 else ""), 
                           bhd=abha,
                           crown_h=6.0, crown_w=3.0,
                           show_height_tooltip=False)
            
            self.fig.add_trace(go.Scatter(
                x=[tx_end, ax], y=[ty_end+(th*0.6), ay+4], 
                mode="lines", line=dict(color=dotted_color, width=1, dash="dot"),
                hoverinfo="skip",
                showlegend=False
            ))

        max_y = max(max_y_candidates) if max_y_candidates else float(np.max(final_ty))
        y_pad = max(5.0, (max_y - min_y) * 0.05)
        self.fig.update_yaxes(range=[bottom_limit, max_y + y_pad])

    def get_widget(self):
        return self.container