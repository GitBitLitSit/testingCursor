# result_selector.py
from typing import Callable, List, Optional

import ipywidgets as w

_THEME = {
    "card_bg": "rgb(241, 248, 241)",
    "card_border": "#94b48a",
    "text": "#0f2010",
    "hover_border": "#48723b",
}

_DROPDOWN_CSS = w.HTML(
    f"""
<style>
  .app-scope .widget-dropdown select {{
    border-radius: 16px;
    border: 2px solid {_THEME['card_border']};
    background-color: {_THEME['card_bg']};
    color: {_THEME['text']};
    outline: none;
  }}
  .app-scope .widget-dropdown select:hover,
  .app-scope .widget-dropdown select:focus {{
    border-color: {_THEME['hover_border']};
    box-shadow: none;
  }}
  .app-scope .widget-dropdown,
  .app-scope .widget-dropdown *,
  .app-scope .sort-label {{
    font-family: -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Inter,Helvetica,Arial,sans-serif;
    font-size: 14px;
  }}
  .app-scope .widget-label {{ background: transparent; }}
</style>
""",
    layout=w.Layout(display="none"),
)

_SENTINEL_NONE = -1

class ResultSelector:
    def __init__(
        self,
        num_results: int = 0,
        on_select: Optional[Callable[[Optional[int]], None]] = None,
        label: str = "Modell",
        prefix: str = "Optimierung",
    ):
        self.num_results = num_results
        self._on_select = on_select
        self.label = label
        self.prefix = prefix
        self._create_widgets()

    def _create_widgets(self) -> None:
        # Initial default options
        self._results_options = self._generate_options(self.num_results)
        
        self._dropdown = w.Dropdown(
            options=self._results_options,
            value=_SENTINEL_NONE,
            disabled=False,
            style={"description_width": "initial"},
            layout=w.Layout(width="300px"),
        )
        self._dropdown.add_class("widget-dropdown")
        self._dropdown.observe(self._handle_change, names="value")

        label_html = w.HTML(f"<span class='sort-label'><b>{self.label}:</b></span>")
        container = w.HBox(
            [label_html, self._dropdown],
            layout=w.Layout(width="300px", align_items="center", margin="0 0 10px 0"),
        )
        self.widget = w.VBox([_DROPDOWN_CSS, container])
        self.widget.add_class("app-scope")

    def _generate_options(
        self,
        count: int,
        custom_labels: Optional[List[str]] = None,
    ) -> List[tuple[str, int]]:
        if custom_labels and len(custom_labels) == count:
            labels = [(label, i) for i, label in enumerate(custom_labels)]
        else:
            labels = [(f"{self.prefix} {i + 1}", i) for i in range(count)]
        return [("Keine Auswahl", _SENTINEL_NONE)] + labels

    def _handle_change(self, change: dict) -> None:
        if change.get("name") == "value":
            if callable(self._on_select):
                new_val = change["new"]
                self._on_select(None if new_val == _SENTINEL_NONE else new_val)

    def set_options(self, count: int, custom_labels: Optional[List[str]] = None) -> None:
        """
        Update options dynamically.
        If custom_labels is provided, it uses those strings for the dropdown.
        """
        self.num_results = count
        self._dropdown.options = self._generate_options(count, custom_labels)
        self._dropdown.value = _SENTINEL_NONE

    def set_value(self, idx: Optional[int]) -> None:
        self._dropdown.value = _SENTINEL_NONE if idx is None else idx

    def get_widget(self) -> w.Widget:
        return self.widget
