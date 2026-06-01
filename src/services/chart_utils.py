import json

def build_chart_payload(chart_cfg: dict, rows: list, columns: list) -> str | None:
    """JSON payload for Streamlit ECharts."""
    try:
        ctype = chart_cfg.get("type", "bar").lower()
        x_col = chart_cfg.get("x")
        y_col = chart_cfg.get("y")
        title = chart_cfg.get("title", "Chart")

        if not x_col or not y_col:
            return None

        if x_col not in columns or y_col not in columns:
            return None

        x_idx = columns.index(x_col)
        y_idx = columns.index(y_col)

        x_data = [str(r[x_idx]) for r in rows]
        y_data = [float(r[y_idx]) if r[y_idx] is not None else 0.0 for r in rows]

        if ctype == "pie":
            series_data = [{"name": x_data[i], "value": y_data[i]} for i in range(len(x_data))]
            option = {
                "title": {"text": title, "left": "center"},
                "tooltip": {"trigger": "item"},
                "series": [
                    {
                        "type": "pie",
                        "radius": "50%",
                        "data": series_data,
                    }
                ]
            }
        else:
            # bar, line, scatter
            option = {
                "title": {"text": title, "left": "center"},
                "tooltip": {"trigger": "axis"},
                "xAxis": {"type": "category", "data": x_data, "name": x_col},
                "yAxis": {"type": "value", "name": y_col},
                "series": [{"data": y_data, "type": ctype}],
            }
        return json.dumps(option)
    except Exception as e:
        print(f"Error building chart payload: {e}")
        return None
