import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

from figures.utils.load_data import load_color, load_figure_4_2_data

st.set_page_config(layout="wide")
st.title("Figure_4_2")


def plot_dataset(data, method_colors, dataset_name, y_start, y_end):
    methods = [
        "cellseg1",
        "stardist",
        "cellpose-scratch",
        "cellpose-cyto2",
        "CellSAM",
    ]

    fig = go.Figure()

    for i, method in enumerate(methods):
        if (method not in data) or (data[method] is None):
            continue

        value = data[method]

        fig.add_trace(
            go.Bar(
                x=[i],
                y=[max(min(value, y_end), y_start)],
                name=method,
                marker_color=method_colors[method],
                text=[f"{value:.2f}"],
                textfont=dict(size=20, color="white"),
                customdata=[value < y_start or value > y_end],
                showlegend=True,
            )
        )

    fig.update_layout(
        plot_bgcolor="white",
        title_font={"family": "Arial", "size": 25},
        width=400,
        height=400,
        xaxis=dict(
            showgrid=False,
            linecolor="black",
            linewidth=2,
            tickfont={"family": "Arial", "size": 25, "color": "black"},
            title="",
            showticklabels=False,
            tickmode="array",
            tickvals=list(range(len(methods))),
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="#CCCCCC",
            gridwidth=0.5,
            griddash="dot",
            linecolor="black",
            linewidth=2,
            range=[y_start, y_end],
            zeroline=False,
            tickfont={"family": "Arial", "size": 25, "color": "black"},
            title="",
            title_font={"family": "Arial", "size": 25, "color": "black"},
        ),
        margin=dict(l=50, r=20, t=20, b=40),
        showlegend=False,
    )

    for i, method in enumerate(methods):
        if method not in data or data[method] is None:
            continue

        value = data[method]

        if value < y_start or value > y_end:
            marker_y = y_start if value < y_start else y_end
            marker_symbol = "triangle-up" if value < y_start else "triangle-down"
            fig.add_trace(
                go.Scatter(
                    x=[i],
                    y=[marker_y],
                    mode="markers+text",
                    marker=dict(
                        symbol=marker_symbol, size=10, color=method_colors[method], line=dict(width=2, color="black")
                    ),
                    text=f"{value:.2f}",
                    textposition="top center" if value < y_start else "bottom center",
                    textfont=dict(size=15),
                    showlegend=False,
                )
            )

    return fig


if __name__ == "__main__":
    method_colors = load_color()
    dataset_name = "cellpose_generalized"

    cols = st.columns(2)
    with cols[0]:
        y_start = st.number_input("Y-axis start", min_value=0.0, max_value=1.0, value=0.0, step=0.05)
    with cols[1]:
        y_end = st.number_input("Y-axis end", min_value=0.0, max_value=1.2, value=1.0, step=0.05)

    data = load_figure_4_2_data()
    fig = plot_dataset(data, method_colors, dataset_name, y_start, y_end)
    st.plotly_chart(fig, use_container_width=False, theme=None)
    if st.download_button(
        label=f"Download PDF {dataset_name}",
        data=pio.to_image(fig, format="pdf"),
        file_name="figure_4_2.pdf",
        mime="application/pdf",
    ):
        pass
