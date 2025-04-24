import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

from figures.utils.load_data import load_extended_figure_4_1_data

st.set_page_config(layout="wide")
st.title("Figure_4_2")


def plot_dataset(data, method_colors, y_start, y_end):
    methods = ["vit_b", "vit_l", "vit_h"]

    fig = go.Figure()

    for i, method in enumerate(methods):
        if (method not in data) or (data[method] is None):
            continue

        value = data[method]
        if value > 0.2:
            color = "white"
        else:
            color = "black"
        fig.add_trace(
            go.Bar(
                x=[i],
                y=[max(min(value, y_end), y_start)],
                name=method,
                marker_color=method_colors[method],
                text=[f"{value:.2f}"],
                textfont=dict(size=20, color=color),
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
    vit_colors = {
        "vit_h": "#E14774",
        "vit_l": "#1C8CA9",
        "vit_b": "#7058BD",
    }

    cols = st.columns(2)
    with cols[0]:
        y_start = st.number_input("Y-axis start", min_value=0.0, max_value=1.0, value=0.0, step=0.05)
    with cols[1]:
        y_end = st.number_input("Y-axis end", min_value=0.0, max_value=1.2, value=1.0, step=0.05)

    df = load_extended_figure_4_1_data()

    images_per_row = 2
    available_datasets = df["dataset_name"].unique()
    for i in range(0, len(available_datasets), images_per_row):
        cols = st.columns(images_per_row)
        for j in range(images_per_row):
            if i + j < len(available_datasets):
                dataset_name = available_datasets[i + j]
                data = df[df["dataset_name"] == dataset_name].copy()
                data.drop(columns=["dataset_name"], inplace=True)
                data = {row["vit_name"]: row["ap_0.5"] for _, row in data.iterrows()}
                with cols[j]:
                    fig = plot_dataset(data, vit_colors, y_start, y_end)
                    img_bytes = pio.to_image(fig, format="pdf", width=400, height=400, scale=2)
                    st.plotly_chart(fig, use_container_width=False, theme=None)
                    st.download_button(
                        label=f"Download {dataset_name}",
                        data=img_bytes,
                        file_name=f"extended_figure_4_{dataset_name}.pdf",
                    )
