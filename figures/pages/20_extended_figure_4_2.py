import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

from figures.utils.load_data import load_extended_figure_4_2_data

st.set_page_config(layout="wide")
st.title("Figure_4_2")


def plot_dataset(data, method_colors, y_start, y_end):
    methods = ["vit_b", "vit_l", "vit_h"]

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
                text=(f"{int(value/1000000)}M"),
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
        margin=dict(l=80, r=20, t=20, b=40),
        showlegend=False,
    )

    return fig


if __name__ == "__main__":
    vit_colors = {
        "vit_h": "#E14774",
        "vit_l": "#1C8CA9",
        "vit_b": "#7058BD",
    }

    data = load_extended_figure_4_2_data()

    fig = plot_dataset(data, vit_colors, 0, 700000000)
    img_bytes = pio.to_image(fig, format="pdf", width=400, height=400, scale=2)
    st.plotly_chart(fig, use_container_width=False, theme=None)
    st.download_button(
        label="Download",
        data=img_bytes,
        file_name="extended_figure_4_2.pdf",
    )
