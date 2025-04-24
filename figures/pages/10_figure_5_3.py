import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

from figures.utils.load_data import load_figure_5_3_data

st.set_page_config(layout="wide")
st.title("Figure_5_3")


dicts = load_figure_5_3_data()

cols = st.columns(3)


with cols[0]:
    fig_font_color = st.selectbox("Font Color", ["black", "auto", "white"])
with cols[1]:
    show_number = st.checkbox("Show Number", value=False)
    show_diagonal = st.checkbox("Show Diagonal", value=True)
with cols[2]:
    font_sie = st.slider("Font Size", min_value=10, max_value=30, value=20, step=1)

methods = list(dicts.keys())

heatmaps_per_row = 2
method_groups = [methods[i : i + heatmaps_per_row] for i in range(0, len(methods), heatmaps_per_row)]

red_to_blue = [[0.0, "#0000FF"], [0.5, "#FFFFFF"], [1.0, "#ff0000"]]
test_template = "%{text:.0f}" if show_number else ""

x_ticks = [f"TSN-{i}" for i in range(1, 15)]
y_ticks = [f"TSN-{i}" for i in range(1, 15)]

if fig_font_color == "black":
    fig_text_font = {"family": "Arial", "size": font_sie, "color": "black"}
elif fig_font_color == "white":
    fig_text_font = {"family": "Arial", "size": font_sie, "color": "white"}
elif fig_font_color == "auto":
    fig_text_font = {"family": "Arial", "size": font_sie}

for group in method_groups:
    cols = st.columns(heatmaps_per_row)

    for col, method in zip(cols, group):
        with col:
            heatmap_data = dicts[method]
            value_with_half_diagonal = heatmap_data.values
            if not show_diagonal:
                for i in range(heatmap_data.shape[0]):
                    for j in range(heatmap_data.shape[1]):
                        if i == j:
                            value_with_half_diagonal[i, j] = 0.5
            fig = go.Figure(
                data=go.Heatmap(
                    z=value_with_half_diagonal,
                    x=x_ticks,
                    y=y_ticks,
                    text=heatmap_data.values * 100,
                    texttemplate=test_template,
                    textfont=fig_text_font,
                    colorscale=red_to_blue,
                    zmin=0.0,
                    zmax=1.0,
                    xgap=1,
                    ygap=1,
                    showscale=False,
                )
            )

            fig.update_layout(
                plot_bgcolor="white",
                paper_bgcolor="white",
                width=500,
                height=500,
                title_font_size=10,
                autosize=True,
                margin={"b": 0, "t": 100, "r": 0, "l": 100},
                font={"size": font_sie, "family": "Arial", "color": "black"},
                xaxis_side="top",
                xaxis=dict(
                    tickangle=-90,
                    ticklen=0,
                    showticklabels=True,
                    showgrid=False,
                    tickfont={"family": "Arial", "size": font_sie, "color": "black"},
                ),
                yaxis=dict(
                    ticklen=0,
                    showticklabels=True,
                    showgrid=False,
                    scaleanchor="x",
                    scaleratio=1,
                    autorange="reversed",
                    tickfont={"family": "Arial", "size": font_sie, "color": "black"},
                ),
                coloraxis_showscale=False,
            )

            st.plotly_chart(fig, use_container_width=False, theme=None)

            if st.download_button(
                label=f"Download PDF {method}",
                data=pio.to_image(fig, format="pdf"),
                file_name=f"figure_5_3_{method}.pdf",
                mime="application/pdf",
            ):
                pass
