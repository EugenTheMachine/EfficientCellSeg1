import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

from figures.utils.load_data import load_color, load_figure_3_1_data, load_short_names

if __name__ == "__main__":
    st.set_page_config(layout="wide")
    st.title("Figure_3_1")

    row1_cols = st.columns(6)

    with row1_cols[0]:
        y_range_start = st.slider("Y Axis Start", -0.3, 0.1, 0.0, step=0.05)

    with row1_cols[1]:
        y_range_stop = st.slider("Y Axis Stop", 0.8, 1.2, 1.1, step=0.05)

    with row1_cols[2]:
        violin_width = st.slider("Violin Width", 0.5, 1.5, 0.9, step=0.05)

    with row1_cols[3]:
        box_width = st.slider("Box Width", 0.05, 0.3, 0.12, step=0.01)

    with row1_cols[4]:
        violin_line_width = st.slider("Violin Line width", 0.0, 3.0, 0.0, step=0.05)

    with row1_cols[5]:
        box_line_width = st.slider("Box Line width", 0.0, 3.0, 1.5, step=0.05)

    row2_cols = st.columns(6)

    with row2_cols[0]:
        bandwidth = st.slider("Bandwidth", 0.01, 0.1, 0.05, step=0.01)

    with row2_cols[1]:
        quartilemethod = st.selectbox(
            "Quartile Method",
            ["linear", "exclusive", "inclusive"],
        )

    with row2_cols[2]:
        show_points = st.selectbox(
            "Show Points",
            [
                False,
                "all",
                "outliers",
                "suspectedoutliers",
            ],
        )

    with row2_cols[3]:
        spanmode = st.selectbox(
            "Spanmode",
            [
                "manual",
                "soft",
                "hard",
            ],
        )

    with row2_cols[4]:
        span_multipler = st.slider("Span Multiple", 0.0, 4.0, 2.0, step=0.1)

    metric = "ap_0.5"

    def plot_violin(
        datas,
        colors,
        extra_info=None,
        x_title="x title",
        y_title="y title",
        title="Title",
        legend_title="Legend title",
        height=600,
        width=600,
    ):
        fig = go.Figure()
        for i, (data_name, data) in enumerate(datas.items()):
            color = colors[data_name]
            customdata = None
            if extra_info is not None:
                customdata = extra_info[data_name]
            fig.add_trace(
                go.Violin(
                    y=data,
                    name=data_name,
                    x0=i,
                    box_visible=True,
                    meanline_visible=False,
                    fillcolor=color,
                    opacity=1.0,
                    marker=dict(size=1.5, color="black", opacity=1.0),
                    line=dict(color="black", width=violin_line_width),
                    box=dict(
                        visible=False,
                        line=dict(
                            color="black",
                            width=0,
                        ),
                        fillcolor="#EEEEEE",
                        width=0,
                    ),
                    bandwidth=bandwidth,
                    quartilemethod=quartilemethod,
                    pointpos=-box_width * 2,
                    span=[
                        max(min(data) - span_multipler * bandwidth, 0),
                        min(max(data) + span_multipler * bandwidth, 1),
                    ],
                    spanmode=spanmode,
                    width=violin_width,
                    scalemode="count",
                    points=show_points,
                    customdata=customdata,
                    hovertemplate=(
                        f"Method: {data_name}<br>"
                        "Value: %{y}<br>"
                        "Info: %{customdata}"
                    ),
                )
            )
            box_q1 = np.percentile(data, 25)
            box_q3 = np.percentile(data, 75)
            box_iqr = box_q3 - box_q1
            box_median = np.median(data)
            box_lowerfence = max(box_q1 - 1.5 * box_iqr, min(data))
            box_upperfence = min(box_q3 + 1.5 * box_iqr, max(data))
            box_plot = go.Box(
                name="box",
                boxmean=False,
                boxpoints=False,
                x0=i,
                line=dict(
                    color="black",
                    width=box_line_width,
                ),
                fillcolor="#EEEEEE",
                width=box_width,
                lowerfence=[box_lowerfence],
                q1=[box_q1],
                median=[box_median],
                q3=[box_q3],
                upperfence=[box_upperfence],
            )
            fig.add_trace(box_plot)

        fig.update_layout(
            plot_bgcolor="white",
            height=height,
            width=width,
            title=title,
            title_font={"family": "Arial", "size": 35, "color": "black"},
            xaxis=dict(
                showgrid=False,
                linecolor="black",
                linewidth=2,
                zeroline=False,
                tickfont={"family": "Arial", "size": 35, "color": "black"},
                title=x_title,
                title_font={"family": "Arial", "size": 35, "color": "black"},
                range=[-1, 4],
                tickvals=[],
                title_standoff=420,
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor="#CCCCCC",
                gridwidth=2.0,
                griddash="dot",
                linecolor="black",
                linewidth=2,
                range=[y_range_start, y_range_stop],
                zeroline=False,
                tickfont={"family": "Arial", "size": 35, "color": "black"},
                title=y_title,
                tickvals=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                title_font={"family": "Arial", "size": 35, "color": "black"},
            ),
            legend_title=legend_title,
            showlegend=False,
            legend=dict(
                x=0.5,
                xanchor="auto",
                y=1.1,
                bgcolor="rgba(255,255,255,0.5)",
                font=dict(family="Arial", size=25, color="black"),
                orientation="h",
            ),
            violinmode="group",
            violingap=0,
            violingroupgap=0,
            margin=dict(l=60, r=0, t=20, b=50),
        )
        return fig

    st.title("Figure Violin")

    colors = load_color()
    columns_per_row = 2

    dataset_names = [
        "cellpose_specialized",
        "cellseg_blood",
        "deepbacs_rod_brightfield",
        "deepbacs_rod_fluorescence",
        "dsb2018_stardist",
    ]

    short_names, _ = load_short_names()
    datasets_in_rows = [
        dataset_names[i : i + columns_per_row]
        for i in range(0, len(dataset_names), columns_per_row)
    ]
    dicts = load_figure_3_1_data()

    for row in datasets_in_rows:
        row1_cols = st.columns(columns_per_row)
        for col, dataset_name in zip(row1_cols, row):
            fig = plot_violin(
                dicts["violin_data"][dataset_name],
                colors,
                extra_info=dicts["train_ids"][dataset_name],
                title="",
                y_title="",
                x_title=short_names[dataset_name],
                legend_title="",
                height=350,
                width=500,
            )
            with col:
                st.plotly_chart(fig, use_container_width=False, theme=None)
                if col.download_button(
                    label=f"Download PDF for {dataset_name}",
                    data=pio.to_image(fig, format="pdf"),
                    file_name=f"figure_3_1_{dataset_name}.pdf",
                    mime="application/pdf",
                ):
                    pass
