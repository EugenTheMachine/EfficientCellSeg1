import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

from figures.utils.load_data import load_color, load_figure_4_1_data


def create_scatter_trace(x, y, method, color, dash_style):
    return go.Scatter(
        x=x,
        y=y,
        mode="lines+markers",
        marker=dict(size=15, color=color),
        line=dict(width=3, color=color, dash=dash_style),
        name=method,
        showlegend=False,
    )


def plot_single_dataset(dataset_data, dataset_name, method_colors):
    methods = list(dataset_data.keys())
    dash_styles = {
        "cellseg1": "solid",
        "cellpose-cyto2": "longdash",
        "cellpose-cyto": "dot",
        "stardist": "dash",
        "cellpose-scratch": "dashdot",
    }

    fig = go.Figure()

    x_mapping = {"1": 1, "5": 6, "10": 11}

    full_num = {
        "cellpose_specialized": 89,
        "dsb2018_stardist": 447,
        "cellseg_blood": 139,
        "deepbacs_rod_brightfield": 19,
        "deepbacs_rod_fluorescence": 80,
    }
    x_mapping[str(full_num[dataset_name])] = 16
    x_ticks = ["1", "5", "10", str(full_num[dataset_name])]

    for method in methods:
        x_values = [x_mapping[x] for x in dataset_data[method].keys()]
        y_values = list(dataset_data[method].values())
        trace = create_scatter_trace(
            x=x_values, y=y_values, method=method, color=method_colors[method], dash_style=dash_styles[method]
        )
        fig.add_trace(trace)

    fig.update_layout(
        height=400,
        width=400,
        plot_bgcolor="white",
        margin=dict(l=40, r=20, t=20, b=40),
        showlegend=False,
    )

    fig.update_xaxes(
        showgrid=True,
        gridcolor="#CCCCCC",
        gridwidth=0.5,
        griddash="dot",
        linecolor="black",
        linewidth=2,
        zeroline=False,
        tickfont={"family": "Arial", "size": 25, "color": "black"},
        tickvals=[1, 6, 11, 16],
        ticktext=x_ticks,
        range=[0, 17],
        title_text="",
    )

    fig.update_yaxes(
        showgrid=True,
        gridcolor="#CCCCCC",
        gridwidth=0.5,
        griddash="dot",
        linecolor="black",
        linewidth=2,
        range=[-0.05, 1.05],
        zeroline=False,
        tickfont={"family": "Arial", "size": 25, "color": "black"},
        title_text="",
    )

    return fig


def display_all_datasets(data):
    method_colors = load_color()
    images_per_row = 2
    for i in range(0, len(data), images_per_row):
        cols = st.columns(images_per_row)
        for j in range(images_per_row):
            if i + j < len(data):
                dataset_name = list(data.keys())[i + j]
                dataset_data = data[dataset_name]
                with cols[j]:
                    fig = plot_single_dataset(dataset_data, dataset_name, method_colors)
                    st.plotly_chart(fig, use_container_width=False)

                    img_bytes = pio.to_image(fig, format="pdf", width=400, height=400, scale=2)
                    st.download_button(
                        label=f"Download {dataset_name}",
                        data=img_bytes,
                        file_name=f"figure_4_{dataset_name}.pdf",
                    )


if __name__ == "__main__":
    st.set_page_config(layout="wide")

    st.title("Figure_4_1")

    data = load_figure_4_1_data()
    display_all_datasets(data)
