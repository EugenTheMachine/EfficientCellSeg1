import plotly.graph_objects as go
import streamlit as st

from figures.utils.load_data import load_figure_3_2_data, load_short_names


def create_scatter_trace(x, y, dataset_name, color, symbol, train_ids):
    return go.Scatter(
        x=x,
        y=y,
        mode="markers",
        marker=dict(
            size=5,
            color=color,
            symbol=symbol,
            opacity=1.0,
            line=dict(
                color=color,
                width=1.5,
            ),
        ),
        name=dataset_name,
        text=[f"{dataset_name}<br>Train ID: {tid}" for tid in train_ids],
        hovertemplate="<b>%{text}</b><br>" + "Train cell numbers: %{x}<br>" + "AP@0.5: %{y}<br>" + "<extra></extra>",
    )


def plot_cellseg1_across_datasets(data, colors):
    short_name, _ = load_short_names()
    fig = go.Figure()

    symbols = {
        "dsb2018_stardist": "circle",
        "cellpose_specialized": "circle",
        "cellseg_blood": "circle",
        "deepbacs_rod_brightfield": "circle",
        "deepbacs_rod_fluorescence": "circle",
    }

    for dataset_name, dataset_data in data.items():
        x_values = dataset_data["train_cell_nums"]
        y_values = dataset_data["accuracy"]
        trace = create_scatter_trace(
            x=x_values,
            y=y_values,
            dataset_name=short_name[dataset_name],
            color=colors[dataset_name],
            symbol=symbols[dataset_name],
            train_ids=dataset_data["train_ids"],
        )
        fig.add_trace(trace)

    fig.update_layout(
        height=350,
        width=500,
        plot_bgcolor="white",
        margin=dict(l=60, r=0, t=20, b=50),
        showlegend=True,
        legend=dict(
            font={"family": "Arial", "size": 25, "color": "black"},
            x=1,
            y=0,
            xanchor="right",
            yanchor="bottom",
            bgcolor="rgba(255,255,255,0.0)",
            itemsizing="constant",
            itemwidth=35,
        ),
        legend_tracegroupgap=5,
        title="",
        hoverlabel=dict(bgcolor="white", font_size=16, font_family="Arial"),
        title_font={"family": "Arial", "size": 35, "color": "black"},
        xaxis=dict(
            showgrid=True,
            gridcolor="#CCCCCC",
            gridwidth=2.0,
            griddash="dot",
            linecolor="black",
            linewidth=2,
            zeroline=False,
            tickfont={"family": "Arial", "size": 35, "color": "black"},
            title_font={"family": "Arial", "size": 35, "color": "black"},
            range=[0, 200],
            tickvals=[10, 30, 50, 100, 150],
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="#CCCCCC",
            gridwidth=2.0,
            griddash="dot",
            linecolor="black",
            linewidth=2,
            range=[0.0, 1.1],
            zeroline=False,
            tickfont={"family": "Arial", "size": 35, "color": "black"},
            tickvals=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            title_font={"family": "Arial", "size": 15, "color": "black"},
        ),
    )

    return fig


def main():
    st.set_page_config(layout="wide")
    st.title("Figure_3_2")

    colors = {
        "dsb2018_stardist": "#935356",
        "cellpose_specialized": "#E14775",
        "cellseg_blood": "#269D69",
        "deepbacs_rod_brightfield": "#7058BE",
        "deepbacs_rod_fluorescence": "#1C8CA8",
    }
    cellseg1_data = load_figure_3_2_data()

    fig = plot_cellseg1_across_datasets(cellseg1_data, colors)
    st.plotly_chart(fig, use_container_width=False, theme=None)

    img_bytes = fig.to_image(format="pdf", scale=2)
    st.download_button(
        label="Download Figure_3_2",
        data=img_bytes,
        file_name="figure_3_2.pdf",
    )


if __name__ == "__main__":
    main()
