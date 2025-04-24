import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

from figures.utils.load_data import load_extended_figure_5_data

st.set_page_config(layout="wide")
st.title("Training Metrics Comparison")


def plot_dataset(data, colors, dataset_names, y_label, y_start, y_end, value_format):
    fig = go.Figure()

    for i, dataset in enumerate(dataset_names):
        dataset_data = data[data["dataset_name"] == dataset]

        for j, (_, row) in enumerate(dataset_data.iterrows()):
            value = row[y_label]
            if y_label == "train_time":
                value = value / 1000 / 60  # Convert to minutes
            elif y_label == "peak_memory":
                value = value / 1024  # Convert to GB

            text = value_format.format(value)

            fig.add_trace(
                go.Bar(
                    x=[i * 4 + j],  # Position bars in groups of 3
                    y=[max(min(value, y_end), y_start)],
                    name=f"{dataset} (BS={row['batch_size']})",
                    marker_color=colors[j],
                    text=text,
                    textfont=dict(size=15, color="white"),
                    customdata=[value < y_start or value > y_end],
                    showlegend=False,
                )
            )

    fig.update_layout(
        plot_bgcolor="white",
        title_font={"family": "Arial", "size": 20},
        bargap=0.0,
        bargroupgap=0.0,
        xaxis=dict(
            showgrid=False,
            linecolor="black",
            linewidth=2,
            tickfont={"family": "Arial", "size": 20, "color": "black"},
            title="",
            showticklabels=True,
            tickmode="array",
            tickvals=[1 + i * 4 for i in range(5)],  # Center dataset labels
            ticktext=dataset_names,
            # tickangle=45,
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
            tickfont={"family": "Arial", "size": 20, "color": "black"},
            # title=y_label,
            title="",
            title_font={"family": "Arial", "size": 20, "color": "black"},
        ),
        margin=dict(l=50, r=10, t=20, b=40),
        showlegend=False,
        barmode="group",
    )
    return fig


if __name__ == "__main__":
    colors = ["#269D68", "#3e5c76", "#bc4749"]

    # Read and prepare data
    data = load_extended_figure_5_data()
    dataset_names = data["dataset_name"].unique()

    # Create three columns for the three plots

    st.subheader("Training Time")
    fig = plot_dataset(
        data=data,
        colors=colors,
        dataset_names=dataset_names,
        y_label="train_time",
        y_start=0,
        y_end=75,
        value_format="{:.0f} min",
    )
    img_bytes = pio.to_image(fig, format="pdf", width=800, height=300)
    st.plotly_chart(fig, use_container_width=False, theme=None)
    st.download_button(
        label="Download",
        data=img_bytes,
        file_name="extended_figure_5_1.pdf",
    )


    st.subheader("Peak Memory Usage")
    fig = plot_dataset(
        data=data,
        colors=colors,
        dataset_names=dataset_names,
        y_label="peak_memory",
        y_start=0,
        y_end=24,
        value_format="{:.1f} GB",
    )
    img_bytes = pio.to_image(fig, format="pdf", width=800, height=300)
    st.plotly_chart(fig, use_container_width=False, theme=None)
    st.download_button(
        label="Download",
        data=img_bytes,
        file_name="extended_figure_5_2.pdf",
    )

    st.subheader("AP@0.5")
    fig = plot_dataset(
        data=data,
        colors=colors,
        dataset_names=dataset_names,
        y_label="ap_0.5",
        y_start=0,
        y_end=1,
        value_format="{:.2f}",
    )
    img_bytes = pio.to_image(fig, format="pdf", width=800, height=300)
    st.plotly_chart(fig, use_container_width=False, theme=None)
    st.download_button(
        label="Download",
        data=img_bytes,
        file_name="extended_figure_5_3.pdf",
    )