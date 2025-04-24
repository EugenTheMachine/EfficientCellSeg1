import shutil

import plotly.io as pio
import streamlit as st

from data.utils import resize_image, resize_mask
from figures.utils.load_image import load_gt_images
from figures.utils.load_data import load_extended_figure_2_1_data
from project_root import PROJECT_ROOT
from visualize_cell import plot_image_with_mask

st.set_page_config(layout="wide")
st.title("Image and Mask Display")


def plot_image_with_annotation(image, true_mask, ap50, cell_num, contour_width, image_size):
    fig = plot_image_with_mask(
        image,
        mask=true_mask,
        contour_dash="solid",
        contour_width=contour_width,
        fig_size=(image_size, image_size),
        box=None,
        contour_color="#EFEF00",
    )

    anno_font_size = int(45 * image_size / 500)
    anno_rect_width = int(image_size)
    anno_rect_height = int(80 * image_size / 500)

    anno_color = "#FFFFFF"
    fig.add_shape(
        dict(
            type="rect",
            x0=0,
            y0=0,
            x1=anno_rect_width,
            y1=anno_rect_height,
            fillcolor="rgba(0, 0, 0, 0.5)",
            line=dict(color="rgba(0, 0, 0, 0.5)", width=0, dash="solid"),
        )
    )
    fig.add_annotation(
        x=0,
        y=anno_rect_height / 2,
        xref="x",
        yref="y",
        xanchor="left",
        yanchor="middle",
        text=f"  {ap50:.2f}",
        showarrow=False,
        font=dict(
            family="Arial",
            color=anno_color,
        ),
    )
    fig.update_layout(
        font=dict(
            family="Arial",
            size=anno_font_size,
            color=anno_color,
        ),
    )
    fig.add_annotation(
        x=anno_rect_width - 10,
        y=anno_rect_height / 2,
        xref="x",
        yref="y",
        xanchor="right",
        yanchor="middle",
        showarrow=False,
        text=f"{cell_num} cells  ",
        font=dict(
            family="Arial",
            size=anno_font_size,
            color=anno_color,
        ),
    )

    fig.update_layout(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        autosize=False,
        width=image_size,
        height=image_size,
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False,
    )
    fig.update_xaxes(scaleanchor="y", scaleratio=1)
    return fig


def configure_ui():
    cols = st.columns(3)
    with cols[0]:
        image_size = st.selectbox("Image Size", [100, 250, 500], index=1)
    with cols[1]:
        contour_width = st.slider("Contour Width", min_value=0, max_value=10, value=2, step=1)
    with cols[2]:
        save_images = st.checkbox("Save Images", value=False)
    return image_size, contour_width, save_images


if __name__ == "__main__":
    train_ids_all, ap50s = load_extended_figure_2_1_data()
    datasets = list(train_ids_all.keys())

    images = []
    true_masks = []
    num_cells = []
    for dataset in datasets:
        train_id = train_ids_all[dataset]
        images_, true_masks_, num_cells_ = load_gt_images(dataset, select_ids=[train_id], train_or_test="train")
        image = images_[0]
        true_mask = true_masks_[0]
        num_cell = num_cells_[0]
        if dataset in ["cellseg_blood", "deepbacs_rod_brightfield", "deepbacs_rod_fluorescence"]:
            true_mask = resize_mask(true_mask, (512, 512))
            image = resize_image(image, (512, 512))
        images.append(image)
        true_masks.append(true_mask)
        num_cells.append(num_cell)

    image_size, contour_width, save_images = configure_ui()
    save_dir = PROJECT_ROOT / "saved_images/extended_figure_2"
    if save_dir.exists():
        shutil.rmtree(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    resize_shape = (image_size, image_size)

    cols = st.columns(len(datasets))
    for col_idx, col in enumerate(cols):
        dataset = datasets[col_idx]
        image = images[col_idx]
        true_mask = true_masks[col_idx]
        num_cell = num_cells[col_idx]
        ap_50 = ap50s[col_idx]
        image_id = train_ids_all[dataset]
        image = resize_image(image, resize_shape)
        true_mask = resize_mask(true_mask, resize_shape)
        fig = plot_image_with_annotation(
            image,
            true_mask,
            ap_50,
            num_cell,
            contour_width,
            image_size,
        )
        if save_images:
            image_file = save_dir / f"dataset_{dataset}_train_image_{image_id}.pdf"
            pio.write_image(fig, image_file)
        with col:
            st.components.v1.html(fig.to_html(include_mathjax="cdn"), height=image_size, width=image_size)
