import shutil

import numpy as np
import plotly.io as pio
import streamlit as st

from data.utils import resize_image, resize_mask
from figures.utils.load_data import load_ap_50_for_train_ids
from figures.utils.load_image import load_multiple_gt_images
from project_root import PROJECT_ROOT
from visualize_cell import plot_image_with_mask

st.set_page_config(layout="wide")
st.title("Image and Mask Display")


def plot_image_with_annotation(image, mask, ap50, cell_num, contour_width, show_contour, image_size):
    if not show_contour:
        mask = None
    else:
        h, w = mask.shape
        triangle_mask = np.triu(np.ones((h, w), dtype=bool))
        triangle_mask = np.flip(triangle_mask, axis=1)
        mask[triangle_mask] = 0

    fig = plot_image_with_mask(
        image,
        mask=mask,
        contour_dash="solid",
        contour_width=contour_width,
        fig_size=(image_size, image_size),
        box=None,
        contour_color="#EFEF00",
    )

    fig.add_shape(
        type="line",
        x0=0,
        y0=image_size,
        x1=image_size,
        y1=0,
        line=dict(color="#EFEF00", width=contour_width, dash="dot"),
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
    )
    fig.update_xaxes(scaleanchor="y", scaleratio=1)
    return fig


def configure_ui():
    cols = st.columns(4)
    with cols[0]:
        image_size = st.selectbox("Image Size", [100, 250, 500], index=1)
    with cols[1]:
        contour_width = st.slider("Contour Width", min_value=0, max_value=10, value=2, step=1)
    with cols[2]:
        show_contour = st.checkbox("Show Contour", value=True)
    with cols[3]:
        save_images = st.checkbox("Save Images", value=False)
    return image_size, contour_width, show_contour, save_images


def main():
    image_ids_for_all_datasets = {
        "cellseg_blood": [87, 91, 117, 83, 49],
        "deepbacs_rod_brightfield": [11, 4, 7, 8, 9],
        "deepbacs_rod_fluorescence": [69, 47, 16, 75, 79],
        "dsb2018_stardist": [278, 312, 162, 313, 41],
        "cellpose_specialized": [51, 49, 48, 0, 12],
    }
    rotations_for_all_datasets = {
        "cellseg_blood": [0, 0, 0, 0, 0],
        "deepbacs_rod_brightfield": [0, 0, 0, 0, 0],
        "deepbacs_rod_fluorescence": [2, 2, 2, 2, 2],
        "dsb2018_stardist": [0, 0, 0, 0, 0],
        "cellpose_specialized": [0, 0, 0, 0, 0],
    }
    datasets = list(image_ids_for_all_datasets.keys())

    ap50s_for_all_datasets = load_ap_50_for_train_ids(image_ids_for_all_datasets, "cellseg1")

    image_size, contour_width, show_contour, save_images = configure_ui()
    save_dir = PROJECT_ROOT / "saved_images/extended_figure_1"
    if save_dir.exists():
        shutil.rmtree(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    resize_shape = (image_size, image_size)

    image_and_gt_data = load_multiple_gt_images(image_ids_for_all_datasets, "train")

    for row_idx in range(len(datasets)):
        dataset = datasets[row_idx]
        st.write(f"**Dataset: {dataset}**")
        images = image_and_gt_data[dataset]["images"]
        masks = image_and_gt_data[dataset]["masks"]
        num_cells = image_and_gt_data[dataset]["num_cells"]
        ap_50s = ap50s_for_all_datasets[dataset]
        image_ids = image_ids_for_all_datasets[dataset]
        cols = st.columns(len(image_ids))
        for col_idx, col in enumerate(cols):
            image = resize_image(images[col_idx], resize_shape)
            mask = resize_mask(masks[col_idx], resize_shape)
            image = np.rot90(image, k=rotations_for_all_datasets[dataset][col_idx])
            mask = np.rot90(mask, k=rotations_for_all_datasets[dataset][col_idx])
            image_id = image_ids[col_idx]
            ap50 = ap_50s[image_id]
            num_cell = num_cells[col_idx]
            fig = plot_image_with_annotation(image, mask, ap50, num_cell, contour_width, show_contour, image_size)
            if save_images:
                image_file = save_dir / f"dataset_{dataset}_image_{image_id}.pdf"
                pio.write_image(fig, image_file)
            with col:
                st.components.v1.html(fig.to_html(include_mathjax="cdn"), height=image_size, width=image_size)


if __name__ == "__main__":
    main()
