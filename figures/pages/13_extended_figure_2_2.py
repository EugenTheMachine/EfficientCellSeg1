import shutil

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

from data.utils import resize_image, resize_mask
from figures.utils.load_image import load_gt_images, load_pred_masks
from metrics import average_precision
from project_root import PROJECT_ROOT
from visualize_cell import find_contours_with_padding, plot_image_with_mask

st.set_page_config(layout="wide")
st.title("Image and Mask Display")


def plot_image_with_annotation(
    image, pred_mask, true_mask, ap50, cell_num, contour_width, image_size, boxs
):
    fig = plot_image_with_mask(
        image,
        mask=true_mask,
        contour_dash="solid",
        contour_width=contour_width,
        fig_size=(image_size, image_size),
        box=None,
        contour_color="#EFEF00",
    )
    if pred_mask is not None:
        contours = find_contours_with_padding(pred_mask)
        x, y = [], []
        for contour in contours:
            y = np.concatenate([y, contour[:, 0], [None]])
            x = np.concatenate([x, contour[:, 1], [None]])
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                line=dict(color="#78DCE8", width=contour_width, dash="solid"),
                hoverinfo="skip",
            )
        )
    if boxs is not None:
        box = boxs[0]
        x0, x1, y0, y1 = box
        fig.add_shape(
            type="rect",
            x0=x0,
            y0=y0,
            x1=x1,
            y1=y1,
            line=dict(color="#FF6188", width=6, dash="solid"),
        )
        box = boxs[1]
        x0, x1, y0, y1 = box
        fig.add_shape(
            type="rect",
            x0=x0,
            y0=y0,
            x1=x1,
            y1=y1,
            line=dict(color="#A9DC76", width=6, dash="solid"),
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
        image_size = st.selectbox("Image Size", [100, 250, 500], index=2)
    with cols[1]:
        contour_width = st.slider(
            "Contour Width", min_value=0, max_value=10, value=2, step=1
        )
    with cols[2]:
        save_images = st.checkbox("Save Images", value=False)
    return image_size, contour_width, save_images


if __name__ == "__main__":
    train_ids_all = {
        "cellseg_blood": 138,
        "deepbacs_rod_brightfield": 6,
        "deepbacs_rod_fluorescence": 70,
        "dsb2018_stardist": 435,
        "cellpose_specialized": 60,
    }
    test_ids_all = {
        "cellseg_blood": [4],
        "deepbacs_rod_brightfield": [11],
        "deepbacs_rod_fluorescence": [7],
        "dsb2018_stardist": [31],
        "cellpose_specialized": [0],
    }
    boxes = {
        "cellseg_blood": [[200, 420, 3, 223], [320, 480, 230, 390]],
        "deepbacs_rod_brightfield": [[70, 170, 360, 460], [290, 410, 210, 330]],
        "deepbacs_rod_fluorescence": [[3, 103, 280, 380], [260, 410, 170, 320]],
        "dsb2018_stardist": [[160, 280, 160, 280], [220, 420, 290, 490]],
        "cellpose_specialized": [[0, 80, 360, 440], [0, 80, 110, 190]],
    }
    rotations_for_all_datasets = {
        "cellseg_blood": [0, 0, 0, 0, 0, 0],
        "deepbacs_rod_brightfield": [0, 0, 0, 0, 0, 0],
        "deepbacs_rod_fluorescence": [0, 0, 0, 0, 0, 0],
        "dsb2018_stardist": [0, 0, 0, 0, 0, 0],
        "cellpose_specialized": [0, 0, 0, 0, 0, 0],
    }
    datasets = list(train_ids_all.keys())

    images_all = {}
    pred_masks_all = {}
    true_masks_all = {}
    ap50_all = {}
    num_cells_all = {}
    for dataset in datasets:
        train_id = train_ids_all[dataset]
        test_ids = test_ids_all[dataset]
        images, true_masks, num_cells = load_gt_images(
            dataset, select_ids=test_ids, train_or_test="test"
        )
        for i, true_mask in enumerate(true_masks):
            if dataset in [
                "cellseg_blood",
                "deepbacs_rod_brightfield",
                "deepbacs_rod_fluorescence",
            ]:
                true_masks[i] = resize_mask(true_mask, (512, 512))
                images[i] = resize_image(images[i], (512, 512))
        pred_masks = load_pred_masks(dataset, train_id=train_id, test_ids=test_ids)
        ap, tp, fp, fn = average_precision(true_masks, pred_masks, threshold=[0.5])
        ap50_all[dataset] = {image_id: ap[i][0] for i, image_id in enumerate(test_ids)}
        pred_masks_all[dataset] = pred_masks
        images_all[dataset] = images
        num_cells_all[dataset] = num_cells
        true_masks_all[dataset] = true_masks

    image_size, contour_width, save_images = configure_ui()
    save_dir = PROJECT_ROOT / "saved_images/extended_figure_2"
    if save_images and save_dir.exists():
        shutil.rmtree(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    resize_shape = (image_size, image_size)

    for row_idx in range(len(datasets)):
        dataset = datasets[row_idx]
        st.write(f"**Dataset: {dataset}**")
        images = images_all[dataset]
        pred_masks = pred_masks_all[dataset]
        true_masks = true_masks_all[dataset]
        num_cells = num_cells_all[dataset]
        ap_50s = ap50_all[dataset]
        image_ids = test_ids_all[dataset]
        cols = st.columns(len(image_ids))
        for col_idx, col in enumerate(cols):
            image = resize_image(images[col_idx], resize_shape)
            pred_mask = resize_mask(pred_masks[col_idx], resize_shape)
            true_mask = resize_mask(true_masks[col_idx], resize_shape)
            boxs = boxes[dataset]
            image = np.rot90(image, k=rotations_for_all_datasets[dataset][col_idx])
            pred_mask = np.rot90(
                pred_mask, k=rotations_for_all_datasets[dataset][col_idx]
            )
            true_mask = np.rot90(
                true_mask, k=rotations_for_all_datasets[dataset][col_idx]
            )
            image_id = image_ids[col_idx]
            ap50 = ap_50s[image_id]
            num_cell = num_cells[col_idx]
            fig = plot_image_with_annotation(
                image,
                pred_mask,
                true_mask,
                ap50,
                num_cell,
                contour_width,
                image_size,
                boxs,
            )
            if save_images:
                image_file = save_dir / f"dataset_{dataset}_image_{image_id}.pdf"
                pio.write_image(fig, image_file)
            with col:
                st.components.v1.html(
                    fig.to_html(include_mathjax="cdn"),
                    height=image_size,
                    width=image_size,
                )
