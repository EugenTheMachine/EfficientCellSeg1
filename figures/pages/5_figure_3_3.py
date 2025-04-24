from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import plotly.io as pio
import streamlit as st
from skimage import exposure

from figures.utils.load_data import load_figure_3_3_data
from figures.utils.load_image import load_multiple_gt_images
from visualize_cell import plot_image_with_mask

st.set_page_config(layout="wide")
st.title("Figure_3_3")


@dataclass
class Annotation:
    good_rotations: list = None
    bad_rotations: list = None
    good_boxes: list = None
    bad_boxes: list = None
    good_cell_nums: list = None
    bad_cell_nums: list = None
    good_ap50s: list = None
    bad_ap50s: list = None
    anno_for_good_zoom_images: list = None
    anno_for_bad_zoom_images: list = None


def resize_and_rescale_images(images, resize_shape):
    resized_images = [
        cv2.resize(image, resize_shape, interpolation=cv2.INTER_LINEAR_EXACT)
        for image in images
    ]
    resized_images = [
        exposure.rescale_intensity(image, out_range=(0, 255)).astype(np.uint8)
        for image in resized_images
    ]
    return resized_images


def resize_mask(masks, resize_shape):
    return [
        cv2.resize(mask, resize_shape, interpolation=cv2.INTER_NEAREST_EXACT)
        for mask in masks
    ]


def extract_patches(images, masks, boxes):
    zoom_images, zoom_masks = [], []
    for i, image in enumerate(images):
        x0, x1, y0, y1 = boxes[i]
        zoom_images.append(images[i][y0:y1, x0:x1])
        zoom_masks.append(masks[i][y0:y1, x0:x1])
    return zoom_images, zoom_masks


def normalize_boxes(boxes, box_base, resize_shape):
    return [
        [
            int(box[0] / box_base[0] * resize_shape[0]),
            int(box[1] / box_base[0] * resize_shape[0]),
            int(box[2] / box_base[1] * resize_shape[1]),
            int(box[3] / box_base[1] * resize_shape[1]),
        ]
        for box in boxes
    ]


def plot_annotations(
    images,
    masks,
    boxes,
    cell_nums,
    ap50s,
    show_contour_on_original,
    box_width,
    contour_width,
    anno_font_size,
    anno_rect_width,
    anno_rect_height,
    anno_color,
    box_dash,
    contour_dash,
    fig_size,
    prefix,
    save=False,
):
    image_files = []
    cols = st.columns(len(images))
    for i, (col, image, mask) in enumerate(zip(cols, images, masks)):
        if not show_contour_on_original:
            mask = None
        fig = plot_image_with_mask(
            image,
            mask=mask,
            box=boxes[i],
            box_dash=box_dash,
            box_width=box_width,
            contour_dash=contour_dash,
            contour_width=contour_width,
            box_color="#FF6188",
            fig_size=fig_size,
        )
        fig.add_shape(
            dict(
                type="rect",
                x0=0,
                y0=0,
                x1=anno_rect_width,
                y1=anno_rect_height,
                fillcolor="rgba(0, 0, 0, 0.5)",
                line=dict(color="rgba(0, 0, 0, 0.5)", width=0, dash=box_dash),
            )
        )
        fig.add_annotation(
            x=0,
            y=anno_rect_height / 2,
            xref="x",
            yref="y",
            xanchor="left",
            yanchor="middle",
            text=f"  {ap50s[i]:.2f}",
            showarrow=False,
            font=dict(
                family="Arial",
                size=anno_font_size,
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
            x=anno_rect_width,
            y=anno_rect_height / 2,
            xref="x",
            yref="y",
            xanchor="right",
            yanchor="middle",
            showarrow=False,
            text=f"{cell_nums[i]} cells  ",
            font=dict(
                family="Arial",
                size=anno_font_size,
                color=anno_color,
            ),
        )
        if save:
            Path("saved_images/figure_3/").mkdir(parents=True, exist_ok=True)
            image_file = (
                f"saved_images/figure_3/figure_3_visualize_{prefix}_images_{i}.pdf"
            )
            pio.write_image(fig, image_file)
            image_files.append(image_file)
        with col:
            st.components.v1.html(
                fig.to_html(include_mathjax="cdn"),
                height=fig_size[0],
                width=fig_size[1],
            )

    return


def plot_zoom_annotations(
    images,
    masks,
    anno_texts,
    contour_dash,
    contour_width,
    fig_size,
    anno_font_size,
    anno_color,
    box_base,
    box_width,
    prefix,
    border_color,
    save=False,
):
    image_files = []
    cols = st.columns(len(images))
    for i, (col, image, mask) in enumerate(zip(cols, images, masks)):
        fig = plot_image_with_mask(
            image,
            mask,
            box=None,
            contour_dash=contour_dash,
            contour_width=contour_width,
            fig_size=fig_size,
        )
        rect_height = 80 * image.shape[1] / box_base[1]
        fig.add_shape(
            dict(
                type="rect",
                x0=0,
                y0=0,
                x1=image.shape[0],
                y1=rect_height,
                fillcolor="rgba(0, 0, 0, 0.5)",
                line=dict(color="rgba(0, 0, 0, 0.5)", width=0, dash=contour_dash),
            )
        )
        fig.add_annotation(
            x=image.shape[0] / 2,
            y=rect_height / 2,
            xref="x",
            yref="y",
            xanchor="center",
            yanchor="middle",
            text=f"{anno_texts[i]}",
            showarrow=False,
            font=dict(
                family="Arial",
                size=anno_font_size,
                color=anno_color,
            ),
        )

        if save:
            image_file = (
                f"saved_images/figure_3/figure_3_visualize_{prefix}_zoom_images_{i}.pdf"
            )
            pio.write_image(fig, image_file)
            image_files.append(image_file)
        with col:
            st.components.v1.html(
                fig.to_html(include_mathjax="cdn"),
                height=box_base[0],
                width=box_base[1],
            )
    return


def configure_ui():
    cols = st.columns(5)
    with cols[0]:
        image_size = st.selectbox("Image Size", [100, 250, 500], index=1)
    with cols[1]:
        box_width = st.slider("Box Width", min_value=0, max_value=10, value=4, step=1)
    with cols[2]:
        contour_width = st.slider(
            "Contour Width", min_value=0, max_value=10, value=2, step=1
        )
    with cols[3]:
        show_contour_on_original = st.checkbox(
            "Show Contour on Original Image", value=False
        )
    with cols[4]:
        save_fig = st.checkbox("Save Images", value=False)
    return image_size, box_width, contour_width, show_contour_on_original, save_fig


def main():
    datasets = [
        "cellseg_blood",
        "deepbacs_rod_brightfield",
        "deepbacs_rod_fluorescence",
        "dsb2018_stardist",
        "cellpose_specialized",
    ]

    image_size, box_width, contour_width, show_contour_on_original, save_fig = (
        configure_ui()
    )
    box_dash = "solid"
    contour_dash = "solid"
    data = load_figure_3_3_data()
    good_train_ids = data["good_train_ids"]
    bad_train_ids = data["bad_train_ids"]
    resize_shape = (image_size, image_size)
    box_base = (500, 500)

    annotations = Annotation(
        good_rotations=data["good_rotations"],
        bad_rotations=data["bad_rotations"],
        good_boxes=data["good_boxes"],
        bad_boxes=data["bad_boxes"],
        good_ap50s=data["good_ap50s"],
        bad_ap50s=data["bad_ap50s"],
        anno_for_good_zoom_images=data["anno_for_good_zoom_images"],
        anno_for_bad_zoom_images=data["anno_for_bad_zoom_images"],
        good_cell_nums=data["good_cell_nums"],
        bad_cell_nums=data["bad_cell_nums"],
    )

    annotations.good_boxes = normalize_boxes(
        annotations.good_boxes, box_base, resize_shape
    )
    annotations.bad_boxes = normalize_boxes(
        annotations.bad_boxes, box_base, resize_shape
    )

    anno_font_size = int(image_size * 45 / box_base[1])
    anno_rect_width = int(500 * image_size / box_base[0])
    anno_rect_height = int(80 * image_size / box_base[1])
    anno_color = "#FFFFFF"
    border_color = "#FF6188"

    good_result = load_multiple_gt_images(good_train_ids, "train")
    bad_result = load_multiple_gt_images(bad_train_ids, "train")
    good_images, good_masks, bad_images, bad_masks = [], [], [], []
    for dataset in datasets:
        good_image = good_result[dataset]["images"][0]
        good_mask = good_result[dataset]["masks"][0]
        bad_image = bad_result[dataset]["images"][0]
        bad_mask = bad_result[dataset]["masks"][0]

        good_images.append(good_image)
        good_masks.append(good_mask)
        bad_images.append(bad_image)
        bad_masks.append(bad_mask)
    good_images = resize_and_rescale_images(good_images, resize_shape)
    good_masks = resize_mask(good_masks, resize_shape)
    bad_images = resize_and_rescale_images(bad_images, resize_shape)
    bad_masks = resize_mask(bad_masks, resize_shape)

    for i in range(len(good_images)):
        good_images[i] = np.rot90(good_images[i], k=annotations.good_rotations[i])
        good_masks[i] = np.rot90(good_masks[i], k=annotations.good_rotations[i])
    for i in range(len(bad_images)):
        bad_images[i] = np.rot90(bad_images[i], k=annotations.bad_rotations[i])
        bad_masks[i] = np.rot90(bad_masks[i], k=annotations.bad_rotations[i])

    good_zoom_images, good_zoom_masks = extract_patches(
        good_images, good_masks, annotations.good_boxes
    )
    bad_zoom_images, bad_zoom_masks = extract_patches(
        bad_images, bad_masks, annotations.bad_boxes
    )

    fig_size = (image_size, image_size)

    plot_annotations(
        good_images,
        good_masks,
        annotations.good_boxes,
        annotations.good_cell_nums,
        annotations.good_ap50s,
        show_contour_on_original,
        box_width,
        contour_width,
        anno_font_size,
        anno_rect_width,
        anno_rect_height,
        anno_color,
        box_dash,
        contour_dash,
        fig_size,
        prefix="good",
        save=save_fig,
    )

    plot_zoom_annotations(
        good_zoom_images,
        good_zoom_masks,
        annotations.anno_for_good_zoom_images,
        contour_dash,
        contour_width,
        fig_size,
        anno_font_size,
        anno_color,
        box_base,
        0,
        prefix="good",
        border_color=border_color,
        save=save_fig,
    )

    plot_annotations(
        bad_images,
        bad_masks,
        annotations.bad_boxes,
        annotations.bad_cell_nums,
        annotations.bad_ap50s,
        show_contour_on_original,
        box_width,
        contour_width,
        anno_font_size,
        anno_rect_width,
        anno_rect_height,
        anno_color,
        box_dash,
        contour_dash,
        fig_size,
        prefix="bad",
        save=save_fig,
    )

    plot_zoom_annotations(
        bad_zoom_images,
        bad_zoom_masks,
        annotations.anno_for_bad_zoom_images,
        contour_dash,
        contour_width,
        fig_size,
        anno_font_size,
        anno_color,
        box_base,
        0,
        prefix="bad",
        border_color=border_color,
        save=save_fig,
    )


if __name__ == "__main__":
    main()
