import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image, ImageDraw


def generate_annular_sector_polygon(inner_radius, outer_radius, num_vertices, start_angle, end_angle):
    angles = np.linspace(start_angle, end_angle, num_vertices, endpoint=True)
    inner_vertices = [(inner_radius * np.cos(angle), inner_radius * np.sin(angle)) for angle in angles]
    outer_vertices = [(outer_radius * np.cos(angle), outer_radius * np.sin(angle)) for angle in angles]
    vertices = inner_vertices + outer_vertices[::-1]
    return vertices


def generate_multiple_annular_sector_polygons(
    inner_radius, outer_radius, num_vertices_per_sector, num_sectors, gap_angle, rotation_angle=0
):
    gap_angle_rad = np.radians(gap_angle)
    total_gap = gap_angle_rad * num_sectors
    available_angle = 2 * np.pi - total_gap
    sector_angle = available_angle / num_sectors

    sectors_vertices = []
    sector_centers = []
    start_angle = np.radians(rotation_angle)
    for i in range(num_sectors):
        end_angle = start_angle + sector_angle
        vertices = generate_annular_sector_polygon(
            inner_radius, outer_radius, num_vertices_per_sector, start_angle, end_angle
        )
        sectors_vertices.append(vertices)

        center_angle = (start_angle + end_angle) / 2
        center_radius = (inner_radius + outer_radius) / 2
        center_x = center_radius * np.cos(center_angle)
        center_y = center_radius * np.sin(center_angle)
        sector_centers.append((center_x, center_y))

        start_angle = end_angle + gap_angle_rad

    return sectors_vertices, sector_centers


def put_image_in_canvas(image, offset, scale_ratio, center, canvas_size):
    image_size = image.size
    scaled_size = (int(image_size[0] * scale_ratio), int(image_size[1] * scale_ratio))
    resized_image = image.resize(scaled_size, Image.LANCZOS)

    canvas = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
    paste_position = (
        int(center[0] - scaled_size[0] / 2 + offset[0]),
        int(center[1] - scaled_size[1] / 2 + offset[1]),
    )
    canvas.paste(resized_image, paste_position)

    return canvas


def create_annular_image(
    images,
    inner_radius,
    outer_radius,
    canvas_size,
    gap_angle=None,
    offsets=None,
    scales=None,
    num_vertices_per_sector=50,
    rotation_angle=0,
):
    images = [Image.fromarray(image) for image in images]
    num_sectors = len(images)
    if gap_angle is None:
        gap_angle = 360 / num_sectors / 2
    if offsets is None:
        offsets = [(0, 0)] * num_sectors
    if scales is None:
        scales = [1] * num_sectors

    sectors_vertices, sector_centers = generate_multiple_annular_sector_polygons(
        inner_radius, outer_radius, num_vertices_per_sector, num_sectors, gap_angle, rotation_angle
    )

    final_image = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
    center = (canvas_size[0] // 2, canvas_size[1] // 2)

    for i, (image, offset, scale, sector_center) in enumerate(zip(images, offsets, scales, sector_centers)):
        sector_center = (sector_center[0] + center[0], sector_center[1] + center[1])

        processed_image = put_image_in_canvas(image, offset, scale, sector_center, canvas_size)

        mask = Image.new("L", canvas_size, 0)
        draw = ImageDraw.Draw(mask)
        vertices = [(x + center[0], y + center[1]) for x, y in sectors_vertices[i]]
        draw.polygon(vertices, fill=255)

        masked_image = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
        masked_image.paste(processed_image, (0, 0), mask)

        final_image = Image.alpha_composite(final_image, masked_image)

    return np.array(final_image)


def plot_radia_chart(
    datas,
    line_names,
    axis_names,
    line_colors,
    initial_angle=14,
    direction="counterclockwise",
    x_domain=[0.1, 0.9],
    y_domain=[0.1, 0.9],
    ticktext=None,
):
    fig = go.Figure()
    for i, data in enumerate(datas):
        fig.add_trace(
            go.Scatterpolar(
                r=data + [data[0]],
                theta=axis_names + [axis_names[0]],
                opacity=1.0,
                name=line_names[i],
                line=dict(color=line_colors.get(line_names[i], "#000000")),
                mode="markers+lines",
            )
        )

    fig.update_layout(
        polar=dict(
            domain=dict(x=x_domain, y=y_domain),
            radialaxis=dict(
                visible=True,
                range=[0.0, 1.0],
                showline=False,
                showgrid=True,
                ticks="",
                showticklabels=False,
            ),
            angularaxis=dict(
                rotation=initial_angle,
                visible=True,
                direction=direction,
                ticklen=100,
                tickfont=dict(size=15),
                ticks="",
                tickwidth=0,
                ticktext=ticktext,
                showline=False,
                showgrid=False,
                showticklabels=False,
            ),
        ),
        showlegend=True,
    )
    return fig


def add_background_to_polar_fig(fig, pil_image, zoom=1.0):
    ax = None
    for a in fig.axes:
        if isinstance(a, plt.PolarAxes):
            ax = a
            break

    if ax is None:
        raise ValueError("Figure does not contain a polar axis")
    background_image = np.array(pil_image)
    imagebox = OffsetImage(background_image, zoom=zoom)
    ab = AnnotationBbox(imagebox, xy=(0.5, 0.5), xycoords="axes fraction", frameon=False, box_alignment=(0.5, 0.5))
    ax.add_artist(ab)
