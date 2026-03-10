from pathlib import Path
import json

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches


DATASET_DIR = Path("../data/dataset")
IMAGES_DIR = DATASET_DIR / "images"
COLORED_LABELS_DIR = DATASET_DIR / "labels_colored"

IMAGE_EXTS = {".png", ".jpg", ".jpeg"}


def load_image_and_annotations(image_path: Path):
    label_path = COLORED_LABELS_DIR / f"{image_path.stem}.txt"
    if not label_path.exists():
        raise FileNotFoundError(f"No label file found for {image_path.name}: {label_path}")

    image = Image.open(image_path).convert("RGB")
    img_w, img_h = image.size

    with open(label_path, "r", encoding="utf-8") as f:
        annotations = json.load(f)

    boxes = []
    for ann in annotations:
        x_center = ann["x_center"] * img_w
        y_center = ann["y_center"] * img_h
        box_w = ann["width"] * img_w
        box_h = ann["height"] * img_h

        x1 = x_center - box_w / 2
        y1 = y_center - box_h / 2
        x2 = x_center + box_w / 2
        y2 = y_center + box_h / 2

        boxes.append({
            "annotation": ann,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "removed": False,
        })

    return image, boxes


def point_in_box(x, y, box):
    return box["x1"] <= x <= box["x2"] and box["y1"] <= y <= box["y2"]


def draw_state(ax, image, boxes, image_name):
    ax.clear()
    ax.imshow(image)
    ax.set_title(
        f"{image_name}\nClick a box to toggle remove/keep. Close window when done testing."
    )
    ax.axis("off")

    for box in boxes:
        ann = box["annotation"]
        color = [c / 255 for c in ann["color_rgb"]]

        width = box["x2"] - box["x1"]
        height = box["y2"] - box["y1"]

        if box["removed"]:
            edgecolor = (1, 0, 0)
            linestyle = "--"
            linewidth = 2
            alpha = 0.7
        else:
            edgecolor = color
            linestyle = "-"
            linewidth = 3
            alpha = 1.0

        rect = patches.Rectangle(
            (box["x1"], box["y1"]),
            width,
            height,
            linewidth=linewidth,
            edgecolor=edgecolor,
            facecolor="none",
            linestyle=linestyle,
            alpha=alpha,
        )
        ax.add_patch(rect)

        ax.text(
            box["x1"],
            max(0, box["y1"] - 5),
            f'id={ann["instance_id"]}',
            color=edgecolor,
            fontsize=9,
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1),
        )

    plt.draw()


def main():
    image_files = sorted([p for p in IMAGES_DIR.iterdir() if p.suffix.lower() in IMAGE_EXTS])
    if not image_files:
        raise RuntimeError(f"No images found in {IMAGES_DIR}")

    # Step 1: only open the first image
    image_path = image_files[0]
    image, boxes = load_image_and_annotations(image_path)

    fig, ax = plt.subplots(figsize=(12, 8))
    draw_state(ax, image, boxes, image_path.name)

    def onclick(event):
        if event.inaxes != ax or event.xdata is None or event.ydata is None:
            return

        # If several boxes overlap, toggle the smallest one first
        clicked_boxes = [
            box for box in boxes
            if point_in_box(event.xdata, event.ydata, box)
        ]

        if not clicked_boxes:
            return

        clicked_boxes.sort(
            key=lambda b: (b["x2"] - b["x1"]) * (b["y2"] - b["y1"])
        )
        clicked_box = clicked_boxes[0]
        clicked_box["removed"] = not clicked_box["removed"]

        ann = clicked_box["annotation"]
        status = "REMOVED" if clicked_box["removed"] else "KEPT"
        print(f'{image_path.name} | instance_id={ann["instance_id"]} -> {status}')

        draw_state(ax, image, boxes, image_path.name)

    fig.canvas.mpl_connect("button_press_event", onclick)
    plt.show()


if __name__ == "__main__":
    main()