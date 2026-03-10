from pathlib import Path
import json

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Button


DATASET_DIR = Path("../data/dataset")
IMAGES_DIR = DATASET_DIR / "images"
COLORED_LABELS_DIR = DATASET_DIR / "labels_colored"
APPROVED_LABELS_DIR = DATASET_DIR / "labels_colored_approved"

APPROVED_LABELS_DIR.mkdir(exist_ok=True)

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


def save_current_review(image_path: Path, boxes):
    kept_annotations = [
        box["annotation"]
        for box in boxes
        if not box["removed"]
    ]

    output_path = APPROVED_LABELS_DIR / f"{image_path.stem}.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(kept_annotations, f, indent=2)

    removed_count = sum(box["removed"] for box in boxes)
    print(
        f"Saved {output_path.name} | kept={len(kept_annotations)} | removed={removed_count}"
    )


class Reviewer:
    def __init__(self):
        self.image_files = sorted(
            [p for p in IMAGES_DIR.iterdir() if p.suffix.lower() in IMAGE_EXTS]
        )
        if not self.image_files:
            raise RuntimeError(f"No images found in {IMAGES_DIR}")

        self.index = 0
        self.image = None
        self.boxes = None

        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        plt.subplots_adjust(bottom=0.12)

        button_ax = self.fig.add_axes([0.82, 0.02, 0.12, 0.06])
        self.next_button = Button(button_ax, "Next")
        self.next_button.on_clicked(self.on_next)

        self.fig.canvas.mpl_connect("button_press_event", self.onclick)

        self.load_current_image()
        plt.show()

    def load_current_image(self):
        image_path = self.image_files[self.index]
        self.image, self.boxes = load_image_and_annotations(image_path)
        self.draw_state()

    def draw_state(self):
        image_path = self.image_files[self.index]

        self.ax.clear()
        self.ax.imshow(self.image)
        self.ax.set_title(
            f"[{self.index + 1}/{len(self.image_files)}] {image_path.name}\n"
            "Click boxes to remove/keep. Press Next to save and continue."
        )
        self.ax.axis("off")

        for box in self.boxes:
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
            self.ax.add_patch(rect)

            self.ax.text(
                box["x1"],
                max(0, box["y1"] - 5),
                f'id={ann["instance_id"]}',
                color=edgecolor,
                fontsize=9,
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1),
            )

        plt.draw()

    def onclick(self, event):
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return

        clicked_boxes = [
            box for box in self.boxes
            if point_in_box(event.xdata, event.ydata, box)
        ]

        if not clicked_boxes:
            return

        # Prefer the smallest box if boxes overlap
        clicked_boxes.sort(
            key=lambda b: (b["x2"] - b["x1"]) * (b["y2"] - b["y1"])
        )
        clicked_box = clicked_boxes[0]
        clicked_box["removed"] = not clicked_box["removed"]

        ann = clicked_box["annotation"]
        status = "REMOVED" if clicked_box["removed"] else "KEPT"
        image_path = self.image_files[self.index]
        print(f'{image_path.name} | instance_id={ann["instance_id"]} -> {status}')

        self.draw_state()

    def on_next(self, event):
        image_path = self.image_files[self.index]
        save_current_review(image_path, self.boxes)

        self.index += 1
        if self.index >= len(self.image_files):
            print("Review finished.")
            plt.close(self.fig)
            return

        self.load_current_image()


if __name__ == "__main__":
    Reviewer()