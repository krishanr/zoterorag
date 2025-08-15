import matplotlib.pyplot as plt
from PIL import Image
import math

def show_thumbnails(data, thumb_size=(128, 128), cols=3):
    """
    Display a list of PIL images as thumbnails in a grid.

    Args:
        images (list[PIL.Image.Image]): List of PIL images.
        thumb_size (tuple): Size of each thumbnail.
        cols (int): Number of columns in the grid.
    """
    images = [img for img, _  in data]
    titles = [title for _, title in data]
    # Resize copies of images to thumbnail size
    thumbs = [img.copy().thumbnail(thumb_size) or img.copy() for img in images]

    rows = (len(images) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols,figsize=(cols * 6, rows * 6))

    # Flatten axes array for easy iteration
    axes = axes.flatten() if rows > 1 or cols > 1 else [axes]

    for ax, img, title in zip(axes, thumbs, titles):
        ax.imshow(img)
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    # Hide unused axes
    for ax in axes[len(images):]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()

    