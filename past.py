import numpy as np
from PIL import Image


def pixel_sorting(image, calculate_value, apply_condition):
    def sort_interval(interval, interval_index):
        return np.argsort(interval) + interval_index

    def process_row(row, row_values):
        interval_indices = np.flatnonzero(condition_mask[row])
        split_values = np.split(row_values, interval_indices)

        for idx, interval in enumerate(split_values[1:]):
            split_values[idx + 1] = sort_interval(interval, interval_indices[idx])

        split_values[0] = np.arange(split_values[0].size)
        merged = np.concatenate(split_values).astype("uint32")

        for c in range(pixels.shape[-1]):
            pixels[row, :, c] = pixels[row, merged, c]

    pixels = np.array(image)
    values = calculate_value(pixels)
    condition_mask = apply_condition(values)

    for row, row_values in enumerate(values):
        process_row(row, row_values)

    return Image.fromarray(pixels)


def luminance(p):
    return np.mean(p, axis=2) / 255.0


def apply_condition(vals):
    return vals > 0.25


img_path = r"C:\Users\Welcome-Pc\Downloads\WhatsApp Image 2026-03-31 at 3.18.57 PM.jpeg"
img = Image.open(img_path)

out = pixel_sorting(img, luminance, apply_condition)
out.save("vintage_sorted.png")

print("Done: vintage_sorted.png")