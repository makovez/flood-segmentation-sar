from datetime import datetime
import numpy as np

def today():
    # Get the current timestamp as a datetime object
    current_time = datetime.now()

    # Convert the datetime object to a string
    current_time_str = current_time.strftime("%Y-%m-%dT%H:%M:%SZ")

    return current_time_str


def generate_preview(data):
    # Read the data and apply 10 * log(x) transformation
    ch1, ch2 = data[0], data[1]

    # Replace nan values with 0
    ch1[np.isnan(ch1)] = 0
    ch2[np.isnan(ch2)] = 0

    # Stretch images to minmax with 2% and 98% percentile
    ch1_min, ch1_max = np.percentile(ch1, (2, 98))
    ch2_min, ch2_max = np.percentile(ch2, (2, 98))
    
    ch1_stretched = np.clip((ch1 - ch1_min) / (ch1_max - ch1_min + 1e-8), 0, 1)
    ch2_stretched = np.clip((ch2 - ch2_min) / (ch2_max - ch2_min + 1e-8), 0, 1)

    # Convert to 8-bit for saving as JPEG
    ch1_8bit = (ch1_stretched * 255).astype(np.uint8)
    ch2_8bit = (ch2_stretched * 255).astype(np.uint8)

    # Merge the channels into an RGB image
    rgb_image = np.stack([ch1_8bit, ch2_8bit, np.zeros_like(ch1_8bit)], axis=-1)

    return rgb_image

