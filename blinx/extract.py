import h5py
import numpy as np
import pandas as pd
import skimage
import funlib.geometry as fg
from tqdm import tqdm


def extract_traces(
    image_file_path,
    pick_file_path,
    drift_file_path,
    spot_size=0
):
    """
    Extracts an intensity trace from a DNA-PAINT movie

    Args:
        image_file_path (string):
            - path to a tiff movie of blinking events

        pick_file_path (string):
            - path to a .hdf5 file containing a list of localizations grouped
                grouped by picks (a cluster of localizations all originating
                from the same structure)
            - generated through picasso render

        drift_file_path (string):
            - a .txt file containg the drift correction values for each frame
                of the tiff movie
            - generated as an output of picasso localize, running undrift RCC,
                or undrift from picks

        spot_size (int):
            - the radius of a grid around the center pixel to include in the
                intensity value measurement for each frame
            - ex. spot_size = 1 measures a 3x3 grid, and sums all 9 values to get
                the intensity measurement for each frame

    Returns:
        trace (1D array):
            The intensity value of a single structure, measured for each frame
            of the movie, corrected for any xy drift
    """

    image_sequence = _read_image(image_file_path)
    picked_spots = _read_hdf5(pick_file_path)
    drifts = np.array(pd.read_csv(drift_file_path, sep=" ", header=None))
    num_spots = picked_spots["group"].max()

    context_size = int(1.5 * spot_size)
    background_size = int(3 * spot_size)

    # all ROIs centered at (0, 0)
    spot_roi = fg.Roi(
        (-spot_size, -spot_size),
        (2 * spot_size + 1, 2 * spot_size + 1)
    )
    context_roi = fg.Roi(
        (-context_size, -context_size),
        (2 * context_size + 1, 2 * context_size + 1)
    )
    background_roi = fg.Roi(
        (-background_size, -background_size),
        (2 * background_size + 1, 2 * background_size + 1)
    )

    # image_sequence: (x, y, t)
    length = image_sequence.shape[2]
    total_frames = np.arange(0, length, dtype=np.int32)

    traces = []
    backgrounds = []

    for spot_num in tqdm(range(num_spots), "spots"):

        # spot_data: rows of (frame, x_coordinate, y_coordinate, ...)
        spot_data = np.array(picked_spots[picked_spots["group"] == spot_num])

        detected_frames = spot_data[:, 0].astype(np.int32)
        displacements = drifts[detected_frames, :]

        # keep only coordinates and correct for drift
        spot_locations = spot_data[:, 1:3] + displacements
        # (x, y) -> (y, x)
        spot_locations = spot_locations[:, ::-1]

        # interpolate between detected frames
        interpolated_spot_locations = np.array([
            np.interp(
                total_frames,
                detected_frames,
                spot_locations[:, 0]
            ),
            np.interp(
                total_frames,
                detected_frames,
                spot_locations[:, 1]
            )]).T

        trace = []
        background = []
        for t in tqdm(range(length), "frames", leave=False):

            # TODO: use rounded version
            # spot_location = np.round(interpolated_spot_locations[t]).astype(np.int32)
            spot_location = interpolated_spot_locations[t].astype(np.int32)
            offset = fg.Coordinate(spot_location)
            shifted_spot_roi = spot_roi.shift(offset)
            shifted_context_roi = context_roi.shift(offset)
            shifted_background_roi = background_roi.shift(offset)

            spot_crop = image_sequence[shifted_spot_roi.to_slices() + (t,)]
            context_crop = image_sequence[shifted_context_roi.to_slices() + (t,)]
            background_crop = image_sequence[shifted_background_roi.to_slices() + (t,)]

            bg = np.sum(background_crop) - np.sum(context_crop)
            # normalize background values to match sum of crop
            bg *= (
                spot_crop.size /
                (background_crop.size - context_crop.size)
            )

            trace.append(np.sum(spot_crop))
            background.append(bg)

        traces.append(np.array(trace))
        backgrounds.append(np.array(background))

    return np.array(traces), np.array(backgrounds)


def _read_hdf5(file):
    temp = h5py.File(file)
    structured_array = np.asarray(temp["locs"])

    return pd.DataFrame(structured_array)


def _read_image(file_path):
    img = skimage.io.imread(file_path)
    np_img = np.array(img)
    np_img = np.moveaxis(np_img, 0, 2)

    return np_img


def array_list(image, xs, ys, pixels):
    x_list = []
    y_list = []
    for x in range(-pixels, pixels + 1):
        for y in range(-pixels, pixels + 1):
            x_list.append(xs + x)
            y_list.append(ys + y)

    return x_list, y_list
