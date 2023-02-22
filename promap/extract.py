import numpy as np
import skimage
import pandas as pd
import h5py


def extract_trace(image_file_path,
                  pick_file_path,
                  drift_file_path,
                  spot_num,
                  pixels=0,
                  all_spots=False):
    '''
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

        spot_num (int):
            - the group number of the localizations to extract a trace from

        pixels (int):
            - the radius of a grid around the center pixel to include in the
                intensity value measurement for each frame
            - ex. pixels = 1 measures a 3x3 grid, and sums all 9 values to get
                the intensity measurement for each frame

        all_spots (bool):
            - if True extracts a trace from every detected spot in the image

    Returns:
        trace (1D array):
            The intensity value of a single structure, measured for each frame
            of the movie, corrected for any xy drift
    '''

    image = _read_image(image_file_path)
    picked_spots = _read_hdf5(pick_file_path)
    drift = pd.read_csv(drift_file_path, sep=' ')

    if all_spots is True:
        num_spots = picked_spots['group'].max()
        trace = np.zeros((image.shape[2], num_spots))
        for i in range(num_spots):
            trace[:, i] = _single_spot(image, picked_spots, drift, i, pixels)

    else:
        trace = _single_spot(image, picked_spots, drift, spot_num, pixels)

    return trace


def _single_spot(image, picked_spots, drift, spot_num, pixels):

    single_spot = picked_spots[picked_spots['group'] == spot_num]

    single_spot_array = np.asarray(single_spot)
    drift_array = np.asarray(drift)
    undrifted_cords = np.zeros((single_spot_array.shape[0], 3))

    # convert picked spot into un-drift-corrected, coordinates
    detected_frames = single_spot_array[:, 0].astype(int)
    displacements = drift_array[detected_frames, :]
    undrifted_cords[:, 1] = single_spot_array[:, 1] + displacements[:, 0]
    undrifted_cords[:, 2] = single_spot_array[:, 2] + displacements[:, 1]
    undrifted_cords[:, 0] = detected_frames

    # interpolate the position of structure between localizations / events
    total_frames = np.arange(0, image.shape[2])
    xs = np.interp(total_frames, undrifted_cords[:, 0],
                   undrifted_cords[:, 1]).astype(int)
    ys = np.interp(total_frames, undrifted_cords[:, 0],
                   undrifted_cords[:, 2]).astype(int)

    # extract intensity of drfiting structure for all frames
    x_list, y_list = array_list(image, xs, ys, pixels)

    image_crop = image[y_list, x_list, total_frames]

    trace = np.sum(image_crop, axis=0)

    return trace


def _read_hdf5(file):
    temp = h5py.File(file)
    structured_array = np.asarray(temp['locs'])

    return pd.DataFrame(structured_array)


def _read_image(file_path):
    img = skimage.io.imread(file_path)
    np_img = np.array(img)
    np_img = np.moveaxis(np_img, 0, 2)

    return np_img


def array_list(image, xs, ys, pixels):
    x_list = []
    y_list = []
    for x in range(-pixels, pixels+1):
        for y in range(-pixels, pixels+1):
            x_list.append(xs + x)
            y_list.append(ys + y)

    return x_list, y_list
