import matplotlib.pyplot as plt
from promap import extract

if __name__ == '__main__':
    image_file_path = 'examples/example_image.tif'
    pick_file_path = 'examples/locs_picked.hdf5'
    drift_file_path = 'examples/drift.txt'

    trace = extract.extract_trace(image_file_path,
                                  pick_file_path,
                                  drift_file_path,
                                  spot_num=9)

    plt.plot(trace[1000:2000])
