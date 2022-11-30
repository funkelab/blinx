import matplotlib.pyplot as plt
from promap import extract

if __name__ == '__main__':
    image_file_path = '../../Images/1116_exp/8spot_B100_400ms_8nt_3lzr_1.tif'
    pick_file_path = '../../Images/1116_exp/8spot_B100_400ms_8nt_3lzr_1_locs_picked.hdf5'
    drift_file_path = '../../Images/1116_exp/8spot_B100_400ms_8nt_3lzr_1_locs_221116_160808_drift.txt'

    trace = extract.extract_trace(image_file_path,
                                  pick_file_path,
                                  drift_file_path,
                                  spot_num=50,
                                  pixels=2)

    plt.plot(trace[1000:2000])