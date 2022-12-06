import matplotlib.pyplot as plt
from promap import extract
from promap import fit

if __name__ == '__main__':
    # image_file_path = '../../Images/1116_exp/8spot_B100_400ms_8nt_3lzr_1.tif'
    # pick_file_path = '../../Images/1116_exp/8spot_B100_400ms_8nt_3lzr_1_locs_picked.hdf5'
    # drift_file_path = '../../Images/1116_exp/8spot_B100_400ms_8nt_3lzr_1_locs_221116_160808_drift.txt'
    
    image_file_path = '../../promap/scripts/examples/example_image.tif'
    pick_file_path = '../../promap/scripts/examples/locs_picked.hdf5'
    drift_file_path = '../../promap/scripts/examples/drift.txt'

    trace = extract.extract_trace(image_file_path,
                                  pick_file_path,
                                  drift_file_path,
                                  spot_num=8,
                                  pixels=2)

    plt.plot(trace[1000:2000])
    
    y = 4
    
    likelihood, p_on, p_off, mu, sigma = fit.optimize_params(y, trace,
                                                             mu_guess = 2000.,
                                                             mu_b_guess=5000.,
                                                             mu_lr=5)
    