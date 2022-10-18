import numpy as np
import skimage
import pandas as pd
import h5py


def _read_hdf5(file):
    temp = h5py.File(file)
    #temp.keys()
    structured_array = np.asarray(temp['locs'])

    return pd.DataFrame(structured_array)

def _read_image(file_path):
    img = skimage.io.imread(file_path)
    np_img = np.array(img)
    np_img = np.moveaxis(np_img, 0, 2)
    
    return np_img

def extract_trace(image_file_path,
            pick_file_path,
            drift_file_path,
            spot_num):
    ''' 
    Extracts an intensity trace from a DNA-PAINT movie
    
    Args:
        iamge_file_path (string):
            a
        
        pick_file_path (string):
            a
        
        drfit_file_path (string):
            a
        
    Returns:
        trace: 
        
    '''
    
    image = _read_image(image_file_path)
    picked_spots = _read_hdf5(pick_file_path)
    drift = pd.read_csv(drift_file_path, sep=' ')
    
    # pick a single spot, can alter to return list of traces for multiple spots
    single_spot = picked_spots[picked_spots['group'] == spot_num]
    
    single_spot_array = np.asarray(single_spot)
    drift_array = np.asarray(drift)
    undrifted_cords = np.zeros((single_spot_array.shape[0],3))
    
    # convert picked spot into un-drift-corrected, coordinates
    detected_frames = single_spot_array[:,0].astype(int)
    displacements = drift_array[detected_frames,:]
    undrifted_cords[:,1] = single_spot_array[:,1] + displacements[:,0]
    undrifted_cords[:,2] = single_spot_array[:,2] + displacements[:,1]
    undrifted_cords[:,0] = detected_frames
    
    # interpolate the position of structure between localizations / events
    total_frames = np.arange(0, image.shape[2])
    xs = np.interp(total_frames, undrifted_cords[:,0],
                   undrifted_cords[:,1]).astype(int)
    ys = np.interp(total_frames, undrifted_cords[:,0],
                   undrifted_cords[:,2]).astype(int)
    
    
    # extract intensity of drfiting structure for all frames
    # TODO: - measure more than single pixel area?
    #       - integrate and convert to photon count?
    trace = image[ys, xs, total_frames]
    
    return trace


if __name__ == '__main__':
    image_file_path = '../../Images/Picasso_practice/w1-02_Pm2-8nt-5nM_p4pc-8nd_exp400_tirf2020-1.tif'
    pick_file_path = '../../Images/Picasso_practice/w1-02_Pm2-8nt-5nM_p4pc-8nd_exp400_tirf2020-1_locs_picked.hdf5'
    drift_file_path = '../../Images/Picasso_practice/w1-02_Pm2-8nt-5nM_p4pc-8nd_exp400_tirf2020-1_locs_221014_110734_drift.txt'
    
    
    trace = extract_trace(image_file_path, pick_file_path, drift_file_path, spot_num=9)
