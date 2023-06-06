import numpy as np 
from matplotlib import pyplot as plt 
import os 
from nilearn import plotting, datasets
from scipy.stats import pearsonr as corr
from tqdm import tqdm


def calculate_corr(fmri_pred, fmri):
    correlation = np.zeros(fmri_pred.shape[1])
    # Correlate each predicted LH vertex with the corresponding ground truth vertex
    for v in tqdm(range(fmri_pred.shape[1])):
        correlation[v] = corr(fmri_pred[:,v], fmri[:,v])[0]
    return correlation

def get_roi_corr(lh_correlation, rh_correlation, roi_class='prf-visualrois'):
    # Load the ROI classes mapping dictionaries
    roi_mapping_files = [f'mapping_{roi_class}.npy']
    roi_name_maps = []
    for r in roi_mapping_files:
        roi_name_maps.append(np.load(os.path.join(subj_paths.roi_masks, r),
            allow_pickle=True).item())

    # Load the ROI brain surface maps
    lh_challenge_roi_files = [f'lh.{roi_class}_challenge_space.npy']
    rh_challenge_roi_files = [f'rh.{roi_class}_challenge_space.npy']
    lh_challenge_rois = []
    rh_challenge_rois = []
    for r in range(len(lh_challenge_roi_files)):
        lh_challenge_rois.append(np.load(os.path.join(subj_paths.roi_masks,
            lh_challenge_roi_files[r])))
        rh_challenge_rois.append(np.load(os.path.join(subj_paths.roi_masks,
            rh_challenge_roi_files[r])))

    # Select the correlation results vertices of each ROI
    roi_names = []
    lh_roi_correlation = []
    rh_roi_correlation = []
    for r1 in range(len(lh_challenge_rois)):
        for r2 in roi_name_maps[r1].items():
            if r2[0] != 0: # zeros indicate to vertices falling outside the ROI of interest
                roi_names.append(r2[1])
                lh_roi_idx = np.where(lh_challenge_rois[r1] == r2[0])[0]
                rh_roi_idx = np.where(rh_challenge_rois[r1] == r2[0])[0]
                lh_roi_correlation.append(lh_correlation[lh_roi_idx])
                rh_roi_correlation.append(rh_correlation[rh_roi_idx])
    roi_names.append(f'All from {roi_class}')
    lh_roi_correlation.append(np.hstack(lh_roi_correlation))
    rh_roi_correlation.append(np.hstack(rh_roi_correlation))
    print(f'Mean correlation in {roi_class} left hemisphere:', np.mean(lh_roi_correlation[-1]))
    print(f'Mean correlation in {roi_class} right hemisphere:', np.mean(rh_roi_correlation[-1]))
    print(f'Mean correlation in all vericies left hemisphere:', np.mean(lh_correlation))
    print(f'Mean correlation in all vericies right hemisphere:', np.mean(rh_correlation))
    

def plot_corr_all_rois(lh_correlation, rh_correlation, plot_title=''):
    # Load the ROI classes mapping dictionaries
    roi_mapping_files = ['mapping_prf-visualrois.npy', 'mapping_floc-bodies.npy',
    'mapping_floc-faces.npy', 'mapping_floc-places.npy',
    'mapping_floc-words.npy', 'mapping_streams.npy']
    roi_name_maps = []
    for r in roi_mapping_files:
        roi_name_maps.append(np.load(os.path.join(subj_paths.roi_masks, r),
            allow_pickle=True).item())

    # Load the ROI brain surface maps
    lh_challenge_roi_files = ['lh.prf-visualrois_challenge_space.npy',
    'lh.floc-bodies_challenge_space.npy', 'lh.floc-faces_challenge_space.npy',
    'lh.floc-places_challenge_space.npy', 'lh.floc-words_challenge_space.npy',
    'lh.streams_challenge_space.npy']
    rh_challenge_roi_files = ['rh.prf-visualrois_challenge_space.npy',
    'rh.floc-bodies_challenge_space.npy', 'rh.floc-faces_challenge_space.npy',
    'rh.floc-places_challenge_space.npy', 'rh.floc-words_challenge_space.npy',
    'rh.streams_challenge_space.npy']
    lh_challenge_rois = []
    rh_challenge_rois = []
    for r in range(len(lh_challenge_roi_files)):
        lh_challenge_rois.append(np.load(os.path.join(subj_paths.roi_masks,
            lh_challenge_roi_files[r])))
        rh_challenge_rois.append(np.load(os.path.join(subj_paths.roi_masks,
            rh_challenge_roi_files[r])))

    # Select the correlation results vertices of each ROI
    roi_names = []
    lh_roi_correlation = []
    rh_roi_correlation = []
    for r1 in range(len(lh_challenge_rois)):
        for r2 in roi_name_maps[r1].items():
            if r2[0] != 0: # zeros indicate to vertices falling outside the ROI of interest
                roi_names.append(r2[1])
                lh_roi_idx = np.where(lh_challenge_rois[r1] == r2[0])[0]
                rh_roi_idx = np.where(rh_challenge_rois[r1] == r2[0])[0]
                lh_roi_correlation.append(lh_correlation[lh_roi_idx])
                rh_roi_correlation.append(rh_correlation[rh_roi_idx])

    roi_names.append('All vertices')
    lh_roi_correlation.append(lh_correlation)
    rh_roi_correlation.append(rh_correlation)
    


    # Create the plot
    lh_median_roi_correlation = [np.median(lh_roi_correlation[r])
        for r in range(len(lh_roi_correlation))]
    rh_median_roi_correlation = [np.median(rh_roi_correlation[r])
        for r in range(len(rh_roi_correlation))]
    plt.figure(figsize=(18,6))
    x = np.arange(len(roi_names))
    width = 0.30
    plt.bar(x - width/2, lh_median_roi_correlation, width, label='Left Hemisphere')
    plt.bar(x + width/2, rh_median_roi_correlation, width,
        label='Right Hemishpere')
    plt.xlim(left=min(x)-.5, right=max(x)+.5)
    plt.ylim(bottom=0, top=1)
    plt.xlabel('ROIs')
    plt.xticks(ticks=x, labels=roi_names, rotation=60)
    plt.ylabel('Median Pearson\'s $r$')
    plt.legend(frameon=True, loc=1);
    plt.title(plot_title)

def plot_accuracy(correlation, hemisphere):

    # Load the brain surface map of all vertices
    roi_dir = os.path.join(subj_paths.roi_masks,
        hemisphere[0]+'h.all-vertices_fsaverage_space.npy')
    fsaverage_all_vertices = np.load(roi_dir)

    # Map the correlation results onto the brain surface map
    fsaverage_correlation = np.zeros(len(fsaverage_all_vertices))
    if hemisphere == 'left':
        fsaverage_correlation[np.where(fsaverage_all_vertices)[0]] = correlation
    elif hemisphere == 'right':
        fsaverage_correlation[np.where(fsaverage_all_vertices)[0]] = correlation

    # Create the interactive brain surface map
    fsaverage = datasets.fetch_surf_fsaverage('fsaverage')
    view = plotting.view_surf(
        surf_mesh=fsaverage['infl_'+hemisphere],
        surf_map=fsaverage_correlation,
        bg_map=fsaverage['sulc_'+hemisphere],
        threshold=1e-14,
        cmap='cold_hot',
        colorbar=True,
        title='Encoding accuracy, '+hemisphere+' hemisphere'
        )
    return view
    
