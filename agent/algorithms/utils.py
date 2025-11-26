import cv2
import numpy as np
from scipy import ndimage

def visualize_q_maps(predictions, color_heightmap, best_pix_ind):
    canvas = None
    num_rotations = predictions.shape[0]
    for canvas_row in range(int(num_rotations/4)):
        tmp_row_canvas = None
        for canvas_col in range(4):
            rotate_idx = canvas_row*4 + canvas_col
            prediction_vis = predictions[rotate_idx,:,:].copy()
            # prediction_vis[prediction_vis < 0] = 0 # assume probability
            # prediction_vis[prediction_vis > 1] = 1 # assume probability
            prediction_vis = np.clip(prediction_vis, 0, 1)
            prediction_vis.shape = (predictions.shape[1], predictions.shape[2])
            prediction_vis = cv2.applyColorMap((prediction_vis*255).astype(np.uint8), cv2.COLORMAP_JET)
            if rotate_idx == best_pix_ind[0]:
                prediction_vis = cv2.circle(prediction_vis, (int(best_pix_ind[2]), int(best_pix_ind[1])), 7, (0,0,255), 2)
            prediction_vis = ndimage.rotate(prediction_vis, rotate_idx*(360.0/num_rotations), reshape=False, order=0)
            background_image = ndimage.rotate(color_heightmap, rotate_idx*(360.0/num_rotations), reshape=False, order=0)
            prediction_vis = (0.5*background_image + 0.5*prediction_vis).astype(np.uint8)
            if tmp_row_canvas is None:
                tmp_row_canvas = prediction_vis
            else:
                tmp_row_canvas = np.concatenate((tmp_row_canvas,prediction_vis), axis=1)
        if canvas is None:
            canvas = tmp_row_canvas
        else:
            canvas = np.concatenate((canvas,tmp_row_canvas), axis=0)

    return canvas