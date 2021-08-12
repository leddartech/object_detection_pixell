from object_detection_pixell.augmentation import box3d_augment

from pioneer.common.IoU3d import matrixIoU
from pioneer.das.api import datatypes

import numba
import numpy as np
import torch


def preprocess_box3d(box3d_sample, cfg, data_augmentation_state=None):

    boxes3d = box3d_sample.raw['data']
    category_names = np.array(box3d_sample.label_names())

    if 'FILTER_OCCLUDED' in cfg['PREPROCESSING']['BOX_3D']:
        if cfg['PREPROCESSING']['BOX_3D']['FILTER_OCCLUDED']:
            boxes3d, category_names = filter_occluded(boxes3d, category_names, box3d_sample, occlusion_threshold=2)

    if data_augmentation_state is not None:
        if 'INSERT' in data_augmentation_state:
            for insert in data_augmentation_state['INSERT']:
                boxes3d = np.append(boxes3d, insert['box'])
                category_names = np.append(category_names, insert['category_name'])
        if data_augmentation_state['FLIP_LR']:
            boxes3d = box3d_augment.global_flip_lr(boxes3d)
        if 'TRANSLATION' in data_augmentation_state:
            boxes3d = box3d_augment.global_translation(boxes3d, data_augmentation_state['TRANSLATION'])
        if 'ROTATION' in data_augmentation_state:
            boxes3d = box3d_augment.global_rotation(boxes3d, data_augmentation_state['ROTATION'])
        if 'SCALING' in data_augmentation_state:
            boxes3d = box3d_augment.global_scaling(boxes3d, data_augmentation_state['SCALING'])

    boxes3d = convert_category_numbers(
        boxes3d=boxes3d, 
        original_names=category_names, 
        classification=cfg['PREPROCESSING']['BOX_3D']['CLASSIFICATION'],
    )

    grid = cfg['PREPROCESSING']['BOX_3D']['GRID']
    nb_categories = len(cfg['PREPROCESSING']['BOX_3D']['CLASSIFICATION'])

    target_array, lost_gt = create_box3d_target(
        boxes3d,
        nb_categories = nb_categories,
        grid_shape = (grid['x'][2], grid['y'][2]),
        x_step = (grid['x'][1] - grid['x'][0])/grid['x'][2],
        y_step = (grid['y'][1] - grid['y'][0])/grid['y'][2],
        z_mid = (grid['z'][1] - grid['z'][0])/2,
        x_min = grid['x'][0],
        y_min = grid['y'][0],
        x_max = grid['x'][1],
        y_max = grid['y'][1],
    )

    return (target_array, lost_gt)

def filter_occluded(boxes3d, category_names, box3d_sample, occlusion_threshold=2):
    try:
        occlusions = box3d_sample.attributes()['occlusions']
        keep = np.where(occlusions < occlusion_threshold)[0]
        return boxes3d[keep], category_names[keep]
    except:
        return boxes3d, category_names

def convert_category_numbers(boxes3d, original_names, classification):
    boxes3d_coverted = boxes3d.copy()
    keep = np.full(len(boxes3d), False)
    for i, new_name in enumerate(classification):
        for original_name in classification[new_name]:
            to_convert = np.where(original_names == original_name)[0]
            boxes3d_coverted['classes'][to_convert] = i
            keep[to_convert] = True
    return boxes3d_coverted[keep]

def create_box3d_target(boxes3d, nb_categories, grid_shape, x_step, y_step, z_mid, x_min, y_min, x_max, y_max):

    # Initialize target arrays
    offsets = np.zeros((2, grid_shape[0], grid_shape[1]))
    height = np.zeros((1, grid_shape[0], grid_shape[1]))
    sizes = np.zeros((3, grid_shape[0], grid_shape[1]))
    angle = np.zeros((1, grid_shape[0], grid_shape[1]))
    heading = np.zeros((1, grid_shape[0], grid_shape[1]))
    heatmap = np.zeros((nb_categories, grid_shape[0], grid_shape[1]))

    # Center of boxes
    cx = (boxes3d['c'][:,0]-x_min)/x_step
    cy = (boxes3d['c'][:,1]-y_min)/y_step
    i_cx = np.floor(cx).astype(int)
    i_cy = np.floor(cy).astype(int)

    # Filter out boxes outside BEV grid
    keep = np.where((i_cx >= 0) & (i_cx < grid_shape[0]) & (i_cy >= 0) & (i_cy < grid_shape[1]))[0]
    boxes3d, cx, cy, i_cx, i_cy = boxes3d[keep], cx[keep], cy[keep], i_cx[keep], i_cy[keep]

    # Box properties at center point
    offsets[0,i_cx,i_cy] = cx - i_cx
    offsets[1,i_cx,i_cy] = cy - i_cy
    height[0,i_cx,i_cy] = boxes3d['c'][:,2] - z_mid
    sizes[0,i_cx,i_cy] = boxes3d['d'][:,0]/x_step
    sizes[1,i_cx,i_cy] = boxes3d['d'][:,1]/y_step
    sizes[2,i_cx,i_cy] = boxes3d['d'][:,2]/(2*z_mid)
    angle[0,i_cx,i_cy] = (boxes3d['r'][:,2] + np.pi/2)%(np.pi) - np.pi/2
    heading[0,i_cx,i_cy] = (abs(boxes3d['r'][:,2]) > np.pi/2).astype(int)
    heatmap[boxes3d['classes'], i_cx, i_cy] = 1

    # Keep track of lost boxes (up to 10) for later evaluation.
    lost_gt_map = np.zeros((nb_categories, grid_shape[0], grid_shape[1]))
    lost_gt = np.zeros((10,8))
    for i in range(min([10, boxes3d.size])):
        lost_gt_map[boxes3d['classes'][i], i_cx[i], i_cy[i]] += 1
        if lost_gt_map[boxes3d['classes'][i], i_cx[i], i_cy[i]] > 1:
            lost_gt[i,[0,1,2]] = boxes3d[i]['c']
            lost_gt[i,[3,4,5]] = boxes3d[i]['d']
            lost_gt[i,6] = boxes3d[i]['r'][2]
            lost_gt[i,7] = boxes3d[i]['classes']

    # All points of the BEV grid
    x = np.arange(x_min, x_max, x_step)
    y = np.arange(y_min, y_max, y_step)
    mesh = np.meshgrid(x, y)
    xx = mesh[0].flatten()
    yy = mesh[1].flatten()
    grid_points = np.vstack([xx,yy]).T

    # Rotation matrices for all boxes
    r = boxes3d['r'][:,2]
    R = np.array([[np.cos(r), -np.sin(r)], [np.sin(r), np.cos(r)]])

    # Corners for all boxes
    # corner1 = boxes3d['c'][:,[0,1]] + np.diagonal((np.array([ 1, 1])*boxes3d['d'][:,[0,1]]/2) @ R, axis1=2).T
    corner2 = boxes3d['c'][:,[0,1]] + np.diagonal((np.array([-1, 1])*boxes3d['d'][:,[0,1]]/2) @ R, axis1=2).T
    corner3 = boxes3d['c'][:,[0,1]] + np.diagonal((np.array([ 1,-1])*boxes3d['d'][:,[0,1]]/2) @ R, axis1=2).T
    corner4 = boxes3d['c'][:,[0,1]] + np.diagonal((np.array([-1,-1])*boxes3d['d'][:,[0,1]]/2) @ R, axis1=2).T

    # Base vectors
    u, v = (corner2 - corner4).T, (corner3 - corner4).T

    # Grid points projected on the base vectors
    pu, pv = np.matmul(grid_points,u), np.matmul(grid_points,v)

    # Corners projected on the base vectors
    c2u = np.diag(np.dot(corner2,u))
    c4u = np.diag(np.dot(corner4,u))
    c3v = np.diag(np.dot(corner3,v))
    c4v = np.diag(np.dot(corner4,v))

    # Find the points that are inside each boxes
    i_pt, i_box = np.where((pu < c2u) & (pu > c4u) & (pv < c3v) & (pv > c4v))
    ix = ((grid_points[i_pt,0]-x_min)/x_step).astype(int)
    iy = ((grid_points[i_pt,1]-y_min)/y_step).astype(int)

    # For each point that is found inside a box, take the 4 closest neighbors to account for partially filled grid cells
    i_pt = np.hstack([i_pt,i_pt,i_pt,i_pt])
    i_box = np.hstack([i_box,i_box,i_box,i_box])
    ix_m1 = np.clip(ix-1,0,grid_shape[0]-1)
    iy_m1 = np.clip(iy-1,0,grid_shape[1]-1)
    ix = np.hstack([ix,ix,ix_m1,ix_m1])
    iy = np.hstack([iy,iy_m1,iy,iy_m1])

    # Find the distance (in nb of cells) from each point inside a box to its center
    nb_cells_to_center = ((ix - i_cx[i_box])**2 + (iy - i_cy[i_box])**2)**0.5
    nb_cells_to_center[nb_cells_to_center==1] = 1.25 #to differentiate distance=1 from distance=0
    nb_cells_to_center[nb_cells_to_center==0] = 1
    
    # By ordering like that, if multiple values are assigned to a single cell, only the highest is kept.
    order = np.argsort(nb_cells_to_center)[::-1]

    heatmap[boxes3d['classes'][i_box][order], ix[order], iy[order]] = 1/nb_cells_to_center[order]

    return (np.vstack([offsets, height, sizes, angle, heading, heatmap]), lost_gt)

def to_box3d_package(raw, cfg, is_ground_truth=False):
        
    if raw.ndim == 4:
        raw = raw[0] # Batch size = 1 for inference

    if not is_ground_truth: #final activations
        raw[0:2] = torch.sigmoid(raw[0:2])
        raw[3:6] = torch.relu(raw[3:6])
        raw[7:8] = torch.sigmoid(raw[7:8])
        raw[8:] = torch.sigmoid(raw[8:])

    hotspots = find_hotspots(raw[8:])

    raw = raw.detach().cpu().numpy()
    hotspots = hotspots.detach().cpu().numpy()

    grid = cfg['PREPROCESSING']['BOX_3D']['GRID']

    boxes3d, confidences = reconstruct_box3d_from_array(
        array=raw,
        hotspots = hotspots,
        confidence_threshold=cfg['POSTPROCESSING']['CONFIDENCE_THRESHOLD'],
        max_nb_detections=cfg['POSTPROCESSING']['MAX_NUMBER_DETECTIONS'], 
        x_step=(grid['x'][1] - grid['x'][0])/grid['x'][2], 
        y_step=(grid['y'][1] - grid['y'][0])/grid['y'][2], 
        x_min=grid['x'][0], 
        y_min=grid['y'][0], 
        z_mid=(grid['z'][1] - grid['z'][0])/2,
        x_max=grid['x'][1], 
        y_max=grid['y'][1],
    )

    boxes3d = np.array(boxes3d, dtype=datatypes.box3d())

    if cfg['POSTPROCESSING']['NON_MAX_SUPPRESSION'] and len(boxes3d) > 1 and not is_ground_truth:

        keep = non_maximum_suppression(
            iou_matrix=matrixIoU([boxes3d['c'], boxes3d['d'], 'z', boxes3d['r'][:,2]]),
            iou_threshold=cfg['POSTPROCESSING']['IOU_THRESHOLD_FOR_NMS'], 
            confidences=confidences,
        )
        boxes3d = boxes3d[keep]

    return {'data':boxes3d, 'confidence':confidences}

@numba.njit
def reconstruct_box3d_from_array(array, hotspots, confidence_threshold, max_nb_detections, x_step, y_step, x_min, y_min, z_mid, x_max, y_max):
    """This function does the opposite transformation that of create_box3d_target()"""

    offsets = array[0:2]
    height = array[2:3]
    sizes = array[3:6]
    angle = array[6:7]
    heading = array[7:8]
    # heatmap = array[8:]

    package = []
    box_confidences = np.empty(0)

    order = np.argsort(-hotspots[:,0])[:max_nb_detections]

    for hotspot in hotspots[order]:

        confidence, category, ix, iy = hotspot
        ix, iy = int(ix), int(iy)

        if confidence < confidence_threshold:
            break
        
        x = (ix + offsets[0,ix,iy])*x_step + x_min
        y = (iy + offsets[1,ix,iy])*y_step + y_min
        z = height[0,ix,iy] + z_mid

        sx = sizes[0,ix,iy] * x_step
        sy = sizes[1,ix,iy] * y_step
        sz = sizes[2,ix,iy] * 2*z_mid
        
        rx = 0
        ry = 0
        rz = (angle[0,ix,iy] + heading[0,ix,iy]*np.pi)%(2*np.pi)

        box = ([x,y,z],[sx,sy,sz],[rx,ry,rz],category,0,0)

        package.append(box)
        box_confidences = np.append(box_confidences, confidence)

    return package, box_confidences

MAXPOOL = torch.nn.MaxPool2d(3, stride=1, padding=1)
def find_hotspots(heatmap):
    local_maxima = MAXPOOL(heatmap)
    match = (heatmap == local_maxima) & (heatmap > 0)
    confidences = heatmap[match]
    indices = torch.where(match)
    return torch.stack((confidences, *indices), dim=1)

@numba.njit
def non_maximum_suppression(iou_matrix, iou_threshold, confidences):
    iou_matrix -= np.eye(iou_matrix.shape[0])
    keep = []
    for i, confidence in enumerate(confidences):
        overlaps = np.where(iou_matrix[i] > iou_threshold)[0]
        if (confidence > confidences[overlaps]).all():
            keep.append(i)
    return keep
