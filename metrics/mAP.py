from utils import to_percentage_string, to_box3d_package

from pioneer.common.IoU3d import matrixIoU
from pioneer.das.api.categories import get_name_color

from ignite.metrics.metric import Metric

import numpy as np
import matplotlib.pyplot as plt
import torch
import time
import yaml


class mAP(Metric):

    def __init__(self, cfg, iou_threshold=0.25, distance_bins=[0,np.inf]):
        self.cfg = cfg
        self.output_transform = lambda x: x
        self.iou_threshold = iou_threshold
        self.distance_bins = distance_bins if isinstance(distance_bins, list) else np.append(np.arange(**distance_bins), distance_bins['stop'])
        self.nb_distance_bins = len(self.distance_bins)-1
        self.nb_categories = len(self.cfg['PREPROCESSING']['BOX_3D']['CLASSIFICATION'])
        super(mAP, self).__init__()

    def reset(self):
        self.total_number_gt_boxes = np.zeros((self.nb_categories, self.nb_distance_bins), dtype=int)
        self.categories = np.empty(0, dtype=int)
        self.confidences = np.empty(0)
        self.true_positives = np.empty(0, dtype=bool)
        self.distances = np.empty(0)

    def update(self, output):
        # s=time.time()
        raw_pred, raw_true = output

        # Some boxes might have been lost in preprocessing. We get them back here.
        raw_true, lost_gt = raw_true
        lost_gt = np.array(lost_gt.cpu())

        for i_batch in range(raw_pred.shape[0]):
            pred, true = to_box3d_package(raw_pred[i_batch], self.cfg), to_box3d_package(raw_true[i_batch], self.cfg, is_ground_truth=True)

            # Put back the lost boxes (if any) in the array.
            lost_gt_batch = lost_gt[i_batch]
            for i_lost in np.where(lost_gt_batch[:,0] != 0)[0]:
                box = lost_gt_batch[i_lost]
                true['data'] = np.append(true['data'], np.array((box[[0,1,2]], box[[3,4,5]], [0,0,box[6]], box[7], 0, 0), dtype=true['data'].dtype))
                true['confidence'] = np.append(true['confidence'], 1)

            # We keep track of the gt boxes that have been detected to avoid having multiple true positives for a single gt box.
            already_paired_true = np.full(true['data'].size, False)

            # The gt boxes are counted for each distance bin and category
            for i_bin, distance_bin_min in enumerate(self.distance_bins[:-1]):
                distance_bin_max = self.distance_bins[i_bin+1]
                true_in_bin = self.in_distance_bin(true['data'], distance_bin_min, distance_bin_max)
                categories, counts = np.unique(true['data']['classes'][true_in_bin], return_counts=True)
                self.total_number_gt_boxes[categories, i_bin] += counts


            for category in range(self.nb_categories):

                pred_in_category = np.where(pred['data']['classes'] == category)[0]
                true_in_category = np.where(true['data']['classes'] == category)[0]

                iou_matrix = matrixIoU(
                    [pred['data']['c'][pred_in_category], pred['data']['d'][pred_in_category], 'z', pred['data']['r'][:,2][pred_in_category]], 
                    [true['data']['c'][true_in_category], true['data']['d'][true_in_category], 'z', true['data']['r'][:,2][true_in_category]]
                )

                for i, i_pred in enumerate(pred_in_category):
                    self.categories = np.append(self.categories, category)
                    self.confidences = np.append(self.confidences, pred['confidence'][i_pred])
                    self.distances = np.append(self.distances, self.get_distances(pred['data'][i_pred]))

                    is_true_positive = False
                    for overlap in np.where(iou_matrix[i] >= self.iou_threshold)[0]:

                        if not already_paired_true[true_in_category[overlap]]:

                            is_true_positive = True
                            already_paired_true[true_in_category[overlap]] = True

                            # In case of true positive, the distance of the prediction is replaced by the true distance
                            # This is to avoid recalls above 100% for individual distance bins. This won't affect the overall AP.
                            self.distances[-1] = self.get_distances(true['data'][true_in_category[overlap]])

                            break

                    self.true_positives = np.append(self.true_positives, is_true_positive)


    def compute(self):
        AP = np.zeros((self.nb_categories, self.nb_distance_bins+1))
        max_recall = np.zeros((self.nb_categories, self.nb_distance_bins+1))
        EDR = np.zeros(self.nb_categories)
        output = {}

        for i_bin, distance_bin_min in enumerate(self.distance_bins):

            if i_bin < self.nb_distance_bins:
                distance_bin_max = self.distance_bins[i_bin+1]
                
            else: # A final iteration is done to output the mAP for the whole distance range
                distance_bin_min = self.distance_bins[0]

            in_bin = np.where((self.distances >= distance_bin_min) & (self.distances < distance_bin_max))

            for category in range(self.nb_categories):

                if i_bin < self.nb_distance_bins:
                    N = self.total_number_gt_boxes[category, i_bin]
                else:
                    N = np.sum(self.total_number_gt_boxes[category])

                indices_category = np.where(self.categories[in_bin] == category)[0]
                decreasing_order = np.argsort(self.confidences[in_bin][indices_category])[::-1]

                tp_cumsum = np.cumsum(self.true_positives[in_bin][indices_category][decreasing_order])
                fp_cumsum = np.cumsum(~self.true_positives[in_bin][indices_category][decreasing_order])

                precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-15)
                recall = tp_cumsum / (N + 1e-15)

                # Ensure precision vs. recall is strictly decreasing
                recall_order = np.argsort(recall)
                recall = recall[recall_order]
                precision = [np.max(precision[recall_order][k:]) for k in range(precision.size)]

                AP[category, i_bin] = np.sum([np.abs(recall[k] - recall[k-1]) * precision[k] for k in range(1, len(indices_category))])
                if recall.size > 0:
                    max_recall[category, i_bin] = np.max(recall)

                if i_bin < self.nb_distance_bins:
                    EDR[category] += AP[category, i_bin]*(distance_bin_max - distance_bin_min)

            bin_key = f'{distance_bin_min:.2f}m to {distance_bin_max:.2f}m'
            output[bin_key] = {f'mAP{int(100*self.iou_threshold)}': to_percentage_string(np.mean(AP[:,i_bin]))}
            
            for i_cat, category in enumerate(self.cfg['PREPROCESSING']['BOX_3D']['CLASSIFICATION'].keys()):
                output[bin_key][category] = f'{to_percentage_string(AP[i_cat,i_bin])} (R: {to_percentage_string(max_recall[i_cat,i_bin])})'
        
        output['EDR'] = {}
        for i_cat, category in enumerate(self.cfg['PREPROCESSING']['BOX_3D']['CLASSIFICATION'].keys()):
            output['EDR'][category] = f'{EDR[i_cat]:.2f}m'

        return output


    def in_distance_bin(self, boxes, dmin, dmax):
        distances = self.get_distances(boxes)
        return np.where((distances >= dmin) & (distances < dmax))[0]

    def get_distances(self, boxes):
        if boxes.size == 0:
            return np.empty(0)
        try:
            return (boxes['c'][:,0]**2+boxes['c'][:,1]**2+boxes['c'][:,2]**2)**0.5
        except:
            return (boxes['c'][0]**2+boxes['c'][1]**2+boxes['c'][2]**2)**0.5


    @staticmethod
    def make_plot(results):

        bins_str = list(results.keys())[:-2]
        mAP_key = list(results[bins_str[0]].keys())[0]
        iou_th = float(mAP_key[3:])
        categories = list(results[bins_str[0]].keys())[1:]

        APs_global = np.zeros(len(categories))
        APs = np.zeros((len(categories), len(bins_str)))
        EDRs = np.zeros(len(categories))
        for ic, category in enumerate(categories):
            for ib, b in enumerate(bins_str):
                APs[ic,ib] = float(results[b][category].split('%')[0])
            EDRs[ic] = float(results['EDR'][category][:-1])
            APs_global[ic] = float(results[list(results.keys())[-2]][category].split('%')[0])

        bins = np.array([float(b.split('m to ')[0]) for b in bins_str] + [float(bins_str[-1].split('m to ')[1][:-1])])
        bin_centers = bins[:-1]+(bins[1:]-bins[:-1])/2
        bin_widths = bins[1:]-bins[:-1]

        for ib, b in enumerate(bins_str):
            for ic in np.argsort(-APs[:,ib]):
                label = f'{categories[ic]} (AP{int(iou_th)}: {APs_global[ic]}%, EDR: {EDRs[ic]}m)' if ib == 0 else None
                color = np.array(get_name_color('toynet', ic)[1])/255
                plt.bar(bin_centers[ib], APs[ic,ib], width=0.9*bin_widths[ib], label=label, color=color)

        plt.title(f"mAP{int(iou_th)}: {np.mean(APs_global):.2f}%")
        plt.ylim(0,100)
        plt.xlabel('distance [m]')
        plt.ylabel('AP [%]')
        plt.legend()
        plt.show()
