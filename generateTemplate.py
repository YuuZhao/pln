from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import os

import cv2
import torch
import numpy as np

from pysot.core.config import cfg
#from config import cfg
#from pysot.models.model_builder import ModelBuilder
from model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker
from pysot.utils.bbox import get_axis_aligned_bbox
from pysot.utils.model_load import load_pretrain
from toolkit.datasets import DatasetFactory
from toolkit.utils.region import vot_overlap, vot_float2str

from os.path import realpath, dirname, join

from pysot.utils.bbox import get_axis_aligned_bbox, cxy_wh_2_rect,overlap_ratio



parser = argparse.ArgumentParser(description='siamrpn tracking')
parser.add_argument('--dataset', default='VOT2016', type=str,
        help='datasets')
parser.add_argument('--config', default='../experiments/siamrpn_r50_l234_dwxcorr/config.yaml', type=str,
        help='config file')
parser.add_argument('--snapshot', default='../experiments/siamrpn_r50_l234_dwxcorr/model.pth', type=str,
        help='snapshot of models to eval')
parser.add_argument('--video', default='', type=str,
        help='eval one special video')
parser.add_argument('--vis', action='store_true',
        help='whether visualzie result')
args = parser.parse_args()

torch.set_num_threads(1)


setfile = 'train_dataset'
temp_path = setfile+'_vot2016'
if not os.path.isdir(temp_path):
    os.makedirs(temp_path)

reset = 1
frames_of_each_video = 20

def main():
    # load config
    cfg.merge_from_file(args.config)

    # create model
    model = ModelBuilder()

    # load model
    model = load_pretrain(model, args.snapshot).cuda().eval()

    # build tracker
    tracker = build_tracker(model)

    #model_name = args.snapshot.split('/')[-1].split('.')[0]
    #total_lost = 0

    #cur_dir = os.path.dirname(os.path.realpath(__file__))

    #dataset_root = os.path.join(cur_dir, '../testing_dataset', args.dataset)

    video_path = '/home/yuuzhao/Documents/project/pysot/testing_dataset/VOT2016'
    #lists = open('/home/lichao/tracking/LaSOT_Evaluation_Toolkit/sequence_evaluation_config/' + setfile + '.txt', 'r')
    #list_file = [line.strip() for line in lists]

    category = os.listdir(video_path)
    category.sort()

    # create dataset
    #dataset = DatasetFactory.create_dataset(name=args.dataset,dataset_root=dataset_root,load_img=False)

    template_acc = [];template_cur = []
    init0 = [];init = [];pre = [];gt = []  # init0 is reset init

    print("Category & Video:")
    for tmp_cat in category:
        tmp_cat_path = temp_path + '/' + tmp_cat
        if not os.path.isdir(tmp_cat_path):
            os.makedirs(tmp_cat_path)

        print("Category:",tmp_cat)
        video = os.listdir(join(video_path, tmp_cat));
        video.sort()
        #video_cut = video[0:frames_of_each_video]
        frame = 0;

        #for picture in video_cut:  # 这个循环或许该去掉
        #    print("Frame:", picture)
        gt_path = join(video_path, tmp_cat, 'groundtruth.txt')

        ground_truth = np.loadtxt(gt_path, delimiter=',')
        # num_frames = len(ground_truth);  # num_frames = min(num_frames, frame_max)
        num_frames = frames_of_each_video
        # print("num_frames: ",num_frames)
        img_path = join(video_path, tmp_cat);
        # print("imgpath",img_path)
        imgFiles = [join(img_path, '%08d.jpg') % i for i in range(1, num_frames + 1)]

        while frame < num_frames:
                print("frame:", frame)
                Polygon = ground_truth[frame]
                cx, cy, w, h = get_axis_aligned_bbox(Polygon)
                gt_rect = [cx, cy, w, h]

                image_file = imgFiles[frame]
                # target_pos, target_sz = np.array([cx, cy]), np.array([w, h])
                img = cv2.imread(image_file)  # HxWxC

                if frame == 0:
                    tracker.init(img, gt_rect)
                if w * h != 0:
                    # image_file = imgFiles[frame]
                    # target_pos, target_sz = np.array([cx, cy]), np.array([w, h])
                    # img = cv2.imread(image_file)  # HxWxC
                    zf_acc = tracker.get_zf(img, gt_rect)

                    output = tracker.track(img)
                    pre_rect = output['bbox']
                    zf_pre = tracker.get_zf(img, pre_rect)

                    template_acc.append(zf_acc)
                    template_cur.append((zf_pre))

                    print("ACC&PRE")
                    init0.append(0);
                    init.append(frame);
                    frame_reset = 0;
                    pre.append(0);
                    gt.append(1)
                    while frame < (num_frames - 1):
                        print("while ", frame, "<", num_frames)
                        frame = frame + 1;
                        frame_reset = frame_reset + 1
                        image_file = imgFiles[frame]
                        if not image_file:
                            break

                        Polygon = ground_truth[frame]
                        cx, cy, w, h = get_axis_aligned_bbox(Polygon)
                        gt_rect = [cx, cy, w, h]

                        img = cv2.imread(image_file)  # HxWxC
                        zf_acc = tracker.get_zf(img, gt_rect)

                        output = tracker.track(img)
                        pre_rect = output['bbox']
                        zf_pre = tracker.get_zf(img, pre_rect)

                        # print("zf_pre:",zf_pre.shape)
                        # print("zf_acc:",zf_acc.shape)
                        # pdb.set_trace()
                        template_acc.append(zf_acc);
                        template_cur.append(zf_pre)
                        init0.append(frame_reset);
                        init.append(frame);
                        pre.append(1);
                        if frame == (num_frames - 1):  # last frame
                            print("if frame == num_frames-1")
                            gt.append(0)
                        else:
                            gt.append(1)

                        pre_rect_arr = np.array(pre_rect)
                        cx, cy, w, h = get_axis_aligned_bbox(pre_rect_arr)
                        target_pos, target_siz = np.array([cx, cy]), np.array([w, h])

                        res = cxy_wh_2_rect(target_pos, target_siz)

                        if reset:
                            cx, cy, w, h = get_axis_aligned_bbox(ground_truth[frame])
                            gt_rect = [cx, cy, w, h]
                            gt_rect = np.array(gt_rect)
                            iou = overlap_ratio(gt_rect, res)
                            if iou <= 0:
                                break
                else:
                    print("else")
                    template_acc.append(torch.zeros([1, 3, 127, 127], dtype=torch.float32));
                    template_cur.append(torch.zeros([1, 3, 127, 127], dtype=torch.float32));
                    init0.append(0);
                    init.append(frame);
                    pre.append(1);
                    if frame == (num_frames - 1):  # last frame
                        gt.append(0)
                    else:
                        gt.append(1)
                frame = frame + 1  # skip

        #写出一次
        #print("template_acc:",template_acc)
        #print("template_cur:",template_cur)
        #print("init:", init)
        #print("init0:",init0)
        #print("pre:",pre)

        #template_acc_con = np.concatenate(template_acc);
        #template_cur_con = np.concatenate(template_cur)

        print("write for each video")
        np.save(tmp_cat_path + '/template', template_acc);
        np.save(tmp_cat_path + '/templatei', template_cur)
        np.save(tmp_cat_path + '/init0', init0);
        np.save(tmp_cat_path + '/init', init);
        np.save(tmp_cat_path + '/pre', pre);
        np.save(tmp_cat_path + '/gt', gt);
    print("template")

    #template_acc_con = np.concatenate(template_acc);
    #template_cur_con = np.concatenate(template_cur)
    #np.save(temp_path + '/template', template_acc);
    #np.save(temp_path + '/templatei', template_cur)
    #np.save(temp_path + '/init0', init0);
    #np.save(temp_path + '/init', init);
    #np.save(temp_path + '/pre', pre);
    #np.save(temp_path + '/gt', gt);



if __name__ == '__main__':
    main()