import cv2
import numpy as np
import os
import os.path as osp
import pandas as pd
import datetime
import psutil

path = '~/Data/AIC19/'
og_fps = 10


def get_bbox(type='gt', det_time='train', fps=5):
    # type = ['gt','det','labeled']
    data_path = osp.join(osp.expanduser(path), 'test' if det_time == 'test' else 'train')
    save_path = osp.join(osp.expanduser('~/Data/AIC19/ALL_{}_bbox/'.format(type)), det_time)

    if type == 'gt':
        save_path = osp.join(save_path, 'gt_bbox_{}_fps'.format(fps))
        fps_pooling = int(og_fps / fps)  # use minimal number of gt's to train ide model
    elif type == 'labeled':
        save_path = osp.join(save_path, 'det_labeled_bbox_{}_fps'.format(fps))
        fps_pooling = int(og_fps / fps)  # use minimal number of gt's to train ide model

    if not osp.exists(save_path):  # mkdir
        if not osp.exists(osp.dirname(save_path)):
            if not osp.exists(osp.dirname(osp.dirname(save_path))):
                os.mkdir(osp.dirname(osp.dirname(save_path)))
            os.mkdir(osp.dirname(save_path))
        os.mkdir(save_path)

    # scene selection for train/val
    if det_time == 'train':
        scenes = ['S03', 'S04']
    elif det_time == 'trainval':
        scenes = ['S01', 'S03', 'S04']
    elif det_time == 'val':
        scenes = ['S01']
    else:  # test
        scenes = os.listdir(data_path)

    for scene_dir in scenes:
        scene_path = osp.join(data_path, scene_dir)

        for camera_dir in os.listdir(scene_path):
            iCam = int(camera_dir[1:])
            # get bboxs
            if type == 'gt':
                bbox_filename = osp.join(scene_path, camera_dir, 'gt', 'gt.txt')
            elif type == 'labeled':
                bbox_filename = osp.join(scene_path, camera_dir, 'det', 'det_ssd512_labeled.txt')
            else:  # det
                bbox_filename = osp.join(scene_path, camera_dir, 'det', 'det_ssd512.txt')
            bboxs = np.array(pd.read_csv(bbox_filename, header=None))
            if type == 'gt' or type == 'labeled':
                bboxs = bboxs[np.where(bboxs[:, 0] % fps_pooling == 0)[0], :]

            # get frame_pics
            video_file = osp.join(scene_path, camera_dir, 'vdo.avi')
            video_reader = cv2.VideoCapture(video_file)
            # frame_pics = []
            frame_num = 0
            success = video_reader.isOpened()
            while (success):
                assert psutil.virtual_memory().percent < 95, "reading video will be killed!!!!!!"
                success, frame_pic = video_reader.read()
                frame_num = frame_num + 1

                bboxs_frame = bboxs[bboxs[:, 0] == frame_num]

                for index in range(len(bboxs_frame)):
                    frame = int(bboxs_frame[index, 0])
                    pid = int(bboxs_frame[index, 1])
                    bbox_left = int(bboxs_frame[index, 2])
                    bbox_top = int(bboxs_frame[index, 3])
                    bbox_width = int(bboxs_frame[index, 4])
                    bbox_height = int(bboxs_frame[index, 5])

                    bbox_bottom = bbox_top + bbox_height
                    bbox_right = bbox_left + bbox_width

                    bbox_pic = frame_pic[bbox_top:bbox_bottom, bbox_left:bbox_right]

                    if type == 'gt' or type == 'labeled':
                        save_file = osp.join(save_path, "{:04d}_c{:02d}_f{:05d}.jpg".
                                             format(pid, iCam, frame))
                    else:
                        same_frame_bbox_count = np.where(bboxs[:index, 0] == frame)[0].size
                        save_file = osp.join(save_path, 'c{:02d}_f{:05d}_{:03d}.jpg'.
                                             format(iCam, frame, same_frame_bbox_count))

                    cv2.imwrite(save_file, bbox_pic)
                    cv2.waitKey(0)

                cv2.waitKey(0)
            video_reader.release()

            print(video_file, 'completed!')
        print(scene_dir, 'completed!')
    print(save_path, 'completed!')


if __name__ == '__main__':
    print('{}'.format(datetime.datetime.today().strftime('%Y-%m-%d_%H-%M-%S')))
    # get_bbox(type='labeled')
    get_bbox()
    # get_bbox(det_time='val', fps=1)
    # get_bbox(type='det', det_time='val')
    # get_bbox(type='det', det_time='trainval')
    # get_bbox(type='det', det_time='test')
    print('{}'.format(datetime.datetime.today().strftime('%Y-%m-%d_%H-%M-%S')))
    print('Job Completed!')
