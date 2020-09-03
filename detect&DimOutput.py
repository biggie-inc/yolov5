import argparse
import os
import platform
import shutil
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords,
    xyxy2xywh, plot_one_box, strip_optimizer, set_logging, hypotenuse)
from utils.torch_utils import select_device, load_classifier, time_synchronized


def detect(save_img=False):
    out, source, weights, view_img, save_txt, imgsz = \
        opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source.isnumeric() or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    # Tailgate dimension dict
    tailgate_dims = []

    # Initialize
    set_logging()
    device = select_device(opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)
            print(pred)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            save_path = str(Path(out) / Path(p).name)
            txt_path = str(Path(out) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                
                handles_ymax = []
                #handles_xmid = []
                handles_ymid = []
                handle_mids = []

                tailgates_ymin = []
                tailgates_ymax = []
                tailgate_ythird_coord = []

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det): #coords, confidence, classes
                    
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format

                    if save_img or view_img:  # Add bbox to image
                        #label = '%s %.2f' % (names[int(cls)], conf) #confidence not needed
                        label = '%s ' % (names[int(cls)])
                        coord1, coord2, dim_label = plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
                        
                        # get important points for line drawing
                        # if int(cls) == 1: #handles
                        #     ymax = max(coord1[1], coord2[1])
                        #     handles_ymax.append(ymax)

                        #     xmid = int((coord1[0] + coord2[0]) / 2)
                        #     ymid = int((coord1[1] + coord2[1]) / 2)
                        #     handle_mids.append([xmid, ymid])
                            #cv2.circle(im0, (xmid,ymax), 8, (255,0,0), -1)
                        
                        if int(cls) == 0: #tailgates
                            tailgate_xmin = min(coord1[0], coord2[0])

                            ymax = max(coord1[1], coord2[1])
                            tailgates_ymax.append(ymax)
                            ymin = min(coord1[1], coord2[1])
                            tailgates_ymin.append(ymin)
                            tailgate_ythird = int(abs(coord1[1]-coord2[1])/3+ymin)
                            tailgate_ythird_coord.append([tailgate_xmin, tailgate_ythird])
                            tailgate_dims.append(dim_label)
                
                # added ability to measure between bottom of handle and bottom of tailgate if handle in top 1/3
                # for i, (handle_mid, max_point) in enumerate(zip(handle_mids, handles_ymax)): 
                #     hyps = [hypotenuse(handle_mid, b) for b in tailgate_ythird_coord]
                #     closest_index = np.argmin(hyps)

                #     if handle_mid[1] < tailgate_ythird_coord[closest_index][1]:
                #         min_dist_tg = min([int(abs(max_point - x)) for x in tailgates_ymax])
                #         start_point = (handle_mid[0], handles_ymax[i])
                #         end_point = (handle_mid[0], handles_ymax[i] + min_dist_tg)
                #         cv2.line(im0, start_point, end_point, (100,100,0), 4)
                #         line_mid = int((start_point[1] + end_point[1])/2)
                #         cv2.putText(im0, label, (start_point[0], line_mid), 0, 1, [0, 0, 0], 
                #                     thickness=2, lineType=cv2.LINE_AA)

                ### Previous ability to measure between bottom of handle and tailgate --- was not robust.
                ### Keeping until determined not needed
                # for i, (mid_point, max_point)  in enumerate(zip(handles_ymid, handles_ymax)):
                #     print(f'\nmidpoint: {mid_point}')
                #     min_y_dist = min([int(abs(mid_point - x)) for x in tailgate_ythird]) # gets min distance from handle midpoint to tailgate third
                #     print(f'min y dist: {min_y_dist}')
                #     print(f'tailgate third: {tailgate_ythird}')
                #     min_dist_third = min([x for x in tailgate_ythird if abs(x - min_y_dist) in handles_ymid])
                #     print(f'min_dist_third: {min_dist_third}')
                #     if mid_point < min_dist_third: #handle mid point in top 1/3 of truck
                #         min_dist_tg = min([int(abs(max_point - x)) for x in tailgates_ymax])
                #         print(f'min_dist_tg {min_dist_tg}')
                #         start_point = (handles_xmid[i], handles_ymax[i])
                #         print(f'start point: {start_point}')
                #         end_point = (handles_xmid[i], handles_ymax[i] + min_dist_tg)
                #         print(f'end point: {end_point}')
                #         cv2.line(im0, start_point, end_point, (100,100,0), 4)
                #         label = f'Distance: {min_dist_tg/300:.4f}"L'
                #         line_mid = int((start_point[1] + end_point[1])/2)
                #         cv2.putText(im0, label, (start_point[0], line_mid), 0, 1, [0, 0, 0], 
                #                     thickness=2, lineType=cv2.LINE_AA)


            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))

            # Stream results
            if view_img:
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fourcc = 'mp4v'  # output video codec
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        print('Results saved to %s' % Path(out))
        if platform.system() == 'Darwin' and not opt.update:  # MacOS
            os.system('open ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))
    return tailgate_dims[0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
