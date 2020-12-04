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



def apply_brightness_contrast(input_img, brightness = 0, contrast = 0):
    
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow)/255
        gamma_b = shadow
        
        buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()
    
    if contrast != 0:
        f = 131*(contrast + 127)/(127*(131-contrast))
        alpha_c = f
        gamma_c = 127*(1-f)
        
        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

    return buf


def auto_canny(gray_image, sigma=0.66):
    #compute median of single channel pixel intentsities
    med = np.median(gray_image)

    #apply automatic canny edge detection using median and sigma
    lower = int(max(0, (1.0-sigma)*med))
    upper = int(min(255, (1.0+sigma)*med))
    edged = cv2.Canny(gray_image, lower, upper)

    return edged


def draw_dist_btm_h_to_btm_t(image, handle_mids, handles_ymax, tailgates_ymax, tailgate_ythird_coord, px_ratio):
#  ability to measure between bottom of handle and bottom of tailgate if handle in top 1/3
    for i, (handle_mid, max_point) in enumerate(zip(handle_mids, handles_ymax)): 
        hyps = [hypotenuse(handle_mid, b) for b in tailgate_ythird_coord]
        closest_index = np.argmin(hyps) # gets index of closest point via hypotenuse

        if handle_mid[1] < tailgate_ythird_coord[closest_index][1]: # if midpoint of handle is in top 1/3 of tailgate
            min_dist_tg = min([int(abs(max_point - x)) for x in tailgates_ymax]) # if multiple handles found, finds closest tailgate
            start_point = (handle_mid[0], handles_ymax[i]) # start point for drawn line
            end_point = (handle_mid[0], handles_ymax[i] + min_dist_tg) # end point for drawn line
            cv2.line(image, start_point, end_point, (100,100,0), 4)
            line_mid = int((start_point[1] + end_point[1])/2) # mid point for text
            label = f'Distance: {((end_point[1] - start_point[1]) / px_ratio):.4f}"'
            cv2.putText(image, label, (start_point[0], line_mid), 0, 1, [0, 0, 0], 
                        thickness=2, lineType=cv2.LINE_AA)
            return start_point
        else:
            return False

def tailgate_masked(image, brightness, contrast, kernel):
    contrast = apply_brightness_contrast(image.copy(), brightness=brightness, contrast=contrast)
    bilat = cv2.bilateralFilter(contrast.copy(),9,75,75)  #gaussian blur faster than bilateralFilter

    edges = auto_canny(bilat)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,kernel) 
    dilated = cv2.dilate(edges.copy(), kernel) #dilating edges to connect segments

    edges2 = auto_canny(dilated.copy()) #getting edges of dilated image

    cnts, _ = cv2.findContours(edges2.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    max_contour = max(cnts, key=cv2.contourArea) #largest contour (hopefully the tailgate)
    hull = cv2.convexHull(max_contour)

    # drawn_ctrs = cv2.drawContours(image.copy(), [hull], -1, (0, 255, 0), 1) #can be removed, just for show

    BGRA = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2BGRA)

    masked = cv2.drawContours(BGRA.copy(), [hull], -1, (0,0,0,0), -1)

    masked_image = cv2.bitwise_and(BGRA, masked)

    # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15,10))
    # ax1.imshow(drawn_ctrs)
    # ax1.axis('off')
    # ax2.imshow(masked_image)
    # ax2.axis('off')
    # plt.tight_layout();

    return masked_image


def handle_masked(image, brightness, contrast):
    # This function needs to use whatever bright/contrast levels for the tg
    adjusted = apply_brightness_contrast(image.copy(), brightness=brightness, contrast=contrast)
    bilat = cv2.bilateralFilter(adjusted.copy(),9,75,75)  #gaussian blur faster than bilateralFilter
    
    edges = auto_canny(bilat.copy())

    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)) 
    # dilated = cv2.dilate(edges.copy(), kernel) #dilating edges to connect segments

    # edges5 = cv2.Canny(dilated.copy(), 100, 200) #getting edges of dilated image

    cnts, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    max_contour = max(cnts, key=cv2.contourArea) #largest contour (hopefully the handle)
    hull = cv2.convexHull(max_contour)

    # drawn_ctrs = cv2.drawContours(image.copy(), [hull], -1, (0, 255, 0), 1) #can be removed, just for show

    BGRA = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGBA)

    mask = np.zeros(BGRA.shape, BGRA.dtype)
    cv2.fillPoly(mask, [hull], (255,)*BGRA.shape[2], )

    masked_image = cv2.bitwise_and(BGRA, mask)

    # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15,10))
    # ax1.imshow(drawn_ctrs)
    # ax1.axis('off')
    # ax2.imshow(masked_image)
    # ax2.axis('off')
    # plt.tight_layout();

    return masked_image


def final_truck(image, transp_tg, transp_h, tg_coords, h_coords):
    final_image = image.copy()

    tg_y1, tg_y2, tg_x1, tg_x2  = tg_coords
    h_y1, h_y2, h_x1, h_x2 = h_coords

    final_image[tg_y1:tg_y2, tg_x1:tg_x2] = transp_tg

    if transp_h:
        final_image[h_y1:h_y2, h_x1:h_x2] = transp_h

    return final_image


def detect(save_img=False):
    out, source, weights, view_img, save_txt, imgsz = \
        opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source.isnumeric() or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

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

        # Process detections
        for i, det in enumerate(pred):  # detections per img
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            img_crops = im0s.copy() # creates a copy making img_crops for cropped versions and im0 separate for bbox version

            out_path = str(Path(out))    
            file_name = str(Path(p).name).split('.')[0] # gets name of file without extension
            save_path = str(Path(out) / Path(p).name)
            txt_path = str(Path(out) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                
                handles_ymax = []
                #handles_xmid = []
                handle_mids = []

                tailgates_ymin = []
                tailgates_ymax = []
                tailgate_ythird_coord = []

                brightness = 25
                contrast = 65
                px_ratio = 1

                crop_coords = {}

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                det_sorted = sorted(det, key=lambda x: x[-1]) # sort detected items by last index which is class
                
                # Write results
                for *xyxy, conf, cls in reversed(det_sorted): #coords, confidence, classes.... reversed for some reason? But actually helpful since plate is cls 2
                    x1, y1, x2, y2 = int(xyxy[0]),int(xyxy[1]),int(xyxy[2]),int(xyxy[3])

                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format


                    if int(cls) == 3: #truck cropping (future development; requires retraining with a truck class)
                        img_crops = img_crops[y1:y2, x1:x2]

                    elif int(cls) == 2: # license plate
                        license_width = abs(int(x2 - x1))
                        px_ratio = license_width / 12   # number of pixels per inch as license plates are 12"
                        # im_p = img_crops[y1:y2, x1:x2]
                        # cv2.imwrite(f'{out_path}/{file_name}_p_edge.png', im_p)
                    
                    elif int(cls) == 1: #handle
                        print(f'handle y1,y2,x1,x2: {y1},{y2},{x1},{x2}')
                        im_h = img_crops[y1:y2, x1:x2]
                        crop_coords['h'] = [y1,y2,x1,x2]

                        cv2.imwrite(f'{out_path}/{file_name}_h_edge.png', im_h)
                        
                    elif int(cls) == 0: #tailgate
                        im_t = img_crops[y1:y2, x1:x2]
                        print(f'tailgate y1,y2,x1,x2: {y1},{y2},{x1},{x2}')
                        crop_coords['tg'] = [y1,y2,x1,x2]

                        cv2.imwrite(f'{out_path}/{file_name}_t_edge.png', im_t)

                    if save_img or view_img:  # Add bbox to image
                        #label = '%s %.2f' % (names[int(cls)], conf) #confidence not needed
                        label = '%s ' % (names[int(cls)])
                        coord1, coord2, dim_label = plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], 
                                                                line_thickness=3, px_ratio=px_ratio)
                        
                        #get important points for line drawing
                        if int(cls) == 1 and int(abs(y1-y2)) < 175: #handle
                            coord1, coord2, dim_label = plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], 
                                                                    line_thickness=3, px_ratio=px_ratio)
                            ymax = max(coord1[1], coord2[1])
                            handles_ymax.append(ymax)

                            xmid = int((coord1[0] + coord2[0]) / 2)
                            ymid = int((coord1[1] + coord2[1]) / 2)
                            handle_mids.append([xmid, ymid])
                            
                            im_h = im0[coord1[0]:coord2[0], coord1[1]:coord2[1]]
                        
                        elif int(cls) == 0: #tailgate
                            coord1, coord2, dim_label = plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], 
                                                                    line_thickness=3, px_ratio=px_ratio)
                            tailgate_xmin = min(coord1[0], coord2[0])

                            ymax = max(coord1[1], coord2[1])
                            tailgates_ymax.append(ymax)
                            ymin = min(coord1[1], coord2[1])
                            tailgates_ymin.append(ymin)
                            tailgate_ythird = int(abs(coord1[1]-coord2[1])/3+ymin)
                            tailgate_ythird_coord.append([tailgate_xmin, tailgate_ythird])

                            im_t = img[coord1[0]:coord2[0], coord1[1]:coord2[1]]
                        
                        else:
                            pass
                            
                # added ability to measure between bottom of handle and bottom of tailgate if handle in top 1/3
                # for i, (handle_mid, max_point) in enumerate(zip(handle_mids, handles_ymax)): 
                #     hyps = [hypotenuse(handle_mid, b) for b in tailgate_ythird_coord]
                #     closest_index = np.argmin(hyps) # gets index of closest point via hypotenuse

                #     if handle_mid[1] < tailgate_ythird_coord[closest_index][1]: # if midpoint of handle is in top 1/3 of tailgate
                #         min_dist_tg = min([int(abs(max_point - x)) for x in tailgates_ymax]) # if multiple handles found, finds closest tailgate
                #         start_point = (handle_mid[0], handles_ymax[i]) # start point for drawn line
                #         end_point = (handle_mid[0], handles_ymax[i] + min_dist_tg) # end point for drawn line
                #         cv2.line(im0, start_point, end_point, (100,100,0), 4)
                #         line_mid = int((start_point[1] + end_point[1])/2) # mid point for text
                #         label = f'Distance: {((end_point[1] - start_point[1]) / px_ratio):.4f}"'
                #         cv2.putText(im0, label, (start_point[0], line_mid), 0, 1, [0, 0, 0], 
                #                     thickness=2, lineType=cv2.LINE_AA)

        
                

                # function draws and labels the distance from bottom of handle to bottom of tailgate
                # if handle in top 1/3 of tailgate
                # returns the y coord of handle bottom if so, else returns False
                adj_tailgate_top = draw_dist_btm_h_to_btm_t(img_crops, handle_mids, handles_ymax, 
                                                            tailgates_ymax, tailgate_ythird_coord, px_ratio)

                if adj_tailgate_top:
                    print(adj_tailgate_top)
                    print(crop_coords['tg'][0])
                    if adj_tailgate_top > crop_coords['tg'][0]:
                        crop_coords['tg'][0] = int(adj_tailgate_top)
                        transp_h = handle_masked(im_h, brightness, contrast)
                else:
                    transp_h = 0
                    pass

            
                #function gets the handle surrounded by transparency
                

                transp_tg = tailgate_masked(im_t, brightness, contrast, (3,3))

                final_image = final_truck(img_crops, transp_tg, transp_h, crop_coords['tg'], crop_coords['h'])

                cv2.imwrite(f'{out_path}/{file_name}_transparency.png', final_image)



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
