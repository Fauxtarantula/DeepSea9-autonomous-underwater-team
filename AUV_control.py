"""
Run inference on images, videos, directories, streams, etc.

Usage:
    $ python path/to/detect.py --source path/to/img.jpg --weights yolov5s.pt --img 640
    
List of to-do: 
    killswitch - Implemented
    pid - Implemented #technically, it is part of the detect
    angle probem(always keep changing configurations) - Not Implemented
    Thread:Sonar - Not Implemented, Compass - Implemented (has a hardcoded timer to ensure data was recorded), Get_err_angl - Not Implemented
    Shave off reflection from top - Not Implemented
    Confidence-level filter -Not Implemented
    
Current threads:
    Reference angle thread - start_thread
    Continuous compass thread - thread1
"""


import argparse
import os
import sys
from pathlib import Path
import threading

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import time
import queue
import RPi.GPIO as GPIO
import serial

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.experimental import attempt_load
from utils.datasets import LoadImages, LoadStreams
from utils.general import apply_classifier, check_img_size, check_imshow, check_requirements, check_suffix, colorstr, \
    increment_path, non_max_suppression, print_args, save_one_box, scale_coords, set_logging, \
    strip_optimizer, xyxy2xywh
from utils.plots import Annotator, colors
from utils.torch_utils import load_classifier, select_device, time_sync
from ctrl_mod.motor import FORWARD, RIGHT_TURN, LEFT_TURN, stop
#from ctrl_mod.get_dir import FORWARD, RIGHT_TURN, LEFT_TURN, stop
from ctrl_mod.PID import PID

from ctrl_mod.compass import get_angle2, q, compass_pausable, q1, get_start_angle, start_q, get_desired, \
     desire_q
from Algorithms.decide import direction

M =0
SpeedNowL = 65
SpeedNowR = 65
PErr_last =0
IntErr =0
err_now = 30
kp=7
ki=0.2
kd=0.25
global scenario #scenarios will play out the diff possible reasons of no detetction

#Initialize sonar and killswitch. Note that the sabertooth and compass does not need to be initialize as 
#it has alr initialize in its respective module scripts.
GPIO.setmode(GPIO.BCM)
GPIO.setup(22,GPIO.IN)
#ser = serial.Serial("/dev/ttyUSB0",9600) #may change
ser2=serial.Serial('/dev/ttyACM0', 9600, timeout = 1)
ser2.reset_input_buffer()
#global test
#test = 0
            

@torch.no_grad()
def run(weights=ROOT / 'gate.pt',  # model.pt path(s)
        source=0, #ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
        imgsz=640,  # inference size (pixels)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        #thread1_time = 6, #use for multithread
        ):
    
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(device)
    half &= device.type != 'cpu'  # half precision only supported on CUDA
    #test = 0 #use for testing killswitch
    #ser2.write(72) #72 to initialize, 76 to kill motors 

    # Load model
    w = str(weights[0] if isinstance(weights, list) else weights)
    classify, suffix, suffixes = False, Path(w).suffix.lower(), ['.pt', '.onnx', '.tflite', '.pb', '']
    check_suffix(w, suffixes)  # check weights have acceptable suffix
    pt, onnx, tflite, pb, saved_model = (suffix == x for x in suffixes)  # backend booleans
    stride, names = 64, [f'class{i}' for i in range(1000)]  # assign defaults
    if pt:
        model = torch.jit.load(w) if 'torchscript' in w else attempt_load(weights, map_location=device)
        stride = int(model.stride.max())  # model stride
        names = model.module.names if hasattr(model, 'module') else model.names  # get class names
        if half:
            model.half()  # to FP16
        if classify:  # second-stage classifier
            modelc = load_classifier(name='resnet50', n=2)  # initialize
            modelc.load_state_dict(torch.load('resnet50.pt', map_location=device)['model']).to(device).eval()

    # Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        bs = len(dataset)  # batch_size
        scenario = 0
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    #Initialize and get the reference angle
    thread_start = threading.Thread(target = get_start_angle, args = [30])
    
    #Start threads
    thread_start.start()
    #time.sleep(3)
    #
    
    #save thread values here
    ref_ang = start_q.get()
    #

    thread_start.join()
    #
    thread_desire = threading.Thread(target = get_desired, args = [30])
    thread_desire.start()
    desired_ang = desire_q.get()
    thread_desire.join()
    ang_type = 0
    
    # Run inference
    if pt and device.type != 'cpu':
        model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.parameters())))  # run once
    dt, seen = [0.0, 0.0, 0.0], 0
    for path, img, im0s, vid_cap in dataset:
        t1 = time_sync()
        if onnx:
            img = img.astype('float32')
        else:
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        if pt:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(img, augment=augment, visualize=visualize)[0]
        
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3

        # Second-stage classifier (optional)
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            
            seen += 1
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], f'{i}: ', im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            
            
            while not q1.empty():#to flush out previous value of queue if due to thread problem.
                q1.get()
                
            if scenario == 1 or scenario == 2:
                ang_type = ref_ang #this is to change the angle to the ref angle when detected
            else:
                ang_type = desired_ang
                
            #thread1 = threading.Thread(target = compass_pausable, args = [ref_ang])
            thread2 = threading.Thread(target = compass_pausable, args = [ang_type])
            thread_send = threading.Thread(target = sendbytes)
            #thread_killswitch = threading.Thread(target = killswitch)
            #thread1.start()
            thread2.start()
            thread_send.start()
            #thread_killswitch.start()
            
            #time.sleep(1)
            thread_send.join()
            #thread_killswitch.join()
            killswitch()

            if len(det):
                
                #err_now_ref = q1.get()
                err_now_desire = q1.get()
                print(err_now_desire)
                
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                #gonna pull a big brain move here
                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    #get coordinates of the bounding boxes
                    c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                    center_point = round((c1[0]+c2[0])/2), round((c1[1]+c2[1])/2)
                    circle = cv2.circle(im0,center_point,5,(0,255,0),2)
                    
                    #start a nitpicking session here
                    for c in det[:, -1].unique(): 
                        
                        n = (det[:, -1] == c).sum()  # detections per class
                        g = f"{names[int(c)]}" #g is the label of the object detected
                                  
    
                    
                        if(g =="Gate"): #get the gate detection only, ignore rest. Note to change to gate
                            if round((c1[0]+c2[0])/2) > 360: #added more arguments
                                
                                #err_now += desired_ang #+ go right, -go left
                                err_now_desire = abs(err_now_desire)
                                M = pid2(err_now_desire,kp, ki, kd)
                                RIGHT_TURN(SpeedNowL,SpeedNowR,M)
                                scenario = 1
                                
                                 
                            else: 
                                if round((c1[0]+c2[0])/2) < 280:
                                    err_now_desire = abs(err_now_desire)
                                    M = pid2(err_now_desire,kp,ki,kd)
                                    LEFT_TURN(SpeedNowL,SpeedNowR,M)
                                    scenario = 2
                                
                                
                                else:      #change placement
                                    FORWARD(SpeedNowL,SpeedNowR)
                                    #scenario= 1
                                    
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                        if save_crop:
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # Print time (inference-only)
            print(f'{s}Done. ({t3 - t2:.3f}s)')
            
            thread2.join()
            #print(thread1.is_alive())
            
            
            if not len(det): #check if detection is available
                #err_now_ref = q1.get()
                err_now_desire = q1.get()
                #print(scenario)
                
                #whle thng here to move forward
                if(scenario == 0):
                    
                    #print(err_now_desire)
                    if direction(err_now_desire) ==0:#change here
                        err_now_desire = abs(err_now_desire)
                        print(err_now_desire)
                        M = pid2(err_now_desire,kp, ki, kd)
                        RIGHT_TURN(SpeedNowL, SpeedNowR, M)
                        #scenario = 1#test
                        
                    elif direction(err_now_desire)==1:
                        FORWARD(SpeedNowL, SpeedNowR)
                    else:
                        M = pid2(err_now_desire,kp, ki, kd)
                        LEFT_TURN(SpeedNowL, SpeedNowR, M)
                        #scenario =2#test
                    
                    
                    #scenario +=1 #may make use of scenarios here
                   # print("T")
                    
                elif(scenario == 1): #This sceneario will play out when gate was detected but soon lose the detection(move 5 degrees or smth)
                    #need to change to desired here
                    newSpeedNowR = 65
                    err_now_desire = abs(err_now_desire)
                    M = pid2(err_now_desire,kp, ki, kd)
                    LEFT_TURN(SpeedNowL, newSpeedNowR, M)
                    #scenario = 0
                    #scenario+=1
                    #print("L")
                
                elif(scenario == 2):#This scenario will play out when gate was detected but soon lose detection(left)
                    #same here
                    newSpeedNowL = 65
                    err_now_desire = abs(err_now_desire)
                    M = pid2(err_now_desire,kp ,ki ,kd)
                    RIGHT_TURN(newSpeedNowL,SpeedNowR, M)
                    #scenario = 0
                    #print("C")
                else:
                    stop()#need to find a way around this
            

            # Stream results
            im0 = annotator.result()
            #time.sleep(t1_time)
            
            if view_img:
                #print(imgsz[0])
                cv2.imshow(str(p), im0)
                #cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    print(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'gate.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default=0, help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(FILE.stem, opt)
    return opt

def sendbytes():
    ser2.write(b'H')
    #time.sleep(1)
    
    
def killswitch():
    time.sleep(1)
    line = ser2.readline().decode('utf-8').rstrip()
    if(line == "stopped"):
        #stop()
        stop()
        sys.exit(0)

def pid2(error, kp, ki, kd):
    
    global PErr_last
    global IntErr
    
    M_t =0
    PErr_now = (error/360) *100# calculation to get our proportional error
    DErr = PErr_now - PErr_last# calculation to get our differential error
    IntErr += PErr_now# calculation to get our inegral error

    M_t = ( kp * PErr_now + ki * IntErr + kd * DErr) # calculation to get our percentage change in speed needed to fix error
    
    # if statement to cap our percentage change at 100%
    if M_t>100:
        M_t=100
    if M_t<-100:
        M_t=-100
    
    PErr_last= PErr_now
    
    return abs(M_t)

    
def main(opt):
    #check_requirements(exclude=('tensorboard', 'thop')) #fuck this shit. takes too long
    #t1 = threading.Thread(target=(run(**vars(opt))))
    #t2 = threading.Thread(target=tryout, args = [7])
    #t1.start()
    #t2.start()
    #t1.join()
    #t2.join()
    
    run(**vars(opt))
    


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
