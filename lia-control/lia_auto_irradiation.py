from stage_dll_loader import (
    init_stage, move, get_position, set_speed, close_device_safely, wait_for_stop
)

from pypylon import pylon
import cv2 as cv
import numpy as np
from ctypes import *
import time
import os
import csv
import sys
import pyfirmata
from datetime import datetime
import pathlib
import matplotlib.pyplot as plt
from PIL import Image
from threading import Thread
from pynput import keyboard
from scipy import spatial

from ultralytics import YOLO
from pathlib import Path


def append_shutter_event(event_id, opened_ts, closed_ts):
    qc_dir = Path(__file__).parent / "QC"
    qc_dir.mkdir(parents=True, exist_ok=True)
    csv_path = qc_dir / f"{datetime.now():%Y-%m-%d}.csv"

    write_header = (not csv_path.exists()) or (csv_path.stat().st_size == 0)
    with csv_path.open("a", newline="") as csv_file:
        writer = csv.writer(csv_file)
        if write_header:
            writer.writerow(["id", "opened", "colosed"])
        writer.writerow([event_id, opened_ts, closed_ts])

# ==================================================================================================
# User-configurable settings (edit these as needed)
# ==================================================================================================
STAGE_SPEED = 5000                     # stage speed for X (used in set_speed)
EXPOSURE_TIME_SEC = 1.5                  # shutter open time per target (seconds)
STEP_SIZE = 0.33                      # pixels -> stage steps conversion
YOLO_MODEL_PATH = Path(__file__).resolve().parent.parent / "cv_model" / "egg-detector_run_CURTA_A5000_b32_iz800_e200" / "weights" / "best.pt"
YOLO_PARAMS = {                        # YOLO inference kwargs
    "imgsz": 800,
    "conf": 0.75,
    "iou": 0.3,
    "verbose": False,
}
ARDUINO_PORT = "COM3"                  # Arduino serial port
ARDUINO_PIN = 13                       # Arduino pin for shutter control
# ==================================================================================================

######################################################################################################################
### Stage Initialization start ###
######################################################################################################################

if sys.version_info >= (3, 0):
    import urllib.parse

STAGE_XIMC_ROOT = Path(__file__).resolve().parent / "libximc_2.13.2" / "ximc-2.13.3" / "ximc"
STAGE_WRAPPER_PATH = STAGE_XIMC_ROOT / "crossplatform" / "wrappers" / "python"
STAGE_LIBDIR = STAGE_XIMC_ROOT / "win64"
STAGE_KEYFILE_PATH = STAGE_XIMC_ROOT / "win32" / "keyfile.sqlite"

# ---------------- Stage initialization ----------------
lib, device_id1, device_id2 = init_stage(
    wrapper_path=str(STAGE_WRAPPER_PATH),
    libdir=str(STAGE_LIBDIR),
    keyfile_path=str(STAGE_KEYFILE_PATH),
    prefer_virtual=False,   # set True to allow xi-emu virtual device when no hardware is found
    verbose=True,
)
# ---------------------------------------------------------------

def wait_both_axes(interval_ms: int = 50):
    """Block until both axes stop (recommended poll ≈10 ms)."""
    wait_for_stop(lib, device_id1, interval_ms)
    if device_id2 is not None:
        wait_for_stop(lib, device_id2, interval_ms)

######################################################################################################################
### Initialization end ###
######################################################################################################################


def irradiation():
    global gateOpened
    global startTime2
    global img
    global saveImage
    global filename
    global lastKeyPressed
    global board
    global quitMain
    global exposureTime

    shutter_event_id = 1
    pending_opened_ts = None

    # initialize start and finish positions
    startPosX = 13308
    finishPosX = 16000

    # step size (per pixel) got from stage_test.py calibration
    stepSize = STEP_SIZE

    currentPosX, currentUPosX = get_position(lib, device_id1)
    currentPosY, currentUPosY = get_position(lib, device_id2)
    global_Y_center = currentPosY

    frameCount = 1

    # create a directory to save live images
    now = datetime.now()
    year = now.strftime("%Y")
    month = now.strftime("%m")
    day = now.strftime("%d")
    directoryName = "{0}-{1}-{2}".format(year, month, day)
    path = "C:/lab/MBI/img/" + directoryName
    directory = pathlib.Path(path)
    if directory.exists():
        for file_name in os.listdir(path):
            file = path + "/" + file_name
            os.remove(file)
    else:
        os.mkdir(path)

    # load model and set detetction threshold
    model_path = YOLO_MODEL_PATH
    model = YOLO(str(model_path))

    # MAIN-LOOP: repeat whole irradiation process while k not pressed
    while quitMain == False:
        print("Current position: ", currentPosX)
        print("Finish position: ", finishPosX)

        filename = "{0}/frame{1}.jpeg".format(path, frameCount)
        img2 = img.copy()
        cv.line(img2, pt1=(0, 1024), pt2=(2048, 1024), color=(0, 0, 255), thickness=1)
        cv.line(img2, pt1=(1024, 2048), pt2=(1024, 0), color=(0, 0, 255), thickness=1)
        cv.imwrite(filename, img)

        # Use color input for best accuracy
        img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        # YOLOv11 inference (note: kwarg is 'imgsz', not 'isz')
        results = model(img_rgb, **YOLO_PARAMS)  # YOLO11 predict args.

        # Save labeled image (results[0].plot() returns BGR numpy array)
        labeled_bgr = results[0].plot()  # annotated frame.
        cv.imwrite(f"{path}/frame{frameCount}_labels.jpg", labeled_bgr)

        # Extract detections from Results/Boxes
        boxes = results[0].boxes  # Boxes object: .xyxy, .cls, .conf.
        xyxy = boxes.xyxy.cpu().numpy() if boxes is not None and len(boxes) else np.empty((0,4))
        cls_idx = boxes.cls.cpu().numpy().astype(int) if boxes is not None and len(boxes) else np.array([], dtype=int)
        names = results[0].names  # dict id->name

        # coords
        coordinatesXmin = xyxy[:, 0] if xyxy.size else np.array([])
        coordinatesYmin = xyxy[:, 1] if xyxy.size else np.array([])
        coordinatesXmax = xyxy[:, 2] if xyxy.size else np.array([])
        coordinatesYmax = xyxy[:, 3] if xyxy.size else np.array([])
        classes = np.array([names[i] for i in cls_idx]) if cls_idx.size else np.array([])
        print('classes: ', classes)

        # filter out unembryonated eggs and too small boxes (false positives)
        minBoxSize = 45
        maxBoxSize = 110
        coordinatesXminFiltered = []
        coordinatesYminFiltered = []
        coordinatesXmaxFiltered = []
        coordinatesYmaxFiltered = []
        for i in range(len(coordinatesXmin)):
            if (coordinatesYmax[i] - coordinatesYmin[i]) > minBoxSize and (coordinatesYmax[i] - coordinatesYmin[i]) < maxBoxSize and (coordinatesXmax[i] - coordinatesXmin[i]) > minBoxSize and (coordinatesXmax[i] - coordinatesXmin[i]) < maxBoxSize: #and (classes[i] == 'embryonated'):
                coordinatesXminFiltered.append(coordinatesXmin[i])
                coordinatesYminFiltered.append(coordinatesYmin[i])
                coordinatesXmaxFiltered.append(coordinatesXmax[i])
                coordinatesYmaxFiltered.append(coordinatesYmax[i])

        print('coordinatesXminFiltered: ', coordinatesXminFiltered)
        # calculate centers of the boxes
        centers = []
        for i in range(len(coordinatesXminFiltered)):
            x = coordinatesXminFiltered[i] + ((coordinatesXmaxFiltered[i] - coordinatesXminFiltered[i]) / 2)
            y = coordinatesYminFiltered[i] + ((coordinatesYmaxFiltered[i] - coordinatesYminFiltered[i]) / 2)
            center = [x, y]
            centers.append(center)

        print('centers: ', centers)

        # convert y-values to be coordinates relative to image center (center = (x = 1024, y = 1024))
        for center in centers:
            center[1] = center[1] - 1024

        
        print('centers after y correction: ', centers)

        # calculate relative x/y-distances from first egg to position (0, 1024) and then from egg to egg
        x_distances = []
        y_distances = []
        print('len(centers): ', len(centers))

        if len(centers) == 1:
            x_distances.append(centers[0][0])
            y_distances.append(centers[0][1])
        elif len(centers) > 1:
            # threshold against double detections
            previousCenter = [0, 0]
            centers = sorted(centers, key=lambda x: x[0])
            for center in centers:
                if (np.abs(center[0] - previousCenter[0]) < 10) and (np.abs(center[1] - previousCenter[1]) < 10):
                    centers.remove(center)
                    print('CENTER REMOVED')
                previousCenter = center
            # threshold against double detections end

            x_distances.append(centers[0][0])
            y_distances.append(centers[0][1])
            print('centers: ', centers)
            for i in range(1, len(centers)):
                x_distances.append(centers[i][0] - centers[i - 1][0])
                if centers[i][1] >= 0 and centers[i - 1][1] >= 0:                   # both points above 0, current point higher than last point
                    if centers[i][1] > centers[i - 1][1]:
                        y_distances.append(centers[i][1] - centers[i - 1][1])
                    elif centers[i][1] < centers[i - 1][1]:                         # both points above 0, current point higher than last point
                        y_distances.append(centers[i][1] - centers[i - 1][1])
                elif centers[i][1] >= 0 and centers[i - 1][1] <= 0:                 # current point above 0, last point below 0, current point higher than last
                    y_distances.append(np.abs(centers[i - 1][1]) + centers[i][1])                           
                elif centers[i][1] <= 0 and centers[i - 1][1] >= 0:                 # current point below 0, last point above 0
                    y_distances.append(centers[i][1] - centers[i - 1][1])
                elif centers[i][1] <= 0 and centers[i - 1][1] <= 0:                 # both points below 0, current point higher than last point
                    if centers[i][1] > centers[i - 1][1]:
                        y_distances.append(centers[i][1] - centers[i - 1][1])
                    elif centers[i][1] < centers[i - 1][1]:                         # both points below 0, current point higher than last point
                        y_distances.append(centers[i][1] - centers[i - 1][1])

        # make coordinate pairs from x/y-distances + calculate sum of egg distances on x-axis
        centers = []
        xDistSum = 0
        for i in range (len(x_distances)):
            center = [x_distances[i], y_distances[i]]
            centers.append(center)
            xDistSum += x_distances[i]

        print('centers: ', centers)
        print('xDistSum: ', xDistSum)

        # steps on x-axis for moving from image center to left end of image + from last egg to right end of image
        xPxToSteps = int(1024 * stepSize * -1)
        currentPosX, currentUPosX = get_position(lib, device_id1)
        move(lib, device_id1, currentPosX + xPxToSteps, currentUPosX)
        wait_both_axes() #time.sleep(2)   

        filename = "img/{0}/frame{1}_left.jpeg".format(directoryName, frameCount)
        print(filename)
        saveImage += 1

        # calculate remaining distance to end of image behind last egg + to center of next frame
        xToNextFrame = int(((2048 - xDistSum) * stepSize) + (1024 * stepSize))
        print('xToNextFrame: ', xToNextFrame)

        # initialize counting variables for x in y, in case of canceled run stage goes back to starting position 
        xDist = 0
        yDist = 0

        # move each egg from the current image to center and illuminate it for exposureTime
        imgCount = 1
        # endTime_irradiation = -1
        centersIndex = 0
        while centersIndex < len(centers):
            # if time.time() >= endTime_irradiation:
            if quitMain == False:
                print('quitMain: ', quitMain)
                print('current time: ', time.time())
                # close gate if not already closed
                board.digital[pin].write(0)
                if gateOpened and pending_opened_ts is not None:
                    closed_ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                    append_shutter_event(shutter_event_id, pending_opened_ts, closed_ts)
                    shutter_event_id += 1
                    pending_opened_ts = None
                print('gate closed')
                gateOpened = False
                startTime2 = -1
                
                # compute needed steps to reach egg (pixels -> stage steps)
                xPxToSteps = int(round((centers[centersIndex][0] * stepSize), 3))
                yPxToSteps = int(round((centers[centersIndex][1] * stepSize), 3))
                print('xPxToSteps, yPxToSteps: ', xPxToSteps, yPxToSteps)

                xDist += centers[centersIndex][0]
                yDist += centers[centersIndex][1]
                print('xDist, yDist: ', xDist, yDist)

                # get current position and move to required position
                currentPosX, currentUPosX = get_position(lib, device_id1)
                currentPosY, currentUPosY = get_position(lib, device_id2)

                #move(lib, device_id1, currentPosX + 10, currentUPosX) # x movement
                move(lib, device_id1, currentPosX + xPxToSteps, currentUPosX) # x movement
                move(lib, device_id2, currentPosY + yPxToSteps, currentUPosY) # y movement
                print("\ngoing to {0}x, {1}y\n".format(centers[centersIndex][0], centers[centersIndex][1]))
                wait_both_axes() #time.sleep(2)

                # save current live image
                filename = "img/{0}/frame{1}_img{2}_from_{3}.jpeg".format(directoryName, frameCount, imgCount, len(centers))
                print(filename)
                saveImage += 1

                # --- CORRECTION PASS (YOLOv11 API) ---
                imgCorrection = img.copy()
                img_corr_rgb = cv.cvtColor(imgCorrection, cv.COLOR_BGR2RGB)

                # Predict with same settings as main pass
                res_list = model(img_corr_rgb, **YOLO_PARAMS)  # returns a list of Results
                r_corr = res_list[0]
                boxes_corr = r_corr.boxes  # Boxes object with .xyxy/.cls/.conf

                centersCorrection = []
                if boxes_corr is not None and len(boxes_corr):
                    xyxy_corr = boxes_corr.xyxy.cpu().numpy()
                    for j in range(xyxy_corr.shape[0]):
                        x1, y1, x2, y2 = xyxy_corr[j]
                        centersCorrection.append([(x1 + x2) / 2.0, (y1 + y2) / 2.0])
                else:
                    print('No detections during correction; skipping fine alignment.')

                print('centersCorrection: ', centersCorrection)

                # Default to no correction if nothing found
                xPxToStepsCorrection = 0
                yPxToStepsCorrection = 0

                if centersCorrection:
                    tree = spatial.KDTree(centersCorrection)
                    nearestPointIndex = tree.query([(1024, 1024)])[1][0]
                    point = centersCorrection[nearestPointIndex]
                    print('point: ', point)

                    # convert from image coords to center-relative coords
                    point[0] = point[0] - 1024
                    point[1] = point[1] - 1024
                    print('point after -1024: ', point)

                    xPxToStepsCorrection = int(round((point[0] * stepSize), 3))
                    yPxToStepsCorrection = int(round((point[1] * stepSize), 3))

                currentPosX, currentUPosX = get_position(lib, device_id1)
                currentPosY, currentUPosY = get_position(lib, device_id2)

                print('-------------------------------------CORRECTION-------------------------------------')
                move(lib, device_id1, currentPosX + xPxToStepsCorrection, currentUPosX) # x movement
                move(lib, device_id2, currentPosY + yPxToStepsCorrection, currentUPosY) # y movement
                wait_both_axes() #time.sleep(1)
                
                currentPosX, currentUPosX = get_position(lib, device_id1)
                print('currentPosX: ', currentPosX)
                currentPosY, currentUPosY = get_position(lib, device_id2)
                print('currentPosY: ', currentPosY)

                # open gate
                board.digital[pin].write(1)
                startTime2 = datetime.now()
                pending_opened_ts = startTime2.strftime("%H:%M:%S.%f")[:-3]
                print('gate opened')
                gateOpened = True
                time.sleep(exposureTime)

                currentPosX, currentUPosX = get_position(lib, device_id1)

                centersIndex += 1
                print('centersIndex: ', centersIndex)
                imgCount += 1

        # move to next frame behind last egg
        currentPosX, currentUPosX = get_position(lib, device_id1)
        currentPosY, currentUPosY = get_position(lib, device_id2)
        #time.sleep(2)

        #######################CONTINUE HERE#########################
        board.digital[pin].write(0)
        if gateOpened and pending_opened_ts is not None:
            closed_ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            append_shutter_event(shutter_event_id, pending_opened_ts, closed_ts)
            shutter_event_id += 1
            pending_opened_ts = None
        gateOpened = False
        print('CURRENT Y POSITION: ', currentPosY)
        print('GLOBAL Y CENTER: ', global_Y_center)
        print('DIFFERENCE: ', currentPosY - global_Y_center)
        yToCenter = int((currentPosY - global_Y_center) * (-1))
        print('yToCenter: ', yToCenter)
        move(lib, device_id2, currentPosY + yToCenter, currentUPosY) # y movement
        #wait_both_axes() #time.sleep(1)
        move(lib, device_id1, currentPosX + xToNextFrame, currentUPosX) # x movement
        wait_both_axes() #time.sleep(1)
        currentPosX, currentUPosX = get_position(lib, device_id1)
        currentPosY, currentUPosY = get_position(lib, device_id2)
        
        print("\n-------\nNEXT FRAME\n-------\n")
        frameCount += 1

    cv.destroyAllWindows()

    # move back to starting position
    currentPosX, currentUPosX = get_position(lib, device_id1)
    currentPosY, currentUPosY = get_position(lib, device_id2)
    moveToBeginning = int(frameCount * round(2048 * stepSize))
    move(lib, device_id1, currentPosX - moveToBeginning, currentUPosX)
    wait_both_axes() #time.sleep(5)

    print("\nClosing")

    # The device_t device parameter in this function is a C pointer, unlike most library functions that use this parameter
    lib.close_device(byref(cast(device_id1, POINTER(c_int))))

    print("Done")

def cameraViewer():
    global img
    global gateOpened
    global startTime2
    global saveImage
    global filename
    global lastKeyPressed
    global board
    global quitMain

    oldSaveImage = saveImage
    irradiationStart = False
    startTime = -1

    while camera.IsGrabbing():
        grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

        if grabResult.GrabSucceeded():
            image = converter.Convert(grabResult)
            img = image.GetArray()
            imgWithPrints = img.copy()
            cv.namedWindow('NematoVis', cv.WINDOW_NORMAL)
            cv.resizeWindow('NematoVis', 1024, 1024)
            cv.line(imgWithPrints, (1004, 1024), (1044, 1024), (0, 0, 255), 2)
            cv.line(imgWithPrints, (1024, 1004), (1024, 1044), (0, 0, 255), 2)
            cv.putText(imgWithPrints, '{0}'.format(lastKeyPressed), (1978, 110), cv.FONT_HERSHEY_SIMPLEX , 1, (100, 100, 180), 2, cv.LINE_AA)
            k = cv.waitKey(1)
            if k == 27:
                break
            
            if irradiationStart == True:
                diff = (datetime.now() - startTime).seconds + 1
                hours = diff // 3600
                if hours < 10:
                    hours = '0' + str(hours)
                minutes = diff // 60
                if minutes < 10:
                    minutes = '0' + str(minutes)
                seconds = diff % 60
                if seconds < 10:
                    seconds = '0' + str(seconds)
                cv.putText(imgWithPrints, '{0}:{1}:{2}'.format(hours, minutes, seconds), (70, 70), cv.FONT_HERSHEY_SIMPLEX , 1, (0, 0, 255), 2, cv.LINE_AA) # total timer

            if gateOpened == True:
                cv.putText(imgWithPrints, 'gate opened', (1800, 70), cv.FONT_HERSHEY_SIMPLEX , 1, (0, 0, 255), 2, cv.LINE_AA)
                diff2 = (datetime.now() - startTime2).seconds + 1
                hours2 = diff2 // 3600
                if hours2 < 10:
                    hours2 = '0' + str(hours2)
                minutes2 = diff2 // 60
                if minutes2 < 10:
                    minutes2 = '0' + str(minutes2)
                seconds2 = diff2 % 60
                if seconds2 < 10:
                    seconds2 = '0' + str(seconds2)
                cv.putText(imgWithPrints, '{0}:{1}:{2}'.format(hours2, minutes2, seconds2), (70, 110), cv.FONT_HERSHEY_SIMPLEX , 1, (0, 255, 0), 2, cv.LINE_AA) # gate timer
            else:
                cv.putText(imgWithPrints, 'gate closed', (1800, 70), cv.FONT_HERSHEY_SIMPLEX , 1, (0, 0, 255), 2, cv.LINE_AA)

            cv.imshow('NematoVis', imgWithPrints)

            if oldSaveImage < saveImage:
                cv.imwrite(filename, img)
                oldSaveImage = saveImage

        grabResult.Release()

        k = cv.waitKey(20) & 0xFF
        # start irradiation
        if k == ord('m'):
            startTime = datetime.now()
            lastKeyPressed = 'm'
            irradiationStart = True
            thread1 = Thread(target = irradiation, args = [])
            thread1.start()
        # mpve up
        elif k == ord('w'):
            lastKeyPressed = 'w'
            currentPosY, currentUPosY = get_position(lib, device_id2)
            move(lib, device_id2, currentPosY - 50, currentUPosY)
        elif k == ord('a'):
            lastKeyPressed = 'a'
            currentPosX, currentUPosX = get_position(lib, device_id1)
            move(lib, device_id1, currentPosX - 150, currentUPosX)
        # move right
        elif k == ord('d'):
            lastKeyPressed = 'd'
            currentPosX, currentUPosX = get_position(lib, device_id1)
            move(lib, device_id1, currentPosX + 150, currentUPosX)
        # move down
        elif k == ord('s'):
            lastKeyPressed = 's'
            currentPosY, currentUPosY = get_position(lib, device_id2)
            move(lib, device_id2, currentPosY + 50, currentUPosY)
        # quit program
        elif k == ord('k'):
            lastKeyPressed = 'k'
            print('k')
            board.digital[pin].write(0)
            quitMain = True
            sys.exit()
        elif k == ord('t'):
            #proc2.terminate()  # sends a SIGTERM
            pass
        elif k == ord('o'):
            print('o pressed')
            lastKeyPressed = 'o'
            # open and close gate
            if gateOpened == False:
                board.digital[pin].write(1)
                gateOpened = True
        elif k == ord('c'):
            print('c pressed')
            lastKeyPressed = 'c'
            # open and close gate
            if gateOpened == True:
                board.digital[pin].write(0)
                gateOpened = False

    # Releasing the resource
    camera.StopGrabbing()
    
    cv.destroyAllWindows()

if __name__ == "__main__":
    img = 0
    gateOpened = False
    startTime2 = -1
    saveImage = 0
    filename = ''
    lastKeyPressed = ''
    quitMain = False
    exposureTime = EXPOSURE_TIME_SEC

    set_speed(lib, device_id1, STAGE_SPEED) #1000
    
    # initialize arduino for the shutter
    port = ARDUINO_PORT
    board = pyfirmata.Arduino(port)
    pin = ARDUINO_PIN
    
    camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
    camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
    converter = pylon.ImageFormatConverter()
    converter.OutputPixelFormat = pylon.PixelType_BGR8packed
    converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

    # proc = multiprocessing.Process(target=cameraViewer, args=())
    # proc.start()

    thread2 = Thread(target = cameraViewer, args = [])
    thread2.start()
    thread2.join()

# ... your code ...
# on shutdown:
close_device_safely(lib, device_id1)
close_device_safely(lib, device_id2)
