# This file is for real-time yolo detection
from ultralytics import YOLO
import cv2
import numpy as np
from ultralytics.utils.plotting import Annotator
from ctypes import windll
import win32gui
import win32ui
import time


def capture_win_alt(window_name: str):
    windll.user32.SetProcessDPIAware()
    hwnd = win32gui.FindWindow(None, window_name)

    left, top, right, bottom = win32gui.GetClientRect(hwnd)
    w = right - left
    h = bottom - top

    hwnd_dc = win32gui.GetWindowDC(hwnd)
    mfc_dc = win32ui.CreateDCFromHandle(hwnd_dc)
    save_dc = mfc_dc.CreateCompatibleDC()
    bitmap = win32ui.CreateBitmap()
    bitmap.CreateCompatibleBitmap(mfc_dc, w, h)
    save_dc.SelectObject(bitmap)

    # If Special K is running, this number is 3. If not, 1
    result = windll.user32.PrintWindow(hwnd, save_dc.GetSafeHdc(), 3)

    bmpinfo = bitmap.GetInfo()
    bmpstr = bitmap.GetBitmapBits(True)

    img = np.frombuffer(bmpstr, dtype=np.uint8).reshape((bmpinfo["bmHeight"], bmpinfo["bmWidth"], 4))
    img = np.ascontiguousarray(img)[..., :-1]  # make image C_CONTIGUOUS and drop alpha channel

    if not result:  # result should be 1
        win32gui.DeleteObject(bitmap.GetHandle())
        save_dc.DeleteDC()
        mfc_dc.DeleteDC()
        win32gui.ReleaseDC(hwnd, hwnd_dc)
        raise RuntimeError(f"Unable to acquire screenshot! Result: {result}")

    return img


if __name__ == '__main__':
  # Load the model first
  # model = YOLO('../yolov8n.pt')
  model = YOLO('models/175.pt')
  model.to('cuda')  # use GPU for faster inference time

  WINDOW_NAME = "Counter-Strike 2"  # change this to the game window

  loop_time = time.time()
  while cv2.waitKey(1) != ord('q'):
      # capture window
      screenshot = capture_win_alt(WINDOW_NAME)

      #######################################################################
      
      if type(screenshot) is np.ndarray:
        screenshot = cv2.cvtColor(screenshot, cv2.COLOR_BGR2RGB)  # we need to convert to RGB channel for YOLO to work
      # predict the screen  
      results = model.predict(screenshot, conf=0.25)
      
      # draw the bounding boxes
      for result in results:
        if hasattr(result.boxes, 'xyxy') and len(result.boxes.xyxy) <= 0: # only continue if object detected
          continue

        annotator = Annotator(screenshot) # has to be continuous
        boxes = result.boxes
        for box in boxes:
          b = box.xyxy[0]  # get box coordinates in (top, left, bottom, right) format
          c = box.cls      # get the class name
          annotator.box_label(b, model.names[int(c)])
      
      # Downscale
      if type(screenshot) is np.ndarray:
        # resize for demoing purpose
        # screenshot = cv2.resize(screenshot, (0, 0), fx=0.25, fy=0.25)
        
        screenshot = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)
        # print out the FPS
        fps_text = 1/(time.time() - loop_time)
        print('FPS: {}'.format(fps_text))
        loop_time = time.time()
        cv2.putText(screenshot, "{:.2f}".format(fps_text), (5, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,255), 1)
        
        # show on the screen
        cv2.imshow('Screenshot', np.array(screenshot))
      #######################################################################

  cv2. destroyAllWindows()