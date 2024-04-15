# This file is to generate the screenclips
import cv2
import numpy as np
from ctypes import windll
import win32gui
import win32ui
from PIL import Image
from time import sleep
import os


def capture_win_alt(window_name: str):
    # Adapted from https://stackoverflow.com/questions/19695214/screenshot-of-inactive-window-printwindow-win32gui
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
    # define the window you want to capture
    WINDOW_NAME = "Counter-Strike 2"

    while cv2.waitKey(1) != ord('q'):
        img = capture_win_alt(WINDOW_NAME)
        im = Image.fromarray(img[..., [2, 1, 0]])   # note that win32ui returns images in BGR color channels, and we want to save it in RGB format
        im.save(f"./raw_data/img_{len(os.listdir('images'))}.jpg")

        sleep(0.8)  # sampling frequency

    cv2.destroyAllWindows()