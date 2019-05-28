from tkinter import *
import argparse
import imutils
import time
import cv2
from imutils.video import VideoStream
from PIL import Image, ImageTk

class Window(Frame):

    def __init__(self, master=None):
        
        Frame.__init__(self, master,bg="#5ac1f4")   

        self.master = master

        self.init_window()

    def init_window(self):
        self.master.title("Style Transfer")
        self.pack(fill=BOTH, expand=1)

        StartButton = Button(self, text="Start WebCam",command=self.client_start,bg="white")
        StartButton.place(x=75, y=190)

        StopButton = Button(self, text="Exit",command=self.client_stop,bg="white")
        StopButton.place(x=250, y=190)

        load = Image.open("style.jpg")
        render = ImageTk.PhotoImage(load)

        # labels can be text or images
        img = Label(self, image=render)
        img.image = render
        img.place(x=50, y=50)

       

    def client_start(self):
        style="starry_night.t7"
        net = cv2.dnn.readNetFromTorch(style)

        vs = VideoStream(src=0).start()
        time.sleep(2.0)

        while True:
            frame = vs.read()
            frame = imutils.resize(frame, width=600)
            orig = frame.copy()
            (h, w) = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(frame, 1.0, (w, h),(103.939, 116.779, 123.680),swapRB=False, crop=False)
            net.setInput(blob)
            output = net.forward()
            output = output.reshape((3, output.shape[2], output.shape[3]))
            output[0] += 103.939
            output[1] += 116.779
            output[2] += 123.680
            output /= 255.0
            output = output.transpose(1, 2, 0)
            #cv2.imshow("Input", frame)
            cv2.imshow("Press Q to Quit", output)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
    	        break
        cv2.destroyAllWindows()
        vs.stop()

    def client_stop(self):
        exit()

        

root = Tk()

root.geometry("400x300")
root.resizable(False, False)
#creation of an instance
app = Window(root)

#mainloop 
root.mainloop()  