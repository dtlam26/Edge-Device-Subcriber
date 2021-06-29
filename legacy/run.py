import numpy as np
import tflite_runtime.interpreter as tflite
from tflite_runtime.interpreter import load_delegate
import cv2


def det_and_display(cv2_im,interpreter,threshold,labels):
    img_orig_shape = cv2_im.shape
    # cv2_im = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]['shape']
    cv2_im_rgb = cv2.resize(cv2_im, (input_shape[1],input_shape[2]))
    cv2_im_rgb = np.expand_dims(cv2_im_rgb,0)
    # Test the model on random input data.

    interpreter.set_tensor(input_details[0]['index'], cv2_im_rgb)

    interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    output_data1 = interpreter.get_tensor(output_details[0]['index'])[0]
    output_data2 = interpreter.get_tensor(output_details[1]['index'])[0]
    output_data3 = interpreter.get_tensor(output_details[2]['index'])[0]
    output_data4 = interpreter.get_tensor(output_details[3]['index'])[0]
    # print(output_data1,output_data2,output_data3,output_data4)

    for bb,score,cls in zip(output_data1,output_data3,output_data2):
        if score > threshold:
        # print(cls)
            x1 = int(bb[1]*img_orig_shape[1])
            x2 = int(bb[3]*img_orig_shape[1])
            y1 = int(bb[0]*img_orig_shape[0])
            y2 = int(bb[2]*img_orig_shape[0])
            cv2.rectangle(cv2_im,(x1,y1),(x2,y2),(111,255,220), thickness=2)
            cv2.putText(cv2_im,labels[int(cls)],(int((x1+x2)/2),int((y1+y2)/2)), cv2.FONT_HERSHEY_SIMPLEX, 3,(100, 210, 0),3)
    final = cv2.resize(cv2_im, (600,600))
    return final

import threading

class StoppableDetection(threading.Thread):
    """Thread class with a stop() method. The thread itself has to check
    regularly for the stopped() condition."""

    def __init__(self,  default_model,threshold,camera):
        super(StoppableDetection, self).__init__()
        self._stop_event = threading.Event()
        default_labels = 'label.txt'
        self.threshold = threshold
        with open(default_labels, 'r') as f:
            pairs = (l.strip().split(maxsplit=1) for l in f.readlines())
            self.labels = dict((int(k), v) for k, v in pairs)
        print(self.labels)
        self.interpreter = tflite.Interpreter(model_path=default_model,experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
        self.interpreter.allocate_tensors()
        self.cap = cv2.VideoCapture(camera)

    def stop(self):
        self._stop_event.set()
        cv2.destroyAllWindows()
        self.cap.release()

    def stopped(self):
        return self._stop_event.is_set()

    def inference(self):
        while(self.cap.isOpened()):
            ret, cv2_im = self.cap.read()
            final = det_and_display(cv2_im,self.interpreter,self.threshold,self.labels)
            cv2.imshow('track',final)
            k = cv2.waitKey(1)
            if k == ord('q'):
                break
        cv2.destroyAllWindows()
        self.cap.release()

    def run(self):
        self.inference()
