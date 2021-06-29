import cv2
import gi
import numpy as np
import tflite_runtime.interpreter as tflite
from tflite_runtime.interpreter import load_delegate

gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')
from gi.repository import Gst, GstRtspServer, GObject

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

class SensorFactory(GstRtspServer.RTSPMediaFactory):
    def __init__(self, default_model,threshold,camera):
        super(SensorFactory, self).__init__()
        self.cap = cv2.VideoCapture(camera)
        self.number_frames = 0
        self.fps = 10
        self.duration = 1 / self.fps * Gst.SECOND  # duration of a frame in nanoseconds
        self.launch_string = 'appsrc name=source is-live=true block=true format=GST_FORMAT_TIME ' \
                             'caps=video/x-raw,format=BGR,width=600,height=600,framerate={}/1 ' \
                             '! videoconvert ! video/x-raw,format=I420 ' \
                             '! x264enc speed-preset=ultrafast tune=zerolatency ! queue ' \
                             '! h264parse ! rtph264pay config-interval=1 name=pay0 pt=96 '.format(self.fps)
        # streams to gst-launch-1.0 rtspsrc location=rtsp://localhost:8554/test latency=50 ! decodebin ! autovideosink
        self.interpreter = tflite.Interpreter(model_path=default_model,experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
        self.interpreter.allocate_tensors()
        self._stop_event = threading.Event()
        default_labels = 'label.txt'
        self.threshold = threshold
        with open(default_labels, 'r') as f:
            pairs = (l.strip().split(maxsplit=1) for l in f.readlines())
            self.labels = dict((int(k), v) for k, v in pairs)

    def on_need_data(self, src, lenght):
        if self.cap.isOpened():
            ret, frame = self.cap.read()
            # print(frame.shape)
            if ret:
                frame = det_and_display(frame,self.interpreter,self.threshold,self.labels)
                # cv2.imshow('track',data)
                # k = cv2.waitKey(1)
                # if k == ord('q'):
                #     break
                #print(data)
                data = frame.tostring()
                buf = Gst.Buffer.new_allocate(None, len(data), None)
                buf.fill(0, data)
                buf.duration = self.duration
                timestamp = self.number_frames * self.duration
                buf.pts = buf.dts = int(timestamp)
                buf.offset = timestamp
                self.number_frames += 1
                retval = src.emit('push-buffer', buf)
                #print('pushed buffer, frame {}, duration {} ns, durations {} s'.format(self.number_frames,
                #                                                                       self.duration,
                #                                                                       self.duration / Gst.SECOND))
                if retval != Gst.FlowReturn.OK:
                    print("Status",retval)
        else:
            self.cap.release()

    def do_create_element(self, url):
        return Gst.parse_launch(self.launch_string)

    def do_configure(self, rtsp_media):
        self.number_frames = 0
        appsrc = rtsp_media.get_element().get_child_by_name('source')
        appsrc.connect('need-data', self.on_need_data)

    def stop(self):
        while self.cap.isOpened():
            self.cap.release()


class GstServer(GstRtspServer.RTSPServer):
    def __init__(self, default_model,threshold,camera):
        super(GstServer, self).__init__()
        self.factory = SensorFactory(default_model,threshold,camera)
        self.factory.set_shared(True)
        self.get_mount_points().add_factory("/test", self.factory)
        self.attach(None)

import threading

class StoppableDetection(threading.Thread):
    """Thread class with a stop() method. The thread itself has to check
    regularly for the stopped() condition."""

    def __init__(self, default_model,threshold,camera):
        super(StoppableDetection, self).__init__()
        self._stop_event = threading.Event()
        GObject.threads_init()
        Gst.init(None)
        self.server = GstServer(default_model,threshold,camera)
        self.loop = GObject.MainLoop()


    def stop(self):
        self._stop_event.set()
        self.server.factory.set_shared=False
        self.server.factory.stop()
        while (self.loop.is_running()):
            self.loop.quit()


    def stopped(self):
        return self._stop_event.is_set()

    def run(self):
        self.loop.run()
