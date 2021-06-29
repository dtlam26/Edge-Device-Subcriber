import tensorflow as tf
import numpy as np
import cv2
from deployment.utils import ThreadidManagement

def get_frozen_graph(graph_file):
    """Read Frozen Graph file from disk."""
    with tf.compat.v1.gfile.FastGFile(graph_file, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
    return graph_def
@ThreadidManagement
def load_model(model_path,inputs,outputs,image_shape,gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.2)):
    trt_graph = get_frozen_graph(model_path)
    tf_config = tf.compat.v1.ConfigProto(gpu_options=gpu_options)
    tf_sess = tf.compat.v1.Session(config=tf_config)
    tf.compat.v1.import_graph_def(trt_graph, name='')
    tf_input = tf_sess.graph.get_tensor_by_name(inputs[0]+':0')
    tf_scores = tf_sess.graph.get_tensor_by_name(outputs[0]+':0')
    tf_boxes = tf_sess.graph.get_tensor_by_name(outputs[1]+':0')
    return tf_sess,tf_input,[tf_scores, tf_boxes]

@ThreadidManagement
def allocate(tf_sess,tf_input,tf_output):
    image_np = np.random.uniform(-1,1,(1,image_shape,image_shape,3))
    scores, boxes = tf_sess.run(tf_output, feed_dict={
        tf_input: image_np
    })
    return tf_sess


class TF_TRT_model(object):
    def __init__(self,sess,input,output,normalize,image_shape):
        self.tf_sess=sess
        self.tf_input=input
        self.tf_output=output
        self.normalize=normalize
        self.image_shape=image_shape

    def inference(self,frame):
        image = cv2.resize(frame, (self.image_shape, self.image_shape))
        image_ori_shape = image.shape

        if self.normalize == 2:
            image_np = (image*2/255.0)-127.5
        elif self.normalize == 1:
            image_np = image/255.0
        else:
            image_np = image
        image_np = np.expand_dims(image_np,0)
        scores, boxes = self.tf_sess.run(self.tf_output,
                                    feed_dict={self.tf_input: image_np})
        return [scores,boxes]
