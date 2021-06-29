import os
import numpy as np
import time
import json
# import tensorrt as trt
# import uff
# import graphsurgeon as gs

# import pycuda.driver as cuda
# from trt_model_flow import TrtModel

#
# def convert_to_local_cuda_driver_config(graph_file,info,image_shape,quantized,root_folder,trt_logger):
#     _,input_nodes,ouput_nodes = uff.from_tensorflow(
#         gs.DynamicGraph(graph_file).as_graph_def(),
#         output_nodes=info['outputs'],
#         output_filename=graph_file.replace('pb','.uff'),
#         text=True,
#         quiet=True,
#         return_graph_info=True,
#         debug_mode=False)
#     input_dims = (3,image_shape,image_shape)
#     with trt.Builder(trt_logger) as builder, builder.create_network() as network, trt.UffParser() as parser:
#         builder.max_workspace_size = 1<<28
#         builder.max_batch_size = 1
#         if quantized == 0:
#             builder.fp32_mode = True
#         elif quantized == 1:
#             builder.fp16_mode = True
#         else:
#             builder.int8_mode = True
#             # builder.int8_calibrator
#
#         parser.register_input(info['inputs'][0], input_dims)
#         for i in range(len(info['outputs'])):
#             parser.register_output(f'MarkOutput_{i}')
#         # parser.parse(graph_file, network)
#         parser.parse(graph_file.replace('pb','.uff'), network)
#         engine = builder.build_cuda_engine(network)
#
#         buf = engine.serialize()
#         with open(os.path.join(root_folder,'trt_graph.bin'), 'wb') as f:
#             f.write(buf)

def load_in_out_info(folder):
    with open(os.path.join(folder,'model.json')) as json_file:
        data = json.load(json_file)
    return data

def order_outputs(outputs):
    """outputs should be class score then box"""
    ordered_outputs = outputs.copy()
    for i,o in enumerate(outputs):
        if 'box' in o:
            ordered_outputs[0] = len(outputs)-i-1
        elif 'class' in o:
            ordered_outputs[1] = len(outputs)-i-1
        elif 'anchor' in o:
            ordered_outputs[2] = len(outputs)-i-1
    print(outputs,ordered_outputs)
    return ordered_outputs
