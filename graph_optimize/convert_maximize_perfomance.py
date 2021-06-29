import sys
import numpy as np
import uff
import tensorrt as trt
import graphsurgeon as gs
from utils import load_in_out_info
import os

def convert_to_local_cuda_driver_config(graph_file,output_node_names,input_node_names,image_shape,quantized,root_folder):
    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    trt.init_libnvinfer_plugins(TRT_LOGGER, '')
    _,input_nodes,ouput_nodes = uff.from_tensorflow(
        gs.DynamicGraph(graph_file).as_graph_def(),
        output_nodes=output_node_names,
        output_filename=graph_file.replace('pb','.uff'),
        text=True,
        quiet=True,
        return_graph_info=True,
        debug_mode=False)
    input_dims = (3,image_shape,image_shape)
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, trt.UffParser() as parser:
        builder.max_workspace_size = 1<<28
        builder.max_batch_size = 1
        if quantized == 0:
            builder.fp32_mode = True
        elif quantized == 1:
            builder.fp16_mode = True
        else:
            builder.int8_mode = True
            # builder.int8_calibrator

        parser.register_input(input_node_names[0], input_dims)
        for i in range(len(output_node_names)):
            parser.register_output(f'MarkOutput_{i}')
        # parser.parse(graph_file, network)
        parser.parse(graph_file.replace('pb','.uff'), network)
        engine = builder.build_cuda_engine(network)

        buf = engine.serialize()
        with open(os.path.join(root_folder,'trt_graph.bin'), 'wb') as f:
            f.write(buf)

if __name__ == '__main__':
    graph_file = sys.argv[1]
    root_folder = sys.argv[2]
    image_shape = int(sys.argv[3])
    quantized = int(sys.argv[4])
    info = load_in_out_info(root_folder)
    convert_to_local_cuda_driver_config(graph_file,info['outputs'],info['inputs'],image_shape,quantized,root_folder)
