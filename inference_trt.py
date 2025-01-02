import tensorrt as trt
import numpy as np
from PIL import Image

import pycuda.driver as cuda
import pycuda.autoinit

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def load_engine(engine_file_path):
    with open(engine_file_path, 'rb') as f:
        runtime = trt.Runtime(TRT_LOGGER)
        return runtime.deserialize_cuda_engine(f.read())

def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()

    for binding in engine:
        size = trt.volume(engine.get_tensor_shape(binding))
        dtype = trt.nptype(engine.get_tensor_dtype(binding))
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(device_mem))
        if engine.get_tensor_mode(binding) == trt.TensorIOMode.INPUT:
            inputs.append({'host': host_mem, 'device': device_mem})
        else:
            outputs.append({'host': host_mem, 'device': device_mem})
    return inputs, outputs, bindings, stream

def do_inference(context, bindings, inputs, outputs, stream):
    [cuda.memcpy_htod_async(inp['device'], inp['host'], stream) for inp in inputs]
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    [cuda.memcpy_dtoh_async(out['host'], out['device'], stream) for out in outputs]
    stream.synchronize()
    return [out['host'] for out in outputs]

def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    original_size = image.size
    image = image.resize((256, 256))
    image = np.array(image) / 255.0
    image = image.transpose((2, 0, 1)).astype(np.float32)
    return image, original_size

def postprocess_image(output, original_size):
    output = output.reshape(3, 256, 256)
    output = (output * 255).astype(np.uint8)
    output_image = Image.fromarray(output.transpose(1, 2, 0))
    output_image = output_image.resize(original_size)
    return output_image

def main():
    engine_file_path = 'model.trt'
    image_path = 'test_image/sample.jpg'
    output_path = 'output.jpg'

    engine = load_engine(engine_file_path)
    context = engine.create_execution_context()
    inputs, outputs, bindings, stream = allocate_buffers(engine)

    input_image, original_size = preprocess_image(image_path)
    np.copyto(inputs[0]['host'], input_image.ravel())

    output = do_inference(context, bindings, inputs, outputs, stream)
    output_image = postprocess_image(output[0], original_size)

    # write the output image to disk
    output_image.save(output_path)
    print(f"Output saved to {output_path}")

if __name__ == '__main__':
    main()