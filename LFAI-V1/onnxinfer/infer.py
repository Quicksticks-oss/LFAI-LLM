import onnxruntime as ort
import numpy as np

# Load the ONNX model
onnx_model_path = 'chat-lstm-10.38M-20230824-4-512-ctx512.onnx'
sess = ort.InferenceSession(onnx_model_path)

# Input data for the model
input_name = sess.get_inputs()[0].name
input_shape = sess.get_inputs()[0].shape
input_data = np.random.random(input_shape).astype(np.int64)

# Perform inference
output_names = [output.name for output in sess.get_outputs()]
outputs = sess.run(output_names, {input_name: input_data})

# Print the output
for output_name, output_data in zip(output_names, outputs):
    print(f"Output '{output_name}':\n{output_data}")
