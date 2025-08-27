# test_onnx.py
import numpy as np
import onnxruntime as ort

sess = ort.InferenceSession("onnx_models/policy.onnx", providers=['CPUExecutionProvider'])
# make dummy inputs
history = np.zeros((3,18), dtype=np.float32)   # seq_len 3
hand = np.zeros((15,), dtype=np.float32)
actions = np.zeros((5,15), dtype=np.float32)   # 5 candidate moves
lord = np.array([1.0], dtype=np.float32)

out = sess.run(None, {"history": history, "hand": hand, "actions": actions, "lord": lord})
scores = out[0]
print("scores.shape", scores.shape, "scores:", scores)
