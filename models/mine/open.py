import onnx
import cv2
# onnx_model = onnx.load( 'best.onnx')
# onnx.checker.check_model(onnx_model)


model = cv2.dnn.readNetFromONNX('best.onnx')
