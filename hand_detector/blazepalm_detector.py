import tensorflow as tf
import numpy as np

class PalmDetector:
	def __init__(self):
		model_path = "palm_detection_full.tflite"
		self.interpreter = tf.lite.Interpreter(model_path=model_path)
		print(self.interpreter)
		input_details = self.interpreter.get_input_details()
		output_details = self.interpreter.get_output_details()
		self.interpreter.allocate_tensors()
		in_idx = input_details[0]['index']
		print(in_idx)
		out_reg_idx = output_details[0]['index']
		out_clf_idx = output_details[1]['index']
		print(out_reg_idx)
		print(out_clf_idx)
		
	def __call__(self, img: np.ndarray):
		assert -1 <= img.min() and img.max() <= 1,\
		"img_norm should be in range [-1, 1]"
		assert img.shape == (256, 256, 3),\
		"img_norm shape must be (256, 256, 3)"
		self.interp_palm.set_tensor(self.in_idx, img[None])




if __name__ == '__main__':