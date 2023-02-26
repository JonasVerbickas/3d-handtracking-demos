class Identity:
	def __init__(self):
		self.requires_detector = True
	
	def __call__(self, img):
		return img