class Identity:
	def __init__(self):
		self.requires_detector = False
	
	def __call__(self, img):
		return img