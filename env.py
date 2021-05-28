import numpy as np

class env(object):
	def __init__(self, rank):
		self.rank = rank
		# print(self.rank)

	def generate(self):

		return np.random.random(6) + self.rank