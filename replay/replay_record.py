import numpy as np
class ReplayRecord:
	def __init__(self) :
		self.observations = None
		self.actions = None

	def Add(self, obs, acts):
		obs = np.array([obs])
		acts = np.array([acts])
		if self.observations is None :
			self.observations = np.array(obs)
			self.actions = np.array(acts)
		else :
			self.observations = np.append(self.observations, obs, axis=0)
			self.actions = np.append(self.actions, acts, axis=0)

	def Detail(self) :
		return self.observations, self.actions