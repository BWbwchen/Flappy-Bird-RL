
class ReplayBuffer:
	def __init__(self) :
		self.records = []
	
	def Add(self, record) :
		self.records.append(record)

	def Recently_data(self, batch_size) :
		return self.records[-batch_size:]