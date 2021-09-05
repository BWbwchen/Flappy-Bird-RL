
class ReplayBuffer:
	def __init__(self) :
		self.records = []
	
	def Add(self, record) :
		self.records.append(record)

	def Recently_data(self, batch_size) :
		if batch_size >= len(self.records) :
			return self.records
		else :
			return self.records[-batch_size:]