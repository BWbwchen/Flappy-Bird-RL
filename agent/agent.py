from policy import Policy
from replay import ReplayBuffer
from datetime import datetime

BATCH_SIZE = 2

class Agent:
	def __init__(self, input_size, output_size, load_model=False, model_path=None) :
		self.policy = Policy(input_size, output_size, load_model, model_path)
		self.replay_buffer = ReplayBuffer()

	def Train(self, epoch):
		recent_record_list = self.replay_buffer.Recently_data(BATCH_SIZE)
		#recent_record_list = self.replay_buffer.Random_data(BATCH_SIZE)
		self.policy.Train(epoch, recent_record_list)

	def Add_to_replay_buffer(self, record):
		self.replay_buffer.Add(record)

	def Act(self, obs) :
		return self.policy.Get_action(obs)

	def Action_distribution(self, obs) :
		return self.policy.Get_action_distribution(obs)

	def Save(self, filePath: str) :
		if filePath is None :
			filePath = "./model/" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + ".h5"
		self.policy.Save(filePath)

	def Log(self, msg) :
		print(msg)