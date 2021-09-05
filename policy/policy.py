from numpy.core.defchararray import mod
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

LR = 0.003
LOG_INTERVAL = 100

class Policy:
	def __init__(self, input, output, load_model=False, model_path=None) :
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

		self.net = Net(input, output).to(self.device)
		if load_model :
			self.net.load_state_dict(torch.load(model_path))
			print("Load model Finish !")

		self.optimizer = optim.Adadelta(self.net.parameters(), lr=LR)
		self.loss = nn.CrossEntropyLoss()

	def Train(self, epoch, recent_record_list) :
		for batch_idx, record in enumerate(recent_record_list) :
			# input
			obs, acts = record.Detail()
			obs = torch.tensor(obs).to(self.device)
			acts = torch.tensor(acts).to(self.device)
			acts = torch.argmax(acts, dim=1).long()

			# gradient
			self.optimizer.zero_grad()
			actions = self.net(obs)
			loss = self.loss(actions, acts)
			loss.backward()
			self.optimizer.step()

			#if batch_idx % LOG_INTERVAL == 0 :
			#	print("Train Epoch: {} [{}/{}], Loss: {:.6f}".format(epoch, batch_idx, len(recent_record_list), loss.item()))

	def Get_action(self, obs) :
		obs = torch.tensor(obs).to(self.device)
		output = self.net(obs)
		return torch.argmax(output, dim=0).cpu().numpy()

	def Get_action_distribution(self, obs) :
		obs = torch.tensor(obs).to(self.device)
		output = self.net(obs)
		return output.detach().cpu().numpy()

	def Save(self, filePath) :
		torch.save(self.net.state_dict(), filePath)
		print("Save model in : ", filePath)

class Net(nn.Module) :
	def __init__(self, input, output) :
		super(Net, self).__init__()
		self.fc1 = nn.Linear(input, 32, dtype=torch.double)
		self.fc2 = nn.Linear(32, output, dtype=torch.double)

	def forward(self, x) :
		x = self.fc1(x)
		x = F.relu(x)
		x = self.fc2(x)
		x = F.relu(x)
		output = F.log_softmax(x, dim=0)
		return output
