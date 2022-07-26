import torch
import torch.nn.functional as f
import torch.utils.data as data
import constant
import torchvision
import subprocess
import pandas as pd
import random
from test_bond_analysis import new_create_bond_matrix
from test_bond_analysis import new_process_bond_list
from test_bond_analysis import read_neighbor_list
from test_bond_analysis import new_create_force_matrix
from test_bond_analysis import new_create_formal_force_matrix

from left_test_bond_analysis import new_create_bond_matrix as left_new_create_bond_matrix
from left_test_bond_analysis import new_process_bond_list as left_new_process_bond_list
from left_test_bond_analysis import read_bond_list as left_read_bond_list

from right_test_bond_analysis import new_create_bond_matrix as right_new_create_bond_matrix
from right_test_bond_analysis import new_process_bond_list as right_new_process_bond_list
from right_test_bond_analysis import read_bond_list as right_read_bond_list
#from test_bond_analysis import new_create_force_matrix
#from test_bond_analysis import new_create_formal_force_matrix

# preprocessing
torch.manual_seed(2293)
torch.set_default_dtype(torch.float64)

bond_list = read_neighbor_list("data_input/bond_5_3only_74_458_2432.csv", 74)
bond_list = new_process_bond_list(bond_list)

random.seed(32768)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.input = torch.nn.Linear(539, 1024)
        self.hidden_1 = torch.nn.Linear(1024, 512)
        self.hidden_2 = torch.nn.Linear(512, 256)
        self.hidden_3 = torch.nn.Linear(256, 128)
        self.hidden_4 = torch.nn.Linear(128, 64)
        self.output = torch.nn.Linear(64, 2)

    def forward(self, x):
        x = f.relu(self.input(x))
        x = f.relu(self.hidden_1(x))
        x = f.relu(self.hidden_2(x))
        x = f.relu(self.hidden_3(x))
        x = f.relu(self.hidden_4(x))
        x = self.output(x)
        return x
 

class BoundNet(torch.nn.Module):
    def __init__(self, input_size_list):
        super(BoundNet, self).__init__()
        self.input = torch.nn.ModuleList([torch.nn.Linear(input_size_list[i], 1024) for i in range(5)])
        self.hidden_1 = torch.nn.Linear(1024, 512)
        self.hidden_2 = torch.nn.Linear(512, 256)
        self.hidden_3 = torch.nn.Linear(256, 128)
        self.hidden_4 = torch.nn.Linear(128, 64)
        self.output = torch.nn.Linear(64, 2)

    def forward(self, x):
        x = [f.relu(self.input[i](x[i])) for i in range(5)]
        x = torch.cat(x)
        x = f.relu(self.hidden_1(x))
        x = f.relu(self.hidden_2(x))
        x = f.relu(self.hidden_3(x))
        x = f.relu(self.hidden_4(x))
        x = self.output(x)
        return x
        
        
left_bond_list = []
left_bond_count = []
left_chirality_count = []
left_bond_c = []
left_chirality_c = []
left_feature_count = []

right_bond_list = []
right_bond_count = []
right_chirality_count = []
right_bond_c = []
right_chirality_c = []
right_feature_count = []

for i in range(5):
    left_bond_list_t, left_bond_count_t, left_chirality_count_t, left_bond_c_t, left_chirality_c_t = left_read_bond_list("data_input/bond" + str(i) + ".csv")
    left_bond_list.append(left_bond_list_t)
    left_bond_count.append(left_bond_count_t)
    left_chirality_count.append(left_chirality_count_t)
    left_bond_c.append(left_bond_c_t)
    left_chirality_c.append(left_chirality_c_t)
    left_feature_count.append(left_bond_c_t + left_chirality_c_t + 1)
    
for i in range(5):
    right_bond_list_t, right_bond_count_t, right_chirality_count_t, right_bond_c_t, right_chirality_c_t = right_read_bond_list("data_input/bond" + str(i + 27) + ".csv")
    right_bond_list.append(right_bond_list_t)
    right_bond_count.append(right_bond_count_t)
    right_chirality_count.append(right_chirality_count_t)
    right_bond_c.append(right_bond_c_t)
    right_chirality_c.append(right_chirality_c_t)
    right_feature_count.append(right_bond_c_t + right_chirality_c_t + 1)
    
    
print("**************************************")
print("Left Parameters: ")
print("Bond layer count: ", left_bond_count)
print("Chirality layer count", left_chirality_count)
print("Bond feature: ", left_bond_c)
print("Chirality feature: ", left_chirality_c)
print("Total feature count: ", left_feature_count)
left_bond_list = [left_new_process_bond_list(left_bond_list[i]) for i in range(5)]
print("**************************************")

print("**************************************")
print("Right Parameters: ")
print("Bond layer count: ", right_bond_count)
print("Chirality layer count", right_chirality_count)
print("Bond feature: ", right_bond_c)
print("Chirality feature: ", right_chirality_c)
print("Total feature count: ", right_feature_count)
right_bond_list = [right_new_process_bond_list(right_bond_list[i]) for i in range(5)]
print("**************************************")


def generate_force():
    data = pd.read_csv("share_file.csv", header=None, delimiter=",")
    
    spin_tensor = torch.tensor(data.iloc[:, 3:6].values, device=constant.device).requires_grad_(True)
    left_spin_tensor = torch.tensor(data.iloc[:, 3:6].values, device=constant.device).requires_grad_(True)
    right_spin_tensor = torch.tensor(data.iloc[:, 3:6].values, device=constant.device).requires_grad_(True)
    
    x_temp = new_create_bond_matrix(spin_tensor, bond_list)
        
    energy_prediction_temp = net(x_temp)
        
    energy_prediction = torch.sum(energy_prediction_temp, 0)
    energy_A = energy_prediction[0]
    energy_B = energy_prediction[1]
        
    force_prediction_A = -torch.autograd.grad(energy_A, spin_tensor, create_graph=True)[0]
    force_prediction_B = -torch.autograd.grad(energy_B, spin_tensor, create_graph=True)[0]

    force_prediction_temp = force_prediction_A + torch.cross(spin_tensor, force_prediction_B)
    force_prediction = torch.cross(spin_tensor, force_prediction_temp)
    
    left_x_temp = [left_new_create_bond_matrix(left_spin_tensor, left_bond_list[i], left_bond_count[i], left_chirality_count[i]) for i in range(5)]
    
    left_energy_prediction_temp = left_net(left_x_temp)
    left_energy_prediction = torch.sum(left_energy_prediction_temp, 0)
    left_energy_A = left_energy_prediction[0]
    left_energy_B = left_energy_prediction[1]
    print(left_energy_prediction, left_energy_A, left_energy_B)

    left_force_prediction_A = -torch.autograd.grad(left_energy_A, left_spin_tensor, create_graph=True)[0]
    left_force_prediction_B = -torch.autograd.grad(left_energy_B, left_spin_tensor, create_graph=True)[0]
    left_force_prediction_temp = left_force_prediction_A + torch.cross(left_spin_tensor, left_force_prediction_B)
    left_force_prediction = torch.cross(left_spin_tensor, left_force_prediction_temp)
    
    right_x_temp = [right_new_create_bond_matrix(right_spin_tensor, right_bond_list[i], right_bond_count[i], right_chirality_count[i]) for i in range(5)]
    
    right_energy_prediction_temp = right_net(right_x_temp)
    right_energy_prediction = torch.sum(right_energy_prediction_temp, 0)
    right_energy_A = right_energy_prediction[0]
    right_energy_B = right_energy_prediction[1]
    print(right_energy_prediction, right_energy_A, right_energy_B)

    right_force_prediction_A = -torch.autograd.grad(right_energy_A, right_spin_tensor, create_graph=True)[0]
    right_force_prediction_B = -torch.autograd.grad(right_energy_B, right_spin_tensor, create_graph=True)[0]
    right_force_prediction_temp = right_force_prediction_A + torch.cross(right_spin_tensor, right_force_prediction_B)
    right_force_prediction = torch.cross(right_spin_tensor, right_force_prediction_temp)
    
    force_prediction = new_create_formal_force_matrix(force_prediction, left_force_prediction, right_force_prediction)

    data.iloc[:, 9:12] = force_prediction.detach().numpy()
    data.to_csv("share_file.csv", header=False, index=False)
    return energy_prediction.detach().numpy()


print("Initializing net.")
net = Net()
net.load_state_dict(torch.load("data_input/pp_change_final_model_save_at_1_task_1.pt", map_location=lambda storage, loc: storage))
print(net)
net.to(constant.device)
print("Middle net loaded")

left_net = BoundNet(left_feature_count)
left_net.load_state_dict(torch.load("data_input/left_final_model_save_at_1_task_1.pt", map_location=lambda storage, loc: storage))

#print(net)
print("Left net parameters: \n")
print(left_net)
left_net.to(constant.device)
print("Left net loaded")
print("***********************")

right_net = BoundNet(right_feature_count)
right_net.load_state_dict(torch.load("data_input/right_final_model_save_at_1_task_1.pt", map_location=lambda storage, loc: storage))

#print(net)
print("Right net parameters: \n")
print(right_net)
right_net.to(constant.device)
print("Right net loaded")
print("***********************")

print("Initialized!")

# Initialize the first round
print("Generate first set up")
args = "./de_c_pure initialize".split()
args.append(str(constant.lattice_n))
args.append(str(constant.lattice_w))
args.append(str(constant.kT))
args.append(str(constant.dmp))
args.append(str(constant.dt))
args.append(str(0))
args.append(constant.init)
print(args)
popen_status = subprocess.Popen(args)
exit_code = popen_status.wait()
print("exit_status: " + str(exit_code))
print("First set up done!")

# simulation
print("Simulation start")

for round_n in range(constant.total_round):
    energy = generate_force()
    with open("energy.csv", "a") as energy_file:
        energy_file.write(str(round_n) + "," + str(energy) + "\n")
    args = "./de_c_pure calculate".split()
    args.append(str(constant.lattice_n))
    args.append(str(constant.lattice_w))
    args.append(str(constant.kT))
    args.append(str(constant.dmp))
    args.append(str(constant.dt))
    args.append(str(round_n))
    args.append(constant.init)
    args.append(str(random.randint(0, 1000000)))
    print(args)
    popen_status = subprocess.Popen(args)
    exit_code = popen_status.wait()
    print("exit_status: " + str(exit_code))
    if exit_code != 2:
        ValueError("force prediction collapsed.")
    print("round " + str(round_n) + " $$$$$$$$$$$$$$$$$$$$$$$")


print("Simulation done!")
