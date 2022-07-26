import help
from help import read_bond_list
from help import process_partial_bond_list
from help import size_l, size_w, cut, device
from test_bond_analysis import new_create_bond_matrix
from help import new_process_bond_list
from test_bond_analysis import new_create_force_matrix
from test_bond_analysis import new_create_bond_matrix_for_boundary
from test_bond_analysis import new_create_force_matrix_for_boundary

import torch
import torch.nn.functional as f
import torch.utils.data as data
import numpy as np

bench_mark_model = "model_save_2020032302.pt"
bond_list_input = "bond_5_3only_74_458_2432.csv"
bond_ramp = 74
training_start = 1
start_model = "model_save_2020032302.pt"
from_bench_mark = False
task = 3
previous_task = 2
from_before_training = True
from_where = 1
sample_period = 200

# data dir
input_config_data_dir = "../data/input_config/"
input_data_dir = "../data/random_init"
output_data_dir = "../data/input"
file_output_dir = "../data/output_files"

torch.manual_seed(22903)
torch.set_default_dtype(torch.float64)
with open(file_output_dir  + "/semantics.txt", "a") as outfile:
    st = "Using " + str(help.device)
    outfile.write(st)
    outfile.write("\n")
    outfile.close()


class Net(torch.nn.Module):
    def __init__(self, input_size):
        super(Net, self).__init__()
        self.input = torch.nn.Linear(input_size, 1024)
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


bond_list = read_bond_list(input_config_data_dir + bond_list_input, bond_ramp)
#partial_bond_list_total = help.new_process_boundary_bond_list(bond_list)
#n = len(partial_bond_list_total)
bond_list = new_process_bond_list(bond_list)

# dimension test part:
spin_tensor_test = torch.load(output_data_dir + "/test_spin_0125.pt").to(help.device)[0]
major_shape = new_create_bond_matrix(spin_tensor_test, bond_list).shape[1]
print(major_shape)
#other_shape = {}
#index_recorder = []
#for j, partial_bond_list in enumerate(partial_bond_list_total):
#    index = int(partial_bond_list[0][0][0])
#    index_recorder.append(index)
#    other_shape[index] = new_create_bond_matrix_for_boundary(spin_tensor_test, partial_bond_list, index).shape[1]


net = Net(major_shape)
#other_net = []
#for i in range(size_l):
#    if i in other_shape:
#        other_net.append(Net(other_shape[i]))
#
#print(net, other_net)

print(net)
#net.to(device)
if from_bench_mark:
    net.load_state_dict(torch.load("data_input/" + bench_mark_model))

if from_before_training:
    net.load_state_dict(torch.load(file_output_dir + "/final_model_save_at_1_task_2.pt"))
net.to(device)
#params = list(net.parameters())
#for i in range(len(other_net)):
#    params += list(other_net[i].parameters())

optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)
loss_func = torch.nn.MSELoss()

while training_start <= 20:
    with open(file_output_dir + "/task_" + str(task) + "_training_" + str(training_start) + ".txt", "w") as outfile:
        pass
    spin_tensor = torch.load(output_data_dir + "/train_spin_0125.pt").to(help.device)
    # print(new_create_bond_matrix(spin_tensor[0], bond_list).shape)
    force_tensor = torch.load(output_data_dir + "/train_force_0125.pt").to(help.device)
    # spin_tensor_2 = torch.load("training_data/kpm_spin_" + str(training_start) + ".pt").to(help.device)
    # force_tensor_2 = torch.load("training_data/kpm_force_" + str(training_start) + ".pt").to(help.device)
    # spin_tensor = torch.cat((spin_tensor_1, spin_tensor_2))
    # force_tensor = torch.cat((force_tensor_1, force_tensor_2))

    torch_data_set = data.TensorDataset(spin_tensor, force_tensor)
    loader = data.DataLoader(dataset=torch_data_set, batch_size=1, shuffle=False)

    for step, (spin, force) in enumerate(loader):
        temp_coordinate = spin[0].requires_grad_(True)
        temp_force = force[0].requires_grad_(False)
        
        pforce = temp_coordinate * torch.sum(temp_coordinate * temp_force, 1).reshape((-1, 1))
        temp_force = 100.0 * (temp_force - pforce)
        optimizer.zero_grad()
        
        x_temp = new_create_bond_matrix(temp_coordinate, bond_list)
        energy_prediction_temp = net(x_temp)
        
        energy_prediction = torch.sum(energy_prediction_temp, 0)
        energy_A = energy_prediction[0]
        energy_B = energy_prediction[1]
        print(energy_prediction, energy_A, energy_B)
        
        force_prediction_A = -torch.autograd.grad(energy_A, temp_coordinate, create_graph=True)[0]
        force_prediction_B = -torch.autograd.grad(energy_B, temp_coordinate, create_graph=True)[0]
        force_prediction_temp = force_prediction_A + torch.cross(temp_coordinate, force_prediction_B)
        force_prediction = 100.0 * (torch.cross(temp_coordinate, force_prediction_temp))
#        pforce = temp_coordinate * torch.sum(temp_coordinate * force_prediction_temp, 1).reshape((-1, 1))
#        ppforce = force_prediction_temp - pforce
#        print(ppforce.shape)
#        energy_A = energy_prediction[0]
#        energy_B = energy_prediction[1]
#        print(energy_prediction, energy_A, energy_B)
#
#        force_prediction_A = -torch.autograd.grad(energy_A, temp_coordinate, create_graph=True)[0]
#        force_prediction_B = -torch.autograd.grad(energy_B, temp_coordinate, create_graph=True)[0]
#        force_prediction = force_prediction_A + torch.cross(temp_coordinate, force_prediction_B)

        loss = loss_func(new_create_force_matrix(force_prediction), new_create_force_matrix(temp_force))

#        for i in range(len(other_net)):
#            x_t = new_create_bond_matrix_for_boundary(temp_coordinate, partial_bond_list_total[i], index_recorder[i])
#            energy_pt = other_net[i](x_t)
#            energy_p = torch.sum(energy_pt, 0)
#            e_A = energy_p[0]
#            e_B = energy_p[1]
#            force_A = -torch.autograd.grad(e_A, temp_coordinate, create_graph=True)[0]
#            force_B = -torch.autograd.grad(e_B, temp_coordinate, create_graph=True)[0]
#            f_p = force_A + torch.cross(temp_coordinate, force_B)
#            loss += loss_func(new_create_force_matrix_for_boundary(f_p, index_recorder[i]),
#                              new_create_force_matrix_for_boundary(temp_force, index_recorder[i]))
        # print(force_prediction, force_tensor[0], loss.cpu().detach().numpy())

        output_string = "Task: " + str(task) + "| Step: " + str(step) + '| Loss: ' + str(loss.cpu().detach().numpy())
        print(output_string)
        with open(file_output_dir + "/task_" + str(task) + "_training_" + str(training_start) + ".txt", "a") as outfile:
            outfile.write(output_string)
            outfile.write("\n")
            outfile.write("***************\n")

        loss.backward()
        optimizer.step()
        if step != 0 and step % sample_period == 0:
            temp_model_save_string = file_output_dir + "/status_model_save_at_" + str(training_start) + "_task_" + str(
                task) + "_step_" + str(step) + ".pt"
            torch.save(net.state_dict(), temp_model_save_string)
#            for i in range(len(other_net)):
#                other_temp_model_save_string = file_output_dir + "/" + str(i) + "_other_status_model_save_at_" + \
#                                               str(training_start) + "_task_" + str(task) + "_step_" + str(step) + ".pt"
#                torch.save(other_net[i].state_dict(), other_temp_model_save_string)


    model_save_string = file_output_dir + "/final_model_save_at_" + str(training_start) + "_task_" + str(task) + ".pt"
    torch.save(net.state_dict(), model_save_string)
#    for i in range(len(other_net)):
#        other_final_model_save_string = file_output_dir + "/" + str(i) + "_final_model_save_at_" + \
#                                        str(training_start) + "_task_" + str(task) + "_step_" + str(step) + ".pt"
#        torch.save(other_net[i].state_dict(), other_final_model_save_string)

    with open(file_output_dir + "/task_" + str(task) + "_training_" + str(training_start) + ".txt", "a") as outfile:
        outfile.write("\n\n\n")
        outfile.write("Below is for test!")
        outfile.write("\n")

    spin_tensor_test = torch.load(output_data_dir + "/test_spin_0125.pt").to(help.device)
    force_tensor_test = torch.load(output_data_dir + "/test_force_0125.pt").to(help.device)

    torch_data_set = data.TensorDataset(spin_tensor_test, force_tensor_test)
    loader = data.DataLoader(dataset=torch_data_set, batch_size=1, shuffle=False)

    for step, (spin, force) in enumerate(loader):
        temp_coordinate = spin[0].requires_grad_(True)
        temp_force = force[0].requires_grad_(False)
        
        pforce = temp_coordinate * torch.sum(temp_coordinate * temp_force, 1).reshape((-1, 1))
        temp_force = 100.0 * (temp_force - pforce)
        optimizer.zero_grad()
        
        x_temp = new_create_bond_matrix(temp_coordinate, bond_list)
        energy_prediction_temp = net(x_temp)
        
        energy_prediction = torch.sum(energy_prediction_temp, 0)
        energy_A = energy_prediction[0]
        energy_B = energy_prediction[1]
        print(energy_prediction, energy_A, energy_B)
        
        force_prediction_A = -torch.autograd.grad(energy_A, temp_coordinate, create_graph=True)[0]
        force_prediction_B = -torch.autograd.grad(energy_B, temp_coordinate, create_graph=True)[0]
        force_prediction_temp = force_prediction_A + torch.cross(temp_coordinate, force_prediction_B)
        force_prediction = 100.0 * (torch.cross(temp_coordinate, force_prediction_temp))

        loss = loss_func(new_create_force_matrix(force_prediction), new_create_force_matrix(temp_force))

#        for i in range(len(other_net)):
#            x_t = new_create_bond_matrix_for_boundary(temp_coordinate, partial_bond_list_total[i], index_recorder[i])
#            energy_pt = other_net[i](x_t)
#            energy_p = torch.sum(energy_pt, 0)
#            e_A = energy_p[0]
#            e_B = energy_p[1]
#            force_A = -torch.autograd.grad(e_A, temp_coordinate, create_graph=True)[0]
#            force_B = -torch.autograd.grad(e_B, temp_coordinate, create_graph=True)[0]
#            f_p = force_A + torch.cross(temp_coordinate, force_B)
#            loss += loss_func(new_create_force_matrix_for_boundary(f_p, index_recorder[i]),
#                              new_create_force_matrix_for_boundary(temp_force, index_recorder[i]))
#            with open(file_output_dir + "/other_" + str(i) + "_test_out_prediction_" + str(training_start) + ".csv", "ab") as output_file:
#                np.savetxt(output_file, new_create_force_matrix_for_boundary(f_p, index_recorder[i]).cpu().detach().numpy(), delimiter=",")
#                output_file.close()
#            with open(file_output_dir + "/other_" + str(i) + "_test_out_real_" + str(training_start) + ".csv", "ab") as output_file:
#                np.savetxt(output_file, new_create_force_matrix_for_boundary(temp_force, index_recorder[i]).cpu().detach().numpy(), delimiter=",")
#                output_file.close()

        with open(file_output_dir + "/test_out_prediction_" + str(training_start) + ".csv", "ab") as output_file:
            np.savetxt(output_file, new_create_force_matrix(force_prediction).cpu().detach().numpy(), delimiter=",")
            output_file.close()
        with open(file_output_dir + "/test_out_real_" + str(training_start) + ".csv", "ab") as output_file:
            np.savetxt(output_file, new_create_force_matrix(temp_force).cpu().detach().numpy(), delimiter=",")
            output_file.close()
        output_string = "Task: " + str(task) + "| Step: " + str(step) + '| Loss: ' + str(loss.cpu().detach().numpy())
        with open(file_output_dir + "/task_" + str(task) + "_training_" + str(training_start) + ".txt", "a") as outfile:
            outfile.write(output_string)
            outfile.write("\n")
            outfile.write("***************\n")

    outfile.close()

    training_start += 1
