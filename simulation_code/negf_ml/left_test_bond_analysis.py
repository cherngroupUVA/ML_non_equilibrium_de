""" Analysis functions for generate irreducible representations bond on left side """
import torch
from constant import lattice_n as size_l, lattice_w as size_w, cut, device
import csv


def read_bond_list(file_name):
    with open(file_name, 'r') as f:
        reader = csv.reader(f)
        neighbor_list = list(reader)
    convert_list = []
    bond_count = 0
    chirality_count = 0
    bond_c = 0
    chirality_c = 0
    start_index = None
    index = None
    
    for i in range(len(neighbor_list)):
        if neighbor_list[i][-1] == "":
            neighbor_list[i] = neighbor_list[i][:-1]
            
        temp = list(map(int, neighbor_list[i]))
        if i == 0:
            start_index = temp[0]
        if len(temp) == 1:
            convert_list.append([])
            index = temp[0]
        if len(temp) != 1 and index == start_index:
            bond_count += 1
            bond_c += len(temp) // 2
            if temp[0] != index:
                chirality_count += 1
                chirality_c += len(temp) // 2
                
        convert_list[-1].append(temp)

    return convert_list, bond_count, chirality_count, bond_c, chirality_c
    
    
def new_process_bond_list(bond_list):
    all_site = []
    for i in range(len(bond_list)):
        each_site = []
        for j in range(len(bond_list[0])):
            each_site.append(torch.tensor(bond_list[i][j], device=device))

        all_site.append(each_site)

    return all_site


def generate_bond_features(spin_tensor, index, bond_list, bond_start):
    bond_feature = [torch.tensor([bond_list[0][0][0] % size_l], dtype=torch.float64, device=device)]
    base = torch.index_select(spin_tensor, 0, bond_list[index][0])
    for i in range(1, len(bond_list[0])):
        test = torch.index_select(spin_tensor, 0, bond_list[index][i])
        bond_feature.append(torch.sum(test[::2] * test[1::2], 1))
        if i >= bond_start + 1:
            size = test.shape[0]
            bond_feature.append(torch.sum(base.repeat(size // 2, 1) * torch.cross(test[::2], test[1::2]), 1))

    return torch.cat(bond_feature).reshape((1, -1))


def new_create_bond_matrix(spin_tensor, bond_list, bond_count, chirality_count):
    bond_feature_for_all = []
    for i in range(size_w):
        bond_feature_for_all.append(generate_bond_features(spin_tensor, i, bond_list, bond_count - chirality_count))

    return torch.cat(bond_feature_for_all)


def new_create_bond_matrix_for_boundary(spin_tensor, bond_list, cut_index):
    bond_feature_for_all = []
    for i in range(len(bond_list)):
        bond_feature_for_all.append(generate_bond_features_for_boundary(spin_tensor, i, i * size_l + cut_index, bond_list))

    return torch.cat(bond_feature_for_all)
    

def new_create_force_matrix(force_tensor):
    bond_feature_for_all = []
    for i in range(size_l * size_w):
        x = i % size_l
        if x > cut:
            continue
        bond_feature_for_all.append(force_tensor[i].reshape(1, -1))

    return torch.cat(bond_feature_for_all)
