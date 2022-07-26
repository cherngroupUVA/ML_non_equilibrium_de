""" Analysis functions for generate irreducible representations bond """
import torch
import help
from help import size_l, size_w, cut


def generate_bond_features(spin_tensor, index, bond_list):
    bond_feature = [torch.tensor([index % help.size_l], dtype=torch.float64, device=help.device)]
    for i in range(1, len(bond_list[0])):
        test = torch.index_select(spin_tensor, 0, bond_list[index][i])
        bond_feature.append(torch.sum(test[::2] * test[1::2], 1))
        if bond_list[index][i][0] == index:
            bond_feature.append(torch.sum(test[::2] * torch.cross(test[1::2], torch.roll(test[1::2], 1, 0)), 1))

    return torch.cat(bond_feature).reshape((1, -1))


def generate_bond_features_for_boundary(spin_tensor, index, true_index, bond_list):
    bond_feature = []
    for i in range(1, len(bond_list[0])):
        test = torch.index_select(spin_tensor, 0, bond_list[index][i])
        bond_feature.append(torch.sum(test[::2] * test[1::2], 1))
        if bond_list[index][i][0] == true_index:
            bond_feature.append(torch.sum(test[::2] * torch.cross(test[1::2], torch.roll(test[1::2], 1, 0)), 1))

    return torch.cat(bond_feature).reshape((1, -1))


def generate_vectors(spin_tensor, index, bond_list):
    itself = torch.index_select(spin_tensor, 0, bond_list[index][0])
    test = torch.index_select(spin_tensor, 0, bond_list[index][1])
    test = test[1::2]
    test = torch.sum(test, 0).reshape((1, -1))
    
    first = torch.cross(itself, test)
    first = first / torch.norm(first)
    second = torch.cross(itself, first)
    
    return itself, first, second


def new_create_bond_matrix(spin_tensor, bond_list):
    bond_feature_for_all = []
    for i in range(size_l * size_w):
        x = i % help.size_l
        if x < cut or x > size_l - cut - 1:
            continue
        bond_feature_for_all.append(generate_bond_features(spin_tensor, i, bond_list))

    return torch.cat(bond_feature_for_all)


def new_create_bond_matrix_for_boundary(spin_tensor, bond_list, cut_index):
    bond_feature_for_all = []
    for i in range(len(bond_list)):
        bond_feature_for_all.append(generate_bond_features_for_boundary(spin_tensor, i, i * size_l + cut_index, bond_list))

    return torch.cat(bond_feature_for_all)
    

def new_create_force_matrix(force_tensor):
    bond_feature_for_all = []
    for i in range(help.size_l * help.size_w):
        x = i % help.size_l
        if x < cut or x > size_l - cut - 1:
            continue
        bond_feature_for_all.append(force_tensor[i].reshape(1, -1))

    return torch.cat(bond_feature_for_all)


def new_create_force_matrix_for_boundary(force_tensor, cut_index):
    bond_feature_for_all = []
    for i in range(help.size_l * help.size_w):
        x = i % help.size_l
        if x != cut_index:
            continue
        bond_feature_for_all.append(force_tensor[i].reshape(1, -1))

    return torch.cat(bond_feature_for_all)


