import csv
from os import listdir
from os.path import isfile, join
import pandas as pd
import torch
#from natsort import natsorted
import collections
import joblib
import numpy

size_l = 32
size_w = 24
cut = 5

# device = torch.device('cuda')
device = torch.device('cpu')
print('Using device:', device)

if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_cached(0) / 1024 ** 3, 1), 'GB')


def read_bond_list(file_name, neighbor_size):
    with open(file_name, 'r') as f:
        reader = csv.reader(f)
        neighbor_list = list(reader)

    convert_list = []
    temp_neighbor_list = []
    neighbor_counter = 0
    for i in range(len(neighbor_list)):
        temp_neighbor_list.append(list(map(int, neighbor_list[i])))
        neighbor_counter += 1
        if neighbor_counter == neighbor_size + 1:
            convert_list.append(temp_neighbor_list)
            neighbor_counter = 0
            temp_neighbor_list = []

    return convert_list


def read_file_folders(directory):
    file_names = [directory + "/" + f for f in listdir(directory) if not f.startswith(".")]
    return natsorted(file_names)


def new_process_bond_list(bond_list):
    all_site = []
    for i in range(len(bond_list)):
        each_site = []
        for j in range(len(bond_list[0])):
            each_site.append(torch.tensor(bond_list[i][j], device=device))

        all_site.append(each_site)

    return all_site


def process_partial_bond_list(bond_list, cut_index):
    left_range = max(0, cut_index - cut)
    right_range = min(size_l - 1, cut_index + cut)
    focus_list = []
    for b in bond_list:
        if b[0][0] % size_l == cut_index:
            focus_list.append(b)

    real_list = []
    for b in focus_list:
        temp_list = []
        for i, l in enumerate(b):
            if i == 0:
                temp_list.append(l)
                continue
            cut_list = []
            for a_i, b_i in zip(l[::2], l[1::2]):
                if (left_range <= a_i % size_l <= right_range) and (left_range <= b_i % size_l <= right_range):
                    cut_list.append(a_i)
                    cut_list.append(b_i)
            if cut_list:
                temp_list.append(cut_list)
        real_list.append(temp_list)

    return real_list


def new_process_boundary_bond_list(bond_list):
    cut_index = []
    for i in range(size_l):
        if i < cut or i > size_l - cut - 1:
            cut_index.append(i)

    b_bond_list = []
    for index in cut_index:
        b_bond_list.append(new_process_bond_list(process_partial_bond_list(bond_list, index)))

    return b_bond_list


def read_file_names_random(directory):
    locations = read_file_folders(directory)
    train, test = locations[: int(0.8 * len(locations))], locations[int(0.8 * len(locations)):]
    train = [loc + "/c0.dat" for loc in train]
    test = [loc + "/c0.dat" for loc in test]
    return train, test


def read_data(file_name):
    data_raw = pd.read_csv(file_name, header=None, delimiter="\t")
    data_raw = data_raw[data_raw.iloc[:, 0] <= 15].values
    position = data_raw[:, 0:2]
    spin_component = torch.tensor(data_raw[:, 2:5], dtype=torch.float64, device=device)
    force_component = torch.tensor(data_raw[:, 5:8], dtype=torch.float64, device=device)
    return position, spin_component, force_component


def generate_data_matrix(input_data_dir, output_data_dir, ty="train"):
    fi = 0
    if ty == "test":
        fi = 1
    spin_file_names = read_file_names_random(input_data_dir)
    spin_tensor = []
    force_tensor = []
    for i in range(len(spin_file_names[fi])):
        position, spin_component, force_component = read_data(spin_file_names[fi][i])
        spin_tensor.append(spin_component)
        force_tensor.append(force_component)

    spin_part = torch.cat(tuple(spin_tensor), 0).reshape((-1, size_l * size_w, 3))
    force_part = torch.cat(tuple(force_tensor), 0).reshape((-1, size_l * size_w, 3))
    print(force_part.shape)

    torch.save(spin_part, output_data_dir + "/" + ty + "_spin_short.pt")
    torch.save(force_part, output_data_dir + "/" + ty + "_force_short.pt")
    print(ty + " tensor_saved!")
