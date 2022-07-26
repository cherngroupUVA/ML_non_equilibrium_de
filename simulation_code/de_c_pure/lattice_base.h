//
// Created by Puhan Zhang on 2/18/20.
//

#ifndef DE_C_PURE_LATTICE_BASE_H
#define DE_C_PURE_LATTICE_BASE_H

#include <cmath>
#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <unordered_set>
#include <set>
#include <algorithm>
#include <random>
#include <ctime>
#include <chrono>
#include "armadillo"

const double PI = 3.141592653589793238463;

class Square_Site {
protected:
    int index;
    int x_coordinate, y_coordinate;
    double theta, phi;
    double spin[3] = {0};
    double force[3] = {0};
    double c_vector[3] = {0};
    double t_vector[3] = {0};
    double intermediate_spin[3] = {0};
    double intermediate_force[3] = {0};
    double intermediate_t_vector[3] = {0};

public:
    public:
        int neighbor[4] = {0};
        int type = 0;
        int rank = 0;
        int parent = 0;
        double correlation = 0.0;
    Square_Site(const int lattice_index, const int latticeLength, const int latticeWidth) {
            set_x(lattice_index % latticeLength);
            set_y(lattice_index / latticeLength);
            index = lattice_index;
            parent = lattice_index;
            set_neighbors(latticeLength, latticeWidth);
        }
        
        int mod(int a, int b)
        {
            int r = a % b;
            return r < 0 ? r + b : r;
        }
        
        void set_neighbors(int lattice_n, int lattice_w) {
            int row = index / lattice_n;
            int col = index - row * lattice_n;
            
            int candidate_col;
            // case 1, right
            candidate_col = col + 1;
            if (candidate_col < lattice_n) {
                neighbor[0] = lattice_n * row + mod(col + 1, lattice_n);
            }
            else {
                neighbor[0] = -1;
            }
            
            // case 2, up
            neighbor[1] = mod(row + 1, lattice_w) * lattice_n + col;
            
            // case 3, left
            candidate_col = col - 1;
            if (candidate_col >= 0) {
                neighbor[2] = lattice_n * row + mod(col - 1, lattice_n);
            }
            else {
                neighbor[2] = -1;
            }
            
            // case 4, down
            neighbor[3] = mod(row - 1, lattice_w) * lattice_n + col;
        }
        
        int get_neighbor(int i) {
            return neighbor[i];
        }
        
        void set_x(int coordinate) {
            x_coordinate = coordinate;
        }
        void set_y(int coordinate) {
            y_coordinate = coordinate;
        }

        inline int get_index() { return index; }
        inline int get_x() { return x_coordinate; };
        inline int get_y() { return y_coordinate; };

        inline double get_theta() { return theta; }
        inline double get_phi() { return phi; }
        inline int get_parent() { return parent; }
        inline int get_rank() { return rank; }
        inline int get_type() { return type; }

        inline void set_theta(double theta_parameter) { theta = theta_parameter; }
        inline void set_phi(double phi_parameter) { phi = phi_parameter; }
        
        void set_parent(int p) {
            parent = p;
        }
        
        void set_type(int t) {
            type = t;
        }
        
        void set_rank(int r) {
            rank = r;
        }

        void set_spins(double spin_x, double spin_y, double spin_z) {
            spin[0] = spin_x;
            spin[1] = spin_y;
            spin[2] = spin_z;
        }

        void set_force(double force_x, double force_y, double force_z) {
            force[0] = force_x;
            force[1] = force_y;
            force[2] = force_z;
        }

        void set_i_force(double force_x, double force_y, double force_z) {
            intermediate_force[0] = force_x;
            intermediate_force[1] = force_y;
            intermediate_force[2] = force_z;
        }

        void set_c_vector(double c_x, double c_y, double c_z) {
            c_vector[0] = c_x;
            c_vector[1] = c_y;
            c_vector[2] = c_z;
        }

        void set_t_vector(double t_x, double t_y, double t_z) {
            t_vector[0] = t_x;
            t_vector[1] = t_y;
            t_vector[2] = t_z;
        }

        void set_i_t_vector(double t_x, double t_y, double t_z) {
            intermediate_t_vector[0] = t_x;
            intermediate_t_vector[1] = t_y;
            intermediate_t_vector[2] = t_z;
        }

        void set_i_spin_vector(double s_x, double s_y, double s_z) {
            intermediate_spin[0] = s_x;
            intermediate_spin[1] = s_y;
            intermediate_spin[2] = s_z;
        }

        inline double get_spin_x() { return spin[0]; }
        inline double get_spin_y() { return spin[1]; }
        inline double get_spin_z() { return spin[2]; }

        inline double get_force_x() { return force[0]; }
        inline double get_force_y() { return force[1]; }
        inline double get_force_z() { return force[2]; }

        inline double get_c_x() { return c_vector[0]; }
        inline double get_c_y() { return c_vector[1]; }
        inline double get_c_z() { return c_vector[2]; }

        inline double get_t_x() { return t_vector[0]; }
        inline double get_t_y() { return t_vector[1]; }
        inline double get_t_z() { return t_vector[2]; }

        inline double get_i_t_x() { return intermediate_t_vector[0]; }
        inline double get_i_t_y() { return intermediate_t_vector[1]; }
        inline double get_i_t_z() { return intermediate_t_vector[2]; }

        inline double get_i_spin_x() { return intermediate_spin[0]; }
        inline double get_i_spin_y() { return intermediate_spin[1]; }
        inline double get_i_spin_z() { return intermediate_spin[2]; }

        inline double get_i_force_x() { return intermediate_force[0]; }
        inline double get_i_force_y() { return intermediate_force[1]; }
        inline double get_i_force_z() { return intermediate_force[2]; }

    ~Square_Site() = default;

};

void generate_random_spins(std::vector<Square_Site>& test_lattice) {
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();

    std::default_random_engine generator(seed);
    std::uniform_real_distribution<double> distribution(0.0,1.0);
    for (auto &site : test_lattice) {
        site.set_theta(2.0 * PI * distribution(generator));
        site.set_phi(std::acos(2.0 * distribution(generator) - 1));
        site.set_spins(std::sin(site.get_phi()) * std::cos(site.get_theta()),
                       std::sin(site.get_phi()) * std::sin(site.get_theta()), std::cos(site.get_phi()));
    }
}

void initial_from_file(std::vector<Square_Site>& test_lattice, const std::string &filename) {
    arma::mat initial(test_lattice.size(), 12);
    initial.load(filename, arma::csv_ascii);
    for (u_long i = 0; i < test_lattice.size(); ++i) {
        test_lattice[i].set_spins(initial.at(i, 2), initial.at(i, 3), initial.at(i, 4));
    }
}

#endif //DE_C_PURE_LATTICE_BASE_H
