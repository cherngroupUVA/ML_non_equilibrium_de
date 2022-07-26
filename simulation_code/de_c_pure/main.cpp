#include <iostream>
#include "analysis.h"
#include <cassert>
#include "vec3.hpp"
#include "armadillo"
#include <string>
#include <fstream>

typedef std::mt19937 RNG;

int main(int argc, char* argv[]) {

    std::string status = argc > 1 ? argv[1] : "calculate";
    int lattice_length = argc > 2 ? atoi(argv[2]) : 32;
    int lattice_width = argc > 3 ? atoi(argv[3]) : 24;
    double _kT = argc > 4 ? atof(argv[4]) : 0.02;
    double _dmp = argc > 5 ? atof(argv[5]) : 0.55;
    double _dt = argc > 6 ? atof(argv[6]) : 0.5;
    int round = argc > 7 ? atoi(argv[7]) : 0;
    std::string init_file = argc > 8 ? argv[8] : "../data_input/c0.dat";
    int _seed = argc > 9 ? atoi(argv[9]) : 32768;
    
    class Simulation {
    public:
        int lx, ly;
        int Ns;
        vec3 *spin;
        vec3 *h_eff;
        vec3 *spin_prev;
        vec3 *h_eff_prev;
        int seed;
        double kT_spin = 0.0025;
        
        double kT;
        double damping;
        double dt;

        double gyro_ratio = 1;
        
        RNG rng;
        
        Simulation(int l_x, int l_y, int _seed) {
            lx = l_x;
            ly = l_y;
            Ns = lx * ly;
            spin = new vec3[Ns];
            h_eff = new vec3[Ns];
            spin_prev = new vec3[Ns];
            h_eff_prev = new vec3[Ns];
            seed = _seed;
            rng = RNG(seed);
        }
        
        void initialize(const std::string &init_file) {
            std::ofstream share_file("share_file.csv", std::ios::out);
            initial_from_file(init_file);
            arma::mat initial_set_up(Ns, 12);
            initial_set_up.zeros();
            for (int i = 0; i < Ns; ++i) {
                initial_set_up(i, 0) = spin[i].x;
                initial_set_up(i, 1) = spin[i].y;
                initial_set_up(i, 2) = spin[i].z;
                initial_set_up(i, 3) = spin[i].x;
                initial_set_up(i, 4) = spin[i].y;
                initial_set_up(i, 5) = spin[i].z;
                initial_set_up(i, 6) = h_eff[i].x;
                initial_set_up(i, 7) = h_eff[i].y;
                initial_set_up(i, 8) = h_eff[i].z;
                initial_set_up(i, 9) = h_eff[i].x;
                initial_set_up(i, 10) = h_eff[i].y;
                initial_set_up(i, 11) = h_eff[i].z;
//                if (i % lx < 5 || i % lx > 26) {
//                    initial_set_up(i, 6) = 0.0;
//                    initial_set_up(i, 7) = 0.0;
//                    initial_set_up(i, 8) = 0.0;
//                    initial_set_up(i, 9) = 0.0;
//                    initial_set_up(i, 10) = 0.0;
//                    initial_set_up(i, 11) = 0.0;
//                }
            }
            initial_set_up.save(share_file, arma::csv_ascii);
            share_file.close();
        }
        
        void initial_from_file(const std::string &filename) {
            arma::mat initial(Ns, 11);
            initial.load(filename);
            for (int i = 0; i < Ns; ++i) {
                spin[i].x = initial(i, 2);
                spin[i].y = initial(i, 3);
                spin[i].z = initial(i, 4);
                spin_prev[i].x = initial(i, 2);
                spin_prev[i].y = initial(i, 3);
                spin_prev[i].z = initial(i, 4);
                h_eff[i].x = initial(i, 5);
                h_eff[i].y = initial(i, 6);
                h_eff[i].z = initial(i, 7);
                h_eff_prev[i].x = initial(i, 5);
                h_eff_prev[i].y = initial(i, 6);
                h_eff_prev[i].z = initial(i, 7);
            }
        }
        
        void set_up() {
            arma::mat initial(Ns, 12);
            initial.load("share_file.csv", arma::csv_ascii);
            for (int i = 0; i < Ns; ++i) {
                spin_prev[i].x = initial(i, 0);
                spin_prev[i].y = initial(i, 1);
                spin_prev[i].z = initial(i, 2);
                spin[i].x = initial(i, 3);
                spin[i].y = initial(i, 4);
                spin[i].z = initial(i, 5);
                h_eff_prev[i].x = initial(i, 6);
                h_eff_prev[i].y = initial(i, 7);
                h_eff_prev[i].z = initial(i, 8);
                h_eff[i].x = initial(i, 9);
                h_eff[i].y = initial(i, 10);
                h_eff[i].z = initial(i, 11);
            }
        }
        
        void save_up() {
            arma::mat initial(Ns, 12);
            std::ofstream share_file("share_file.csv", std::ios::out);
            for (int i = 0; i < Ns; ++i) {
                initial(i, 0) = spin_prev[i].x;
                initial(i, 1) = spin_prev[i].y;
                initial(i, 2) = spin_prev[i].z;
                initial(i, 3) = spin[i].x;
                initial(i, 4) = spin[i].y;
                initial(i, 5) = spin[i].z;
                initial(i, 6) = h_eff_prev[i].x;
                initial(i, 7) = h_eff_prev[i].y;
                initial(i, 8) = h_eff_prev[i].z;
                initial(i, 9) = 0.0;
                initial(i, 10) = 0.0;
                initial(i, 11) = 0.0;
            }
            initial.save(share_file, arma::csv_ascii);
            share_file.close();
        }

        void update_spin() {
            std::normal_distribution<double> rd(0.0, 1.0);    // default mean = 0, var = 1

            for(int i = 0; i < Ns; i++) {
                vec3 h1;
                vec3 h2;
                h1 = h_eff[i] + damping * spin[i].cross(h_eff[i]);
                h2 = h_eff_prev[i] + damping * spin_prev[i].cross(h_eff_prev[i]);
                vec3 Hm = 0.5 * gyro_ratio * dt * (1.5 * h1 - 0.5 * h2);

                Hm.x += 0.5 * gyro_ratio * damping * sqrt(dt * kT_spin) * rd(rng);
                Hm.y += 0.5 * gyro_ratio * damping * sqrt(dt * kT_spin) * rd(rng);
                Hm.z += 0.5 * gyro_ratio * damping * sqrt(dt * kT_spin) * rd(rng);

                vec3 B = spin[i] -  spin[i].cross(Hm);
                vec3 tmp;
                double _A = 1./(1. + Hm.norm2());
                
                tmp.x = _A * (B.x * (1. + Hm.x * Hm.x) + Hm.y * B.z - Hm.z * B.y + Hm.x * (B.y * Hm.y + B.z * Hm.z));
                tmp.y = _A * (B.y * (1. + Hm.y * Hm.y) + Hm.z * B.x - Hm.x * B.z + Hm.y * (B.z * Hm.z + B.x * Hm.x));
                tmp.z = _A * (B.z * (1. + Hm.z * Hm.z) + Hm.x * B.y - Hm.y * B.x + Hm.z * (B.x * Hm.x + B.y * Hm.y));
//                if (i % lx < 5 || i % lx > 26) {
//                    tmp.x = spin[i].x;
//                    tmp.y = spin[i].y;
//                    tmp.z = spin[i].z;
//                }

                spin_prev[i] = spin[i];
                spin[i] = tmp;
                h_eff_prev[i] = h_eff[i];
            }
        }
        
        void save_configuration(int round) {
            std::ofstream screen_shot("snapshot_save/screenshot_" + std::to_string(round) + ".csv", std::ios::out);
            arma::mat to_save(Ns, 6);
            for (int i = 0; i < Ns; ++i) {
                to_save(i, 0) = spin[i].x;
                to_save(i, 1) = spin[i].y;
                to_save(i, 2) = spin[i].z;
                to_save(i, 3) = h_eff[i].x;
                to_save(i, 4) = h_eff[i].y;
                to_save(i, 5) = h_eff[i].z;
            }
            to_save.save(screen_shot, arma::csv_ascii);
            screen_shot.close();
            
        }
        
        ~Simulation() {
            delete [] spin;
            delete [] h_eff;
            delete [] spin_prev;
            delete [] h_eff_prev;
        }
    };
    
    Simulation system(lattice_length, lattice_width, _seed);
    system.kT = _kT;
    system.kT_spin = _kT;
    system.damping = _dmp;
    system.dt = _dt;

    if (status == "initialize") {
        system.initialize(init_file);
        return 1;
    }
    
    if (status == "calculate") {
        system.set_up();
        system.save_configuration(round);
        system.update_spin();
        system.save_up();
        return 2;
    }

    return 0;
}
