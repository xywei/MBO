//
// Created by Xiaoyu Wei on 20/5/2016.
//

#ifndef MBOX_RUNTIME_PARAMETERS_H
#define MBOX_RUNTIME_PARAMETERS_H

#include "config.h"
#include <deal.II/base/parameter_handler.h>

namespace mbox {

    using namespace dealii;

    class RuntimeParameters {
    public:
        RuntimeParameters ();
        void parse_parameters ();

        // Geometry
        unsigned dim;
        unsigned initial_condition;

        // Equation
        double theta_s;
        bool conservative;

        // MBO
        double dt;

        // Heat eqn
        double theta;
        unsigned int degree;
        unsigned int n_sub_steps;
        double eps;

        // Simulator
        double initial_time;
        double final_time;

        // AMG preconditioner
        unsigned int smoother_sweeps;
        double aggregation_threshold;

        // AMR
        unsigned int n_initial_global_refines;
        unsigned int n_initial_adaptive_refines;
        unsigned int max_adaptation_level;
        unsigned int adaptation_interval;
        unsigned int n_adaptation_sweeps;

        // I/O
        bool verbose;
        unsigned int output_interval;

        std::shared_ptr<ParameterHandler> prm;
    };

    /**
     * Declare parameters in the constructor.
     */
    RuntimeParameters::RuntimeParameters () :
            prm ( std::make_shared<ParameterHandler> () ) {
        prm->enter_subsection ("Geometry");
        prm->declare_entry ("dim", "2",
                            Patterns::Integer(1, 3),
                            "The dimension of computational domain.");
        prm->declare_entry("initial_profile", "elliptic_drop_in_the_air",
                           Patterns::Selection ("elliptic_drop_in_the_air|"
                                                        "drop_on_flat_surface"));
        prm->leave_subsection ();

        prm->enter_subsection ("Algorithm");
        prm->declare_entry ("conservative", "true",
                            Patterns::Bool(),
                            "Whether threshold preserves volume of each phase (Allen-Cahn vs Cahn-Hilliard).");
        prm->declare_entry ("initial_time", "0.",
                           Patterns::Double (0.),
                           "The initial time of the simulation. ");
        prm->declare_entry ("final_time", "1.",
                           Patterns::Double (0.),
                           "The final time of the simulation. ");
        prm->declare_entry ("dt", "0.001",
                            Patterns::Double(0.),
                            "The time step size for MBO." );
        prm->declare_entry ("n_sub_steps", "1",
                            Patterns::Integer(1,100),
                            "The number of time steps to take to solve heat equation in each MBO steps.");
        prm->declare_entry ("theta", "1.0",
                            Patterns::Double(0.),
                            "The parameter for theta-scheme, can be explicit Euler (0), implicit Euler (1), Crank-Nicolson (0.5), etc.");
        prm->declare_entry ("theta_s", "60",
                            Patterns::Double(0.),
                            "The static contact angle, in degree." );

        prm->declare_entry ("adaptation_interval", "5",
                            Patterns::Integer (1, 100),
                            "The number of MBO steps between two AMRs.");
        prm->declare_entry ("n_adaptation_sweeps", "5",
                            Patterns::Integer (1, 100),
                            "The number of sweeps for each adaptation.");
        prm->declare_entry ("max_adaptation_level", "8",
                            Patterns::Integer (0, 20),
                            "Maximal difference in level between the most coarse and the most fine cells.");
        prm->declare_entry ("n_initial_global_refines", "2",
                           Patterns::Integer (0, 15),
                           "The number of global refines we do on the mesh before start adaptive refinement. ");
        prm->declare_entry ("n_initial_adaptive_refines", "8",
                            Patterns::Integer (0, 15),
                            "The number of adaptive refines we do on the globally refined mesh before start of time integration.");
        prm->declare_entry ("degree", "1",
                           Patterns::Integer (1,5),
                           "The polynomial degree for the finite element space." );
        prm->declare_entry ("smoother_sweeps", "2",
                            Patterns::Integer (1, 100),
                            "The number of smoother sweeps used in AMG preconditioner.");
        prm->declare_entry ("aggregation_threshold", "0.02",
                            Patterns::Double (0.),
                            "(ad hoc) Larger aggregation_threshold will decrease the number of iterations, but increase the costs per iteration");
        prm->declare_entry ("eps", "1e-12",
                           Patterns::Double (0.),
                           "The stopping criterion for linear solvers.");
        prm->leave_subsection ();

        prm->enter_subsection ("IO");
        prm->declare_entry ("verbose", "true",
                           Patterns::Bool(),
                           " This indicates whether the output of the solution "
                                   "process should be verbose. ");

        prm->declare_entry ("output_interval", "10",
                           Patterns::Integer(1),
                           " This indicates between how many time steps we print "
                                   "the solution. ");
        prm->leave_subsection ();

    }

    void RuntimeParameters::parse_parameters () {
        prm->enter_subsection ("Geometry");
        {
            dim = prm->get_integer ("dim");
            std::string init = prm->get("initial_profile");
            if (init.compare("elliptic_drop_in_the_air")==0) {
                initial_condition = 0;
            }
            else if (init.compare("drop_on_flat_surface")==0) {
                initial_condition = 1;
            }
            else {
                AssertThrow(false, ExcNotImplemented());
            }
        }
        prm->leave_subsection ();

        prm->enter_subsection ("Algorithm");
        {
            conservative = prm->get_bool ("conservative");
            theta_s = prm->get_double ("theta_s");

            theta = prm->get_double ("theta");
            dt = prm->get_double ("dt");
            initial_time = prm->get_double ("initial_time");
            final_time = prm->get_double ("final_time");

            degree = prm->get_integer ("degree");
            n_sub_steps = prm->get_integer ("n_sub_steps");
            eps = prm->get_double ("eps");

            smoother_sweeps = prm->get_integer("smoother_sweeps");
            aggregation_threshold = prm->get_double("aggregation_threshold");

            n_initial_global_refines = prm->get_integer ("n_initial_global_refines");
            n_initial_adaptive_refines = prm->get_integer ("n_initial_adaptive_refines");
            max_adaptation_level = prm->get_integer ("max_adaptation_level");
            adaptation_interval = prm->get_integer ("adaptation_interval");
            n_adaptation_sweeps = prm->get_integer ("n_adaptation_sweeps");
        }
        prm->leave_subsection ();

        prm->enter_subsection ("IO");
        {
            verbose = prm->get_bool ("verbose");
            output_interval = prm->get_integer ("output_interval");
        }
        prm->leave_subsection ();


    }
}

#endif //MBOX_RUNTIME_PARAMETERS_H
