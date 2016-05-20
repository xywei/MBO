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

        // Equation
        double theta_s;

        // MBO
        double dt;

        // Heat eqn
        unsigned int degree;
        unsigned int n_sub_steps;
        double eps;

        // Simulator
        double initial_time;
        double final_time;

        // AMR
        unsigned int n_initial_global_refines;
        unsigned int n_initial_adaptive_refines;
        unsigned int max_adaptation_level;
        unsigned int adaptation_interval;

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
        prm->leave_subsection ();

        prm->enter_subsection ("Algorithm");
        prm->declare_entry ("initial_time", "0.",
                           Patterns::Double (0.),
                           " The initial time of the simulation. ");
        prm->declare_entry ("final_time", "1.",
                           Patterns::Double (0.),
                           " The final time of the simulation. ");
        prm->declare_entry ("dt", "0.001",
                            Patterns::Double(0.),
                            "The time step size for MBO." );
        prm->declare_entry ("n_sub_steps", "1",
                            Patterns::Integer(1,100),
                            "The number of time steps to take to solve heat equation in each MBO steps.");
        prm->declare_entry ("theta_s", "60",
                            Patterns::Double(0.),
                            "The static contact angle, in degree." );

        prm->declare_entry ("adaptation_interval", "3",
                            Patterns::Integer (1, 100),
                            "The number of MBO steps between two AMRs.");
        prm->declare_entry ("max_adaptation_level", "10",
                            Patterns::Integer (0, 20),
                            "Maximal difference in level between the most coarse and the most fine cells.");
        prm->declare_entry ("n_initial_global_refines", "3",
                           Patterns::Integer (0, 15),
                           "The number of global refines we do on the mesh before start adaptive refinement. ");
        prm->declare_entry ("n_initial_adaptive_refines", "3",
                            Patterns::Integer (0, 15),
                            "The number of adaptive refines we do on the globally refined mesh before start of time integration.");
        prm->declare_entry ("degree", "1",
                           Patterns::Integer (1,5),
                           "The polynomial degree for the finite element space." );
        prm->declare_entry ("eps", "1e-12",
                           Patterns::Double (0.),
                           "The stopping criterion for linear solvers.");
        prm->leave_subsection ();

        prm->enter_subsection ("IO");
        prm->declare_entry ("verbose", "true",
                           Patterns::Bool(),
                           " This indicates whether the output of the solution "
                                   "process should be verbose. ");

        prm->declare_entry ("output_interval", "1",
                           Patterns::Integer(1),
                           " This indicates between how many time steps we print "
                                   "the solution. ");
        prm->leave_subsection ();

    }

    void RuntimeParameters::parse_parameters () {
        prm->enter_subsection ("Geometry");
        {
            dim = prm->get_integer ("dim");
        }
        prm->leave_subsection ();

        prm->enter_subsection ("Algorithm");
        {
            theta_s = prm->get_double ("theta_s");

            dt = prm->get_double ("dt");
            initial_time = prm->get_double ("initial_time");
            final_time = prm->get_double ("final_time");

            degree = prm->get_integer ("degree");
            n_sub_steps = prm->get_integer ("n_sub_steps");
            eps = prm->get_double ("eps");

            n_initial_global_refines = prm->get_integer ("n_initial_global_refines");
            n_initial_adaptive_refines = prm->get_integer ("n_initial_adaptive_refines");
            max_adaptation_level = prm->get_integer ("max_adaptation_level");
            adaptation_interval = prm->get_integer ("adaptation_interval");
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
