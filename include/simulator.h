//
// Created by Xiaoyu Wei on 20/5/2016.
//

#ifndef MBOX_SIMULATOR_H
#define MBOX_SIMULATOR_H

#include "config.h"
#include "runtime_parameters.h"

// Switching initial conditions
#include "initial_bubble_in_the_air.h"

#include <deal.II/base/utilities.h>

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/constraint_matrix.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_refinement.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/solution_transfer.h>

// Interface to Trilinos
#include <deal.II/base/index_set.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_block_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/lac/trilinos_block_vector.h>
#include <deal.II/lac/trilinos_precondition.h>

namespace mbox {

    using namespace dealii;

    template <int dim>
    class Simulator {
    public:
        Simulator (std::shared_ptr<RuntimeParameters>);

        void run ();

    private:
        // Construction must have parameters
        Simulator ();

        void setup_system ();
        void assemble_system ();
        void solve_heat_diffusion ();
        void apply_thresholding ();
        void output_solution ();
        void refine_mesh ();

        Triangulation<dim> triangulation_;

        FE_Q<dim> fe_;
        DoFHandler<dim> dof_handler_;
        ConstraintMatrix constraints_;

        TrilinosWrappers::SparseMatrix system_matrix_;
        TrilinosWrappers::MPI::Vector u0_;
        TrilinosWrappers::MPI::Vector u1_;
        TrilinosWrappers::MPI::Vector u2_;
        TrilinosWrappers::MPI::Vector system_rhs0_;
        TrilinosWrappers::MPI::Vector system_rhs1_;
        TrilinosWrappers::MPI::Vector system_rhs2_;

        unsigned int timestep_number_;

        std::shared_ptr<TrilinosWrappers::PreconditionAMG> preconditioner_;

        // Conserved volumes of each phase
        std::vector<double> volume_;
        std::shared_ptr<RuntimeParameters> prm_;

    };
    template class Simulator<1>;
    template class Simulator<2>;
    template class Simulator<3>;

    template <int dim>
    Simulator<dim>::Simulator (std::shared_ptr<RuntimeParameters> p) :
            triangulation_ (Triangulation<dim>::maximum_smoothing),
            fe_ (p->degree),
            dof_handler_ (triangulation_),
            timestep_number_ (0),
            volume_ (3),
            prm_ (p)
    { }

    template <int dim>
    void Simulator<dim>::run () {

        deallog << "Running the simulator." << std::endl;

    }

}



#endif //MBOX_SIMULATOR_H
