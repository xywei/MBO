//
// Created by Xiaoyu Wei on 20/5/2016.
//

#ifndef MBOX_SIMULATOR_H
#define MBOX_SIMULATOR_H

#include "config.h"
#include "runtime_parameters.h"

// Switching initial conditions
#include "initial_bubble_in_the_air.h"

// Deal.II
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
#include <deal.II/numerics/matrix_tools.h>
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

        void initialize_simulator ();
        void setup_system ();
        void assemble_system ();
        void solve_system ();
        void apply_threshold ();
        void output_solution ();
        void refine_mesh ();

        Triangulation<dim> triangulation_;

        FE_Q<dim> fe_;
        DoFHandler<dim> dof_handler_;
        ConstraintMatrix constraints_;

        std::vector<IndexSet> partitioning_;

        TrilinosWrappers::SparseMatrix system_matrix_;
        TrilinosWrappers::SparseMatrix laplace_matrix_;
        TrilinosWrappers::SparseMatrix mass_matrix_;

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

        // Preconditioner
        std::shared_ptr<TrilinosWrappers::PreconditionAMG> AMG_preconditioner_;

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
    void Simulator<dim>::initialize_simulator(){
        GridGenerator::hyper_cube(triangulation_,
                /*left = */0.0, /*right = */1.0,
                /*colorize = */false);
        triangulation_.refine_global (prm_->n_initial_global_refines);

        setup_system();

        IndexSet cell_partition;
        const unsigned int max_grid_level = prm_->n_initial_global_refines + prm_->max_adaptation_level;
        const unsigned int min_grid_level = prm_->n_initial_global_refines;
        const unsigned int dofs_per_cell = fe_.dofs_per_cell;
        std::vector<unsigned int> dofs (dofs_per_cell);

        const QMidpoint<dim> quadrature_formula;
        const UpdateFlags update_flags = update_gradients;
        FEValues<dim> fe_values (fe_, quadrature_formula, update_flags);

        TrilinosWrappers::MPI::Vector criteria;
        std::vector<Tensor<1,dim>> du0 (1, Tensor<1,dim> ());
        std::vector<Tensor<1,dim>> du1 (1, Tensor<1,dim> ());
        std::vector<Tensor<1,dim>> du2 (1, Tensor<1,dim> ());

        for (unsigned int l=0; l<prm_->n_initial_adaptive_refines; l++) {
            VectorTools::interpolate (dof_handler_, InitialValues0<dim> (), u0_);
            VectorTools::interpolate (dof_handler_, InitialValues1<dim> (), u1_);
            VectorTools::interpolate (dof_handler_, InitialValues2<dim> (), u2_);

            constraints_.distribute (u0_);
            constraints_.distribute (u1_);
            constraints_.distribute (u2_);

            cell_partition = complete_index_set (triangulation_.n_active_cells());
            criteria.reinit (cell_partition, MPI_COMM_WORLD);

            // Calculate criteria
            {
                auto cell = dof_handler_.begin_active();
                auto endc = dof_handler_.end();
                for (unsigned int cell_no = 0; cell != endc; cell++, cell_no++) {
                    fe_values.reinit(cell);
                    fe_values.get_function_gradients(u0_, du0);
                    fe_values.get_function_gradients(u1_, du1);
                    fe_values.get_function_gradients(u2_, du2);

                    criteria(cell_no) = std::log(1 +
                                                 std::sqrt(du0[0] * du0[0]
                                                           + du1[0] * du1[0]
                                                           + du2[0] * du2[0]));
                }
            }

            // Mark cells to refine
            {
                auto cell = triangulation_.begin_active ();
                auto endc = triangulation_.end ();
                for (unsigned int cell_no=0; cell!=endc; cell++, cell_no++) {
                    if (criteria(cell_no) > 1e-2) {
                        cell->set_refine_flag ();
                    }
                    else if (criteria(cell_no) < 1e-12) {
                        cell->set_coarsen_flag ();
                    }
                }
            }

            // Apply level constraints
            if (triangulation_.n_levels() > max_grid_level)
                for (auto cell = triangulation_.begin_active(max_grid_level);
                     cell != triangulation_.end();
                     cell ++) {
                    cell->clear_refine_flag ();
            }
            for (auto cell = triangulation_.begin_active(min_grid_level);
                 cell != triangulation_.end();
                 cell ++ ) {
                cell->clear_coarsen_flag ();
            }

            triangulation_.prepare_coarsening_and_refinement ();
            triangulation_.execute_coarsening_and_refinement ();

            setup_system ();
        }

    }

    template <int dim>
    void Simulator<dim>::setup_system() {
        dof_handler_.distribute_dofs (fe_);

        constraints_.clear ();
        DoFTools::make_hanging_node_constraints (dof_handler_, constraints_);
        constraints_.close ();

        {
            DynamicSparsityPattern dsp (dof_handler_.n_dofs());
            DoFTools::make_sparsity_pattern (dof_handler_,
                                             dsp,
                                             constraints_,
                    /* keep_constrained_dofs = */ false);
            system_matrix_.reinit (dsp);
            laplace_matrix_.reinit (dsp);
            mass_matrix_.reinit (dsp);
        }

        AMG_preconditioner_.reset ();
        AMG_preconditioner_ = std::make_shared<TrilinosWrappers::PreconditionAMG> ();
        TrilinosWrappers::PreconditionAMG::AdditionalData AMG_data;
        AMG_data.elliptic = true;
        if (prm_->degree>1) {
            AMG_data.higher_order_elements = true;
        }
        AMG_data.smoother_sweeps = prm_->smoother_sweeps;
        AMG_data.aggregation_threshold = prm_->aggregation_threshold;
        AMG_preconditioner_->initialize (system_matrix_, AMG_data);

        partitioning_.resize (1);
        partitioning_[0] = complete_index_set (dof_handler_.n_dofs());
        u0_.reinit (partitioning_[0], MPI_COMM_WORLD);
        u1_.reinit (partitioning_[0], MPI_COMM_WORLD);
        u2_.reinit (partitioning_[0], MPI_COMM_WORLD);
        system_rhs0_.reinit (partitioning_[0], MPI_COMM_WORLD);
        system_rhs1_.reinit (partitioning_[0], MPI_COMM_WORLD);
        system_rhs2_.reinit (partitioning_[0], MPI_COMM_WORLD);

        deallog << "----------------------------------------" << std::endl
                << "Number of active cells: "
                << triangulation_.n_active_cells ()
                << " (on "
                << triangulation_.n_levels ()
                << " levels)"
                << std::endl
                << "Number of degrees of freedom: "
                << " (" << dof_handler_.n_dofs() << " * 3)" << std::endl
                << "----------------------------------------"
                << std::endl;
    }

    template <int dim>
    void Simulator<dim>::refine_mesh() {

    }

    template <int dim>
    void Simulator<dim>::apply_threshold() {

    }

    template <int dim>
    void Simulator<dim>::assemble_system() {
        MatrixTools::create_mass_matrix (dof_handler_,
                                         QGauss<dim> (fe_.degree + 1),
                                         mass_matrix_);
        MatrixCreator::create_laplace_matrix (dof_handler_,
                                              QGauss<dim> (fe_.degree + 1),
                                              laplace_matrix_);
        system_matrix_.copy_from (mass_matrix_);
        system_matrix_.add (prm_->theta * prm_->dt, laplace_matrix_);
        constraints_.condense (system_matrix_);
    }

    template <int dim>
    void Simulator<dim>::solve_system() {

    }

    template <int dim>
    void Simulator<dim>::run () {

        deallog << "Running the MBOX simulator." << std::endl;
        AssertThrow(timestep_number_==0, ExcNotInitialized());

        {
            LogStream::Prefix prefix ("MBOX");
            TrilinosWrappers::MPI::Vector tmp0;
            TrilinosWrappers::MPI::Vector tmp1;
            TrilinosWrappers::MPI::Vector tmp2;

            initialize_simulator ();

            const double dt = prm_->dt;
            const double ddt = prm_->dt / prm_->n_sub_steps;
            const double theta = prm_->theta;

            double time = 0;
            while (time < prm_->final_time) {
                time += dt;
                timestep_number_++;

                for (unsigned substep_number = 0;
                     substep_number<prm_->n_sub_steps;
                     substep_number++) {

                    mass_matrix_.vmult (system_rhs0_, u0_);
                    mass_matrix_.vmult (system_rhs1_, u1_);
                    mass_matrix_.vmult (system_rhs2_, u2_);

                    laplace_matrix_.vmult (tmp0, u0_);
                    laplace_matrix_.vmult (tmp1, u1_);
                    laplace_matrix_.vmult (tmp2, u2_);

                    system_rhs0_.add ( -(1.0 - theta) * ddt, tmp0 );
                    system_rhs1_.add ( -(1.0 - theta) * ddt, tmp1 );
                    system_rhs2_.add ( -(1.0 - theta) * ddt, tmp2 );

                    tmp0 = system_rhs0_;
                    tmp1 = system_rhs1_;
                    tmp2 = system_rhs2_;

                    constraints_.condense (tmp0, system_rhs0_);
                    constraints_.condense (tmp1, system_rhs1_);
                    constraints_.condense (tmp2, system_rhs2_);

                    solve_system ();
                }

                if (timestep_number_%prm_->adaptation_interval==0) {
                    refine_mesh ();
                }

                apply_threshold ();
            }
        }

    }

}



#endif //MBOX_SIMULATOR_H
