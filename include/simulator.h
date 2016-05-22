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

#include <deal.II/distributed/solution_transfer.h>
#include <deal.II/distributed/tria.h>

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
        void refine_mesh ();
        void setup_system ();
        void assemble_system_matrix (const double);
        void assemble_system_rhs    (const double);
        void solve_system ();
        void apply_threshold ();
        void apply_conservative_threshold ();

        void output_solution ();
        void print_system_info ();
        void print_volume_info (const std::vector<double> &);

        void compute_volumes (std::vector<double> &);

        parallel::distributed::Triangulation<dim> triangulation_;

        FE_Q<dim> fe_;
        DoFHandler<dim> dof_handler_;
        ConstraintMatrix constraints_;

        std::vector<IndexSet> partitioning_;

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

        // Preconditioner
        std::shared_ptr<TrilinosWrappers::PreconditionAMG> AMG_preconditioner_;

    };
    template class Simulator<2>;
    template class Simulator<3>;

    template <int dim>
    Simulator<dim>::Simulator (std::shared_ptr<RuntimeParameters> p) :
            triangulation_ (MPI_COMM_WORLD, Triangulation<dim>::maximum_smoothing),
            fe_ (p->degree),
            dof_handler_ (triangulation_),
            timestep_number_ (0),
            volume_ (3),
            prm_ (p)
    { }

    template <int dim>
    void Simulator<dim>::initialize_simulator(){

        LogStream::Prefix prefix ("INIT");

        GridGenerator::hyper_cube(triangulation_,
                /*left = */0.0, /*right = */1.0,
                /*colorize = */false);
        triangulation_.refine_global (prm_->n_initial_global_refines);

        setup_system();

        IndexSet cell_partition;
        const unsigned int max_grid_level = prm_->n_initial_global_refines + prm_->max_adaptation_level;
        const unsigned int min_grid_level = prm_->n_initial_global_refines;
        const unsigned int dofs_per_cell = fe_.dofs_per_cell;

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

            // Re-setup
            setup_system ();
        }

        // Apply initial conditions
        VectorTools::interpolate (dof_handler_, InitialValues0<dim> (), u0_);
        VectorTools::interpolate (dof_handler_, InitialValues1<dim> (), u1_);
        VectorTools::interpolate (dof_handler_, InitialValues2<dim> (), u2_);

        deallog << "Simulator initialized." << std::endl;
    }

    template <int dim>
    void Simulator<dim>::compute_volumes (std::vector<double> & volumes) {

        AssertThrow(volumes.size ()==3, ExcNotInitialized ());

        double v0 = 0.0;
        double v1 = 0.0;
        double v2 = 0.0;

        const QGauss<dim> quadrature_formula (prm_->degree + 1);
        FEValues<dim> fe_values (fe_, quadrature_formula,
                                 update_values | update_JxW_values);

        const unsigned int dofs_per_cell = fe_.dofs_per_cell;
        const unsigned int n_q_points    = quadrature_formula.size ();

        std::vector<double> u0_values (n_q_points), u1_values (n_q_points), u2_values(n_q_points);

        auto cell = dof_handler_.begin_active ();
        auto endc = dof_handler_.end ();
        for (; cell != endc; cell++) {
            fe_values.reinit (cell);

            fe_values.get_function_values (u0_, u0_values);
            fe_values.get_function_values (u1_, u1_values);
            fe_values.get_function_values (u2_, u2_values);

            for (unsigned int q=0; q<n_q_points; q++) {
                v0 += u0_values[q] * fe_values.JxW (q);
                v1 += u1_values[q] * fe_values.JxW (q);
                v2 += u2_values[q] * fe_values.JxW (q);
            }
        }

        volumes[0] = v0;
        volumes[1] = v1;
        volumes[2] = v2;
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
        }

        AMG_preconditioner_.reset ();

        partitioning_.resize (1);
        partitioning_[0] = complete_index_set (dof_handler_.n_dofs());
        u0_.reinit (partitioning_[0], MPI_COMM_WORLD);
        u1_.reinit (partitioning_[0], MPI_COMM_WORLD);
        u2_.reinit (partitioning_[0], MPI_COMM_WORLD);
        system_rhs0_.reinit (partitioning_[0], MPI_COMM_WORLD);
        system_rhs1_.reinit (partitioning_[0], MPI_COMM_WORLD);
        system_rhs2_.reinit (partitioning_[0], MPI_COMM_WORLD);
    }

    template <int dim>
    void Simulator<dim>::refine_mesh() {
        IndexSet cell_partition = complete_index_set (triangulation_.n_active_cells ());

        const unsigned int max_grid_level = prm_->n_initial_global_refines + prm_->max_adaptation_level;
        const unsigned int min_grid_level = prm_->n_initial_global_refines;
        const unsigned int dofs_per_cell = fe_.dofs_per_cell;

        const QMidpoint<dim> quadrature_formula;
        const UpdateFlags update_flags = update_gradients;
        FEValues<dim> fe_values (fe_, quadrature_formula, update_flags);

        TrilinosWrappers::MPI::Vector criteria;
        criteria.reinit (cell_partition, MPI_COMM_WORLD);

        std::vector<Tensor<1,dim>> du0 (1, Tensor<1,dim>());
        std::vector<Tensor<1,dim>> du1 (1, Tensor<1,dim>());
        std::vector<Tensor<1,dim>> du2 (1, Tensor<1,dim>());

        // Compute criteria
        {
            auto cell = dof_handler_.begin_active ();
            auto endc = dof_handler_.end ();
            for (unsigned int cell_no = 0; cell != endc; cell++, cell_no++) {
                fe_values.reinit (cell);
                fe_values.get_function_gradients (u0_, du0);
                fe_values.get_function_gradients (u1_, du1);
                fe_values.get_function_gradients (u2_, du2);

                criteria(cell_no) = std::log(1 +
                                             std::sqrt (du0[0] * du0[0]
                                                        + du1[0] * du1[0]
                                                        + du2[0] * du2[0] ));
            }
        }

        // Mark cells to refine
        {
            auto cell = dof_handler_.begin_active ();
            auto endc = dof_handler_.end ();
            for (unsigned int cell_no = 0; cell != endc; cell++, cell_no++) {
                if (criteria(cell_no) > 1e-2) {
                    cell->set_refine_flag ();
                }
                else if (criteria(cell_no) < 1e-3) {
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

        // Solution transfer
        parallel::distributed::SolutionTransfer<dim, TrilinosWrappers::MPI::Vector> solution_trans0 (dof_handler_);
        parallel::distributed::SolutionTransfer<dim, TrilinosWrappers::MPI::Vector> solution_trans1 (dof_handler_);
        parallel::distributed::SolutionTransfer<dim, TrilinosWrappers::MPI::Vector> solution_trans2 (dof_handler_);
        triangulation_.prepare_coarsening_and_refinement ();
        solution_trans0.prepare_for_coarsening_and_refinement (u0_);
        solution_trans1.prepare_for_coarsening_and_refinement (u1_);
        solution_trans2.prepare_for_coarsening_and_refinement (u2_);

        triangulation_.execute_coarsening_and_refinement ();
        setup_system ();

        solution_trans0.interpolate (u0_);
        solution_trans1.interpolate (u1_);
        solution_trans2.interpolate (u2_);

        constraints_.distribute (u0_);
        constraints_.distribute (u1_);
        constraints_.distribute (u2_);
    }

    template <int dim>
    void Simulator<dim>::print_system_info () {

        LogStream::Prefix prefix ("SYSTEM_INFO");

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
    void Simulator<dim>::print_volume_info (const std::vector<double> & volumes) {
        AssertThrow (volumes.size ()==3, ExcMessage ("Invalid volume information!"))

        LogStream::Prefix prefix ("VOLUME_INFO");
        deallog << "Volume of phase 0 (solid)  = " << volumes[0] << ". " << std::endl;
        deallog << "Volume of phase 1 (gas)    = " << volumes[1] << ". " << std::endl;
        deallog << "Volume of phase 2 (liquid) = " << volumes[2] << ". " << std::endl;
    }

    /**
     * Simple threshold. Set the phase to be the one with larger value, and solid (0) is unchanged.
     */
    template <int dim>
    void Simulator<dim>::apply_threshold() {
        // Assuming Lagrange elements, where solution vector stores nodal values
        VectorTools::interpolate (dof_handler_, InitialValues0<dim> (), u0_);
        for (unsigned int i = 0; i < u0_.size (); i++) {
            if (u1_[i] < u2_[i]) {
                u1_[i] = 0.0;
                u2_[i] = 1.0 - u0_[i];
            }
            else {
                u1_[i] = 1.0 - u0_[i];
                u2_[i] = 0.0;
            }
        }
    }

    /**
     * Conservative threshold. Find a level set to preserve volumes of non-solid phases.
     */
    template <int dim>
    void Simulator<dim>::apply_conservative_threshold () {

    }

    template <int dim>
    void Simulator<dim>::assemble_system_matrix(const double ddt) {

        system_matrix_ = 0.0;
        const double theta = prm_->theta;

        const QGauss<dim> quadrature_formula (prm_->degree + 1);
        FEValues<dim> fe_values (fe_, quadrature_formula,
                                 update_values | update_gradients | update_JxW_values);
        const unsigned int dofs_per_cell = fe_.dofs_per_cell;
        const unsigned int n_q_points    = quadrature_formula.size ();

        FullMatrix<double> local_matrix (dofs_per_cell, dofs_per_cell);
        std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

        auto cell = dof_handler_.begin_active ();
        auto endc = dof_handler_.end ();
        for (; cell != endc; cell++) {
            fe_values.reinit (cell);
            local_matrix = 0.0;
            for (unsigned int q=0; q<n_q_points; q++) {

                for (unsigned int i=0; i<dofs_per_cell; i++) {
                    for (unsigned int j=0; j<dofs_per_cell; j++) {
                        local_matrix(i,j) += (fe_values.shape_value (i, q) * fe_values.shape_value (j, q) +
                                              ddt * theta * (fe_values.shape_grad (i, q) * fe_values.shape_grad (j, q))
                                             ) * fe_values.JxW (q);
                    }
                }

            }
            cell->get_dof_indices (local_dof_indices);
            constraints_.distribute_local_to_global (local_matrix, local_dof_indices, system_matrix_);
        }

        // AMG preconditioner
        AMG_preconditioner_ = std::make_shared<TrilinosWrappers::PreconditionAMG> ();
        TrilinosWrappers::PreconditionAMG::AdditionalData AMG_data;
        AMG_data.elliptic = true;
        if (prm_->degree>1) {
            AMG_data.higher_order_elements = true;
        }
        AMG_data.smoother_sweeps = prm_->smoother_sweeps;
        AMG_data.aggregation_threshold = prm_->aggregation_threshold;
        AMG_preconditioner_->initialize (system_matrix_, AMG_data);


    }

    template <int dim>
    void Simulator<dim>::assemble_system_rhs(const double ddt) {

        system_rhs0_ = 0.0;
        system_rhs1_ = 0.0;
        system_rhs2_ = 0.0;

        const double theta = prm_->theta;

        const QGauss<dim> quadrature_formula (prm_->degree + 1);
        FEValues<dim> fe_values (fe_, quadrature_formula,
                                 update_values | update_gradients | update_JxW_values);
        const unsigned int dofs_per_cell = fe_.dofs_per_cell;
        const unsigned int n_q_points    = quadrature_formula.size ();

        Vector<double> local_rhs0 (dofs_per_cell);
        Vector<double> local_rhs1 (dofs_per_cell);
        Vector<double> local_rhs2 (dofs_per_cell);

        std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

        std::vector<double> u0_values (n_q_points), u1_values (n_q_points), u2_values(n_q_points);
        std::vector<Tensor<1,dim>> grad_u0_values (n_q_points), grad_u1_values (n_q_points), grad_u2_values(n_q_points);

        auto cell = dof_handler_.begin_active ();
        auto endc = dof_handler_.end ();
        for (; cell != endc; cell++) {
            fe_values.reinit (cell);
            local_rhs0 = 0.0;
            local_rhs1 = 0.0;
            local_rhs2 = 0.0;

            fe_values.get_function_values (u0_, u0_values);
            fe_values.get_function_values (u1_, u1_values);
            fe_values.get_function_values (u2_, u2_values);
            fe_values.get_function_gradients (u0_, grad_u0_values);
            fe_values.get_function_gradients (u1_, grad_u1_values);
            fe_values.get_function_gradients (u2_, grad_u2_values);

            for (unsigned int q=0; q<n_q_points; q++) {

                for (unsigned int i=0; i<dofs_per_cell; i++) {
                    local_rhs0(i) += (fe_values.shape_value (i, q) * u0_values[q] -
                                      ddt * (1.0 - theta) * (fe_values.shape_grad (i, q) * grad_u0_values[q])
                                     ) * fe_values.JxW (q);
                    local_rhs1(i) += (fe_values.shape_value (i, q) * u1_values[q] -
                                      ddt * (1.0 - theta) * (fe_values.shape_grad (i, q) * grad_u1_values[q])
                                     ) * fe_values.JxW (q);
                    local_rhs2(i) += (fe_values.shape_value (i, q) * u2_values[q] -
                                      ddt * (1.0 - theta) * (fe_values.shape_grad (i, q) * grad_u2_values[q])
                                     ) * fe_values.JxW (q);
                }

            }
            cell->get_dof_indices (local_dof_indices);
            constraints_.distribute_local_to_global (local_rhs0, local_dof_indices, system_rhs0_);
            constraints_.distribute_local_to_global (local_rhs1, local_dof_indices, system_rhs1_);
            constraints_.distribute_local_to_global (local_rhs2, local_dof_indices, system_rhs2_);
        }

    }

    template <int dim>
    void Simulator<dim>::solve_system() {

        LogStream::Prefix prefix ("SOLVE");

        SolverControl solver_control0 (system_matrix_.m (),
                                       prm_->eps * system_rhs0_.l2_norm ());
        SolverControl solver_control1 (system_matrix_.m (),
                                       prm_->eps * system_rhs1_.l2_norm ());
        SolverControl solver_control2 (system_matrix_.m (),
                                       prm_->eps * system_rhs2_.l2_norm ());

        SolverCG<TrilinosWrappers::MPI::Vector> cg0 (solver_control0,
                                                     SolverCG<TrilinosWrappers::MPI::Vector>::AdditionalData());
        SolverCG<TrilinosWrappers::MPI::Vector> cg1 (solver_control1,
                                                     SolverCG<TrilinosWrappers::MPI::Vector>::AdditionalData());
        SolverCG<TrilinosWrappers::MPI::Vector> cg2 (solver_control2,
                                                     SolverCG<TrilinosWrappers::MPI::Vector>::AdditionalData());

        cg0.solve (system_matrix_, u0_, system_rhs0_, *AMG_preconditioner_);
        cg1.solve (system_matrix_, u1_, system_rhs1_, *AMG_preconditioner_);
        cg2.solve (system_matrix_, u2_, system_rhs2_, *AMG_preconditioner_);

        constraints_.distribute (u0_);
        constraints_.distribute (u1_);
        constraints_.distribute (u2_);

        deallog << "Number of CG iterations:" << std::endl;
        deallog << "  u0: " << solver_control0.last_step () << std::endl;
        deallog << "  u1: " << solver_control1.last_step () << std::endl;
        deallog << "  u2: " << solver_control2.last_step () << std::endl;

    }

    template <int dim>
    void Simulator<dim>::output_solution () {
        DataOut<dim> data_out;
        data_out.attach_dof_handler (dof_handler_);
        data_out.add_data_vector (u0_, "U_0");
        data_out.add_data_vector (u1_, "U_1");
        data_out.add_data_vector (u2_, "U_2");

        data_out.build_patches ();

        const std::string filename = ("output/solution-" +
                                      Utilities::int_to_string (timestep_number_, 4) );
        std::ofstream output ((filename + ".vtu").c_str());
        data_out.write_vtu (output);
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

            const double dt = prm_->dt;
            const double ddt = prm_->dt / prm_->n_sub_steps;
            const double theta = prm_->theta;

            initialize_simulator ();
            compute_volumes (volume_);
            assemble_system_matrix (ddt);

            print_system_info ();
            print_volume_info (volume_);
            output_solution ();

            double time = 0;
            while (time < prm_->final_time) {
                time += dt;
                timestep_number_++;
                deallog << "Time step " << timestep_number_ << " on t = " << time << std::endl;

                for (unsigned substep_number = 0;
                     substep_number<prm_->n_sub_steps;
                     substep_number++) {

                    assemble_system_rhs (ddt);
                    solve_system ();
                }

                apply_threshold ();

                if (timestep_number_%prm_->adaptation_interval==0) {
                    LogStream::Prefix adapt_prefix ("ADAPT");
                    for (unsigned int s=0; s<prm_->n_adaptation_sweeps; s++) {
                        refine_mesh ();
                    }
                    assemble_system_matrix (ddt);
                    deallog << "Mesh updated." << std::endl;
                    print_system_info ();
                }


                if (timestep_number_%prm_->output_interval==0) {
                    output_solution ();
                }

            }
        }

    }

}



#endif //MBOX_SIMULATOR_H
