/**
 * An implementation of MBO scheme, using a finite element solver
 * based on Deal.II for heat diffusion.
 *
 * Three phase parameters:
 *      - [ 0 ]: the solid
 *      - [ 1 ]: the gas (outside bubble)
 *      - [ 2 ]: the liquid (inside bubble)
 *
 * By: Jacob Xiaoyu Wei (wxy0516@gmail.com)
 * May 16, 2016
 */

// define INIT_0 to run case "drop in the air"
// define INIT_1 to run case "drop on flat surface"
#undef INIT_0
#undef INIT_1
#define INIT_2

#include "include/simulator.h"

int main (int argc, char *argv[])
{
    try {

        using namespace mbox;

        // We have to initialize and finalize MPI to use Trilinos, which
        // is done using the helper object MPI_InitFinalize.
        Utilities::MPI::MPI_InitFinalize mpi_initFinalization (argc, argv,
                /* max_num_threads = */ numbers::invalid_unsigned_int);

        AssertThrow (Utilities::MPI::n_mpi_processes (MPI_COMM_WORLD)==1,
                     ExcMessage("This program can only be run in serial."));

        /**
         * Handling input
         */
        std::string parameter_filename ("input/mbox.prm");
        std::string prm_xml_filename ("output/parameters.xml");
        std::shared_ptr<RuntimeParameters> param =
                std::make_shared<RuntimeParameters> (RuntimeParameters ());

        if (exists_test(prm_xml_filename)) {
            std::ifstream prm_filein(prm_xml_filename);
            param->prm->read_input_from_xml(prm_filein);
            param->parse_parameters ();
            deallog << "Reusing the parameter file in output folder." << std::endl;
        }
        else {
            // If the input file does not exit, dealii will create one with default values
            // (which is nice)
            param->prm->read_input(parameter_filename);
            param->parse_parameters ();
            std::ofstream prm_fileout(prm_xml_filename);
            param->prm->print_parameters(prm_fileout, ParameterHandler::XML);
            deallog << "Using input parameters, and generated an xml file in output folder" << std::endl;
        }

        /**
         * Calling the simulator
         */
        deallog.depth_console (param->verbose ? 99 : 0);
        switch (param->dim) {
            case 1: {
                deallog << "Cannot run on 1-dimensional domain due to MPI compatibility." << std::endl;
                break;
            }
            case 2: {
                deallog << "Running on 2-dimensional domain." << std::endl;
                Simulator<2> simulator (param);
                deallog << "Simulator initialized." << std::endl;
                simulator.run ();
                break;
            }
            case 3: {
                deallog << "Running on 3-dimensional domain." << std::endl;
                Simulator<3> simulator (param);
                deallog << "Simulator initialized." << std::endl;
                simulator.run ();
                break;
            }
            default: {
                AssertThrow (false, ExcNotImplemented ());
            }
        }
    }

    catch (std::exception &exc) {
        std::cerr << std::endl << std::endl
        << "----------------------------------------------------"
        << std::endl;
        std::cerr << "Exception on processing: " << std::endl
        << exc.what() << std::endl
        << "Aborting!" << std::endl
        << "----------------------------------------------------"
        << std::endl;
        return 1;
    }

    catch (...) {
        std::cerr << std::endl << std::endl
        << "----------------------------------------------------"
        << std::endl;
        std::cerr << "Unknown exception!" << std::endl
        << "Aborting!" << std::endl
        << "----------------------------------------------------"
        << std::endl;
        return 1;
    }

    std::cout << "----------------------------------------------------"
    << std::endl
    << "Apparently everything went fine!"
    << std::endl
    << "Don't forget to brush your teeth :-)"
    << std::endl << std::endl;
    return 0;
}
