// fireredsim-worker: see worker_loop.h (the loop is shared with fireredsim-bench).
#include "worker_loop.h"

int main(int argc, char **argv)
{
    return pn::worker_main(argc, argv);
}
