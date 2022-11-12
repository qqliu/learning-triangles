This is the directory for our Random Order Streaming Algorithm code detailed in
Section 3 of our paper.

To make our code efficient for running experiments on large graphs, we use the
Graph Based Benchmark Suite (GBBS) [https://github.com/ParAlg/gbbs] to allow for
parallelization of experimental code. To run the experimental code in this
folder, please use the following setup code and scripts:

1. Setup.txt: this is the document for setting up the experiments for running
   our code. Specify the following parameters:
    a) Percent: the fraction of the spaces tested in the experiments
    b) Trials: the number of times each experiment is run
    c) Input graph directory: the directory to obtain the input graphs
    d) Dynamic graphs: the names of the files within the input graph directory
    e) Output directory: the directory to print the output files
2. MV_setup.txt: All parameters are the same as the Setup.txt parameters except:
    a) Eps: the epsilons you want to test where epsilon is the approximation parameter.

To run our code, simply use the command:
    python3 test_random_stream.py

To run our MV20 code, simply use the command:
    python3 test_MV_random_stream.py

We include example test graphs in example_temporal_graphs/
We included the two graphs: superuser and wiki.

To read the outputs, simply run:
    python3 read.py
    python3 read_MV.py

The outputs are comma-separated, in the following order:
    graph_file, space_fraction, mean_error, stdev_error
