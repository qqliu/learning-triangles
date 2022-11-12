This directory contains our LinearRegression code. Run the following command to run our regression models.

    python3 linear-regression.py ../PrefixBucket/caida2004/caida1_dedup_train ../PrefixBucket/caida2004/ 1e-3 1000 10000

This runs our linear regression code on the included example caida2004/ dataset. The outputs are for all space fraction parameters. The output is of the following format:

file, mean error of all space fractions, stdev of all space fractions, max error, min error, 0.01 fraction error, 0.02 fraction error, 0.03 fraction error...etc.

