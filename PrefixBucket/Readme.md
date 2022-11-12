All of our BucketDNN code. Running the training code requires a GPU. One can change the code to run on a computer with only CPUs by removing the .cuda() commands where appropriate.

We include sample bucket files in caida2004/

You can run our code via the following command:

    python3 bucketdnn.py 232 1 2 256 9 caida2004/caida1_dedup_train 1 1e-3 caida2004/caida2_dedup_test

To run the prediction code for the rest of the datasets, use the following command:
    python3 predict-from-saved.py 232 1 2 256 caida2004/caida1_dedup_train.pt caida2004 1 1e-3 1.1 count_data/oregon_test_2 1e-1 9 1000 results/caida2004

