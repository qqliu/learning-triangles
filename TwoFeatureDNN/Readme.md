Our experiments on training with only using the triangle counts of the prefix uses the same network architecture as our BucketDNN. We include our Oregon-2 dataset under datasets/oregon2/

Run the following command to see our network trained on using the triangle counts as features:

    python3 ../PrefixBucket/bucketdnn.py 2 1 2 16 42 datasets/oregon2/train/oregon2_010331_dedup 9 1e-1 datasets/oregon2/oregon2_010407_dedup

Run the following command to test on the test sets:

    python3 predict-from-saved.py 232 1 2 256 bucket_data/caida/2006/caida1_dedup_train.pt bucket_data/caida/2006/ 1 1e-3 1.1 count_data/oregon_test_2 1e-1 9 1000 results/dnn_caida_2006_new

The results are saved in the results/ directory. The format of the results is:

file, mean error, stdev error, max error, min error, 0.01 fraction error, 0.02 fraction error,...etc.
