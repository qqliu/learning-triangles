import numpy as np
import sys
import sklearn.pipeline
import os

from sklearn.linear_model import LinearRegression, Ridge, SGDRegressor, Lasso

input_train_file = open(str(sys.argv[1]), 'r')
directory = str(sys.argv[2])

train_lines = input_train_file.readlines()

X = []
Y = []
for i in range(1, len(train_lines)):
    line = train_lines[i]
    values = line.split()
    inp = []
    for i in range(0, len(values) - 1):
        inp.append(float(values[i]))
    X.append(inp)
    Y.append(float(values[-1]))

X = np.array(X)
Y = np.array(Y)

if not len(X) == len(Y):
    print("ERROR: train vectors not equal: ", len(X), len(Y))

sgd_reg = sklearn.pipeline.make_pipeline(sklearn.preprocessing.StandardScaler(), SGDRegressor(max_iter=5000, tol=float(sys.argv[3])))
sgd_reg.fit(X, Y)
ridge_reg = Ridge(alpha = float(sys.argv[4])).fit(X, Y)
linear_reg = LinearRegression().fit(X, Y)
lasso_reg = Lasso(alpha = float(sys.argv[5]), max_iter = 100000)
lasso_reg.fit(X, Y)

sgd_score = sgd_reg.score(X, Y)
linear_score = linear_reg.score(X, Y)
lasso_score = lasso_reg.score(X, Y)
ridge_score = ridge_reg.score(X, Y)
#coeff = reg.coef_
#intercept = reg.intercept_
#print("Fit score: " , linear_score, ridge_score, sgd_score, lasso_score)
#print(X, Y)
#, "Coefficient: " , coeff, "Intercept: ", intercept)
for filename in os.listdir(directory):
    if "_dedup" in filename and not ".pt" in filename:
        f = os.path.join(directory, filename)
        m = {}

        test_file = open(f, 'r')
        test_lines = test_file.readlines()

        X_predict = []
        Y_real = []
        for i in range(1, len(test_lines)):
            line = test_lines[i]
            values = line.split()
            inp = []
            for i in range(0, len(values) - 1):
                inp.append(float(values[i]))
            X_predict.append(inp)
            Y_real.append(float(values[-1]))

        X_predict = np.array(X_predict)
        Y_real = np.array(Y_real)

        Y_sgd_predict = sgd_reg.predict(X_predict)
        Y_ridge_predict = ridge_reg.predict(X_predict)
        Y_linear_predict = linear_reg.predict(X_predict)
        Y_lasso_predict = lasso_reg.predict(X_predict)
        #print(Y_lasso_predict, Y_real)

        if not len(X_predict) == len(Y_real) or not len(Y_real) == len(Y_sgd_predict):
            print("ERROR: test output lengths not equal: ", len(X_predict), len(Y_real), len(Y_sgd_predict))

        sgd_errors = []
        ridge_errors = []
        linear_errors = []
        lasso_errors = []
        for i in range(len(Y_sgd_predict)):
            if not Y_real[i] == 0:
                sgd_error = abs(1 - (1.0 * Y_sgd_predict[i])/Y_real[i])
                sgd_errors.append(sgd_error)
                ridge_error = abs(1 - (1.0 * Y_ridge_predict[i])/Y_real[i])
                ridge_errors.append(ridge_error)
                linear_error = abs(1 - (1.0 * Y_linear_predict[i])/Y_real[i])
                linear_errors.append(linear_error)
                lasso_error = abs(1 - (1.0 * Y_lasso_predict[i])/Y_real[i])
                lasso_errors.append(lasso_error)

        #if not len(linear_errors) == 30:
        #    print("ERROR LENGTH OF ERRORS WRONG", filename)
        errors = linear_errors
        errs = ", ".join([str(x) for x in errors])
        print(filename + ", " + str(np.mean(errors)) + ", " + str(np.std(errors)) + ", " + str(max(errors)) + ", " + str(min(errors)) + ", " + errs)

        errors = ridge_errors
        errs = ", ".join([str(x) for x in errors])
        print(filename + ", " + str(np.mean(errors)) + ", " + str(np.std(errors)) + ", " + str(max(errors)) + ", " + str(min(errors)) + ", " + errs)

        errors = sgd_errors
        errs = ", ".join([str(x) for x in errors])
        print(filename + ", " + str(np.mean(errors)) + ", " + str(np.std(errors)) + ", " + str(max(errors)) + ", " + str(min(errors)) + ", " + errs)

        errors = lasso_errors
        errs = ", ".join([str(x) for x in errors])
        print(filename + ", " + str(np.mean(errors)) + ", " + str(np.std(errors)) + ", " + str(max(errors)) + ", " + str(min(errors)) + ", " + errs)
