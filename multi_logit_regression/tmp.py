from math import exp

sepal_length = 5
setosa_v = 14.771200 + -2.485960 * sepal_length
versicolor_v = -1.852440 + 0.588428 * sepal_length
virginica_v = -13.075700 + 2.402020 *sepal_length

numerator = exp(versicolor_v) + exp(virginica_v) + exp(setosa_v)
setosa_p = exp(setosa_v)/numerator
versicolor_p = exp(versicolor_v)/numerator
virginica_p = exp(virginica_v)/numerator
#output = "generating parameters are: setosa_sepal_mean={:.1f}, beta_truth={:.1f}, n={:d}"
#print(output.format(alpha_truth, beta_truth, n))


output = "setosa={:.2f}, versicolor_p={:.2f}, virginica_p={:.2f}"
print(output.format(setosa_p, versicolor_p, virginica_p))
