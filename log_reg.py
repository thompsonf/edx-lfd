import random
from itertools import permutations
from math import exp, sqrt, log

#given two points, return slope and intercept of line connecting them
def get_slope_int(p1, p2):
	m = (p2[1] - p1[1]) / (p2[0] - p1[0])
	b = p1[1] - m * p1[0]
	return m, b

#given line with slope m and y-intercept b,
#return +1 if p is above the line, -1 otherwise
def classify(m, b, p):
	# if p lies below (or on) mx + b
	if m*p[0] + b >= p[1]:
		return -1
	else:
		return 1

#get random points in [-1,1]x[-1,1]
def get_rand_pts(num_pts):
	return [[random.uniform(-1, 1), random.uniform(-1, 1)] for i in range(num_pts)]

#get random points in [-1,1]x[-1,1] as len 3 vectors with 1 as first entry
def get_rand_pts_with_const(num_pts):
	return [[1, random.uniform(-1, 1), random.uniform(-1, 1)] for i in range(num_pts)]

#find two random points in [-1,1]x[-1,1]
#return slope and intercept of line connecting them
def get_f():
	p1, p2 = get_rand_pts(2)
	return get_slope_int(p1, p2)

#return u dot v
def dot_prod(u, v):
	dp_sum = 0
	for i in range(len(u)):
		dp_sum += u[i] * v[i]
	return dp_sum

#return a * v
def scal_mult(a, v):
	return [a * vi for vi in v]

#return u + v
def vec_sum(u, v):
	return [u[i] + v[i] for i in range(len(u))]

#return u - v
def vec_diff(u, v):
	return [u[i] - v[i] for i in range(len(u))]

#return magnitude of vector
def magnitude(v):
	return sqrt(sum([vi ** 2 for vi in v]))

#given weight vector w, calculate cross-entropy error of w on data x, y
def calc_error(w, x, y):
	N = len(x)
	err = 0
	for n in range(N):
		err += log(1 + exp(-y[n] * dot_prod(w, x[n])))
	return err / N

#x is a vector of 2d points with 1 as the first entry!
#y is a vector of +/- 1 depending on how the targt ftn classified each point
#eta is SGD update factor
#delta_thresh is fraction that w must change by in order to continue SGD
#run logistic regression and return weight vector and num iters to converge
def log_reg(x, y, eta, delta_thresh):
	#merge x and y into a single list so that it is easy to shuffle
	data = [(x[i], y[i]) for i in range(len(y))]

	num_iters = 0
	#initialize w to a vector of all zeroes
	prev_w = [0, 0, 0]
	w = prev_w[:]
	while True:
		w = prev_w[:]
		#permute the data
		random.shuffle(data)
		#perform stochastic grad descent
		for x, y in data:
			#calculate grad of e(h(x_n), y_n)
			alpha = -y / (1 + exp(y * dot_prod(w, x)))
			grad_e = scal_mult(alpha, x)
			#calculate amount to change w by
			delta_w = scal_mult(-eta, grad_e)
			#update w
			w = vec_sum(w, delta_w)
		num_iters += 1
		#check if total change to w is bigger than delta_thresh
		delta_magnitude = magnitude(vec_diff(w, prev_w))
		if delta_magnitude < delta_thresh:
			break
		#if not, update prev_w and start over
		prev_w = w[:]
	return w, num_iters

#run logistic regression many times
#return avg E_out and avg num iterations to converge
def run_tests(N, num_runs, eta, delta_thresh):
	E_out_sum = 0
	iters_sum = 0
	for i in range(num_runs):
		#print run number to see progress
		print(i)
		#get new target function
		m, b = get_f()
		#get new random data points and classify them
		x = get_rand_pts_with_const(N)
		y = [classify(m, b, p) for p in x]
		#run logistic regression
		g, num_iters = log_reg(x, y, eta, delta_thresh)

		#generate new data set to estimate E_out
		x_out = get_rand_pts_with_const(10 * N)
		y_out = [classify(m, b, p) for p in x_out]
		E_out = calc_error(g, x_out, y_out)

		E_out_sum += E_out
		iters_sum += num_iters

	return E_out_sum / num_runs, iters_sum / num_runs

print("Avg E_out: %lf\nAvg iters: %lf" % run_tests(100, 100, .01, .01))