"""
      --------------------------------------------------------------------
                              Echo State Networks
      --------------------------------------------------------------------

Implemented following the specifications in:

[1] http://www.scholarpedia.org/article/Echo_state_network

[2] Lukosevicius, M. (2012). A practical guide to applying echo state networks.
    In Neural networks: Tricks of the trade (pp. 659-686). Springer Berlin Heidelberg.
    http://minds.jacobs-university.de/sites/default/files/uploads/papers/PracticalESN.pdf


Luis F. Simoes
2016-07-29

"""


import numpy as np

from scipy.linalg import solve
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.solve.html

from tqdm import tqdm, trange
import sys



##### --------------------------------------------------------------------

class ESN(object):
	
	# Supported activation functions, for application over reservoir neurons
	_activation_func = {
		'sigmoid' :
			lambda x : 1.0 / (1.0 + np.exp(-x)),		# output in: ( 0.0, 1.0)
		
		'hyp.tan' :
			np.tanh,				# hyperbolic tangent, output in: (-1.0, 1.0)
		'LeCun tanh' :
			lambda x : 1.7159 * np.tanh(2./3. * x),
		
		# https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
		'rectifier' :
			lambda x : np.maximum(0., x),
		'Leaky ReLU' :
			lambda x : np.maximum(0.01 * x, x),
		'softplus' :
			lambda x : np.log(1. + np.exp(x)),
		
		# http://hdl.handle.net/1903/5355
		'Elliott' :
			lambda x : x / (1. + np.abs(x)),
		
		# No activation function / identity
		'identity' :
			lambda x : x,
		}
	
	def __init__(self, nr_neurons=100, prob_connect=None, spectral_radius=0.9,
		         activation='hyp.tan', leaking_rate=1,
		         output_feedback=False, y_noise=0.0,
		         alpha=1e-9, batch_size=5000, random_state=None):
		
		assert 0 < prob_connect <= 1, "`prob_connect` should be in (0,1]"
		assert 0 <= leaking_rate <= 1, "`leaking_rate` should be in [0,1]"
		
		# configure the reservoir
		self.nr_neurons = nr_neurons
		self.prob_connect = prob_connect
		self._spectral_radius = spectral_radius
		self.activation_function = self._activation_func[activation]
		self.leaking_rate = leaking_rate
		
		# dealing with output feedbacks
		self.output_feedback = output_feedback
		self.y_noise = y_noise
		
		# parameters of the readout's training
		# (weights connecting reservoir to outputs)
		self.alpha = alpha
		self.batch_size = batch_size
		
		self.random = np.random if random_state is None else random_state
		
		self.W, self.B = None, None
		self.r = None
		
	
	def __str__(self):
		if self.W is None:
			return '?-%d-? ESN (untrained)' % self.nr_neurons
		return '%d-%d-%d ESN' % self.shape()
		
	
	def shape(self):
		"Get the number of dimensions per layer (input, reservoir, readout)."
		return (self.W.shape[0] - 1,
				self.nr_neurons,
				self.B.shape[1] if self.B.ndim > 1 else 1)
		
	
	def reservoir_rec_weights(self):
		"Get the weights for the reservoir nodes' recurrent connections"
		return self.W[-self.nr_neurons:, :]
		
	
	def spectral_radius(self):
		"Determine the spectral radius of the reservoir connection matrix"
		rw = self.reservoir_rec_weights()
		eigvals = np.linalg.eigvals(rw)
		spec_rad = np.abs(eigvals).max()
		return spec_rad
		# https://en.wikipedia.org/wiki/Spectral_radius
		# http://mathworld.wolfram.com/SpectralRadius.html
		
	
	def fit(self, X, y, sample_weight=None, y_initial=None):
		
		nr_samples, input_dim = X.shape
		assert nr_samples > self.nr_neurons, "Implemented training equations " \
			"expect nr_samples (%d) > nr_neurons (%d)" % (nr_samples, self.nr_neurons)
		
		# ---------- Define the reservoir
		#
		# configure the weights of connections into reservoir neurons
		# (these consist of: bias input, external inputs, feedback from
		# the output (optionally), and recurrent reservoir connections).
		# Initialized to random weights (which then remain constant throughout training)
		self.output_dim = y.shape[1]
		output_feedback_dim = self.output_dim if self.output_feedback else 0
		nr_inputs = 1 + input_dim + output_feedback_dim + self.nr_neurons
		
		self.W = self.random.standard_normal(size=(nr_inputs, self.nr_neurons))
		
		rw = self.reservoir_rec_weights()
		
		# enforce connectivity's sparsity.
		# prob_connect (probability of a link existing) ranges in [0,1],
		# going from a fully disconnected to a fully connected graph.
		if self.prob_connect is not None:
			i = self.random.rand(*rw.shape) > self.prob_connect
			rw[i] *= 0.0
		
		# enforce the requested spectral radius
		# (should be, in most situations, < 1 to ensure the echo state property)
		if self._spectral_radius is not None:
			rw *= (self._spectral_radius / self.spectral_radius())
		
		# ---------- Train readouts
		#
		# split the training set into multiple batches
		nr_batches = np.ceil(nr_samples / float(self.batch_size))
		b = np.linspace(0, nr_samples, nr_batches + 1)[1:-1].astype(np.int)
		
		if sample_weight is None:
			batches = list(zip(np.split(X, b), np.split(y, b)))
		else:
			# "An additional multiplication by A is avoided by applying
			# weights $\sqrt{a_j}$ directly to the rows of matrices H, T"
			# -- http://dx.doi.org/10.1109/ACCESS.2015.2450498
			w = np.sqrt(sample_weight).reshape(-1, 1)
			assert w.shape[0] == nr_samples, "%d weights provided for %d " \
				"training samples" % (w.shape[0], nr_samples)
			batches = list(zip(np.split(X, b), np.split(y, b), np.split(w, b)))
		
		# process batches, aggregating results into the HH and Hy matrices
		HH, Hy, r, y_initial = self._fit_step(*batches[0], r_initial=None, y_initial=y_initial)
		for Xyw in batches[1:]:
			HHb, Hyb, r, y_initial = self._fit_step(*Xyw, r_initial=r, y_initial=y_initial)
			HH += HHb
			Hy += Hyb
		
		# add the regularization term to HH's diagonal
		# See: https://en.wikipedia.org/wiki/Tikhonov_regularization
		HH.ravel()[::HH.shape[1]+1] += self.alpha
		
		# calculate the output weights matrix, beta
		# (finds the regularized least squares fit to a [multivariate] linear regression problem)
		#self.B = np.linalg.lstsq(HH, Hy)[0]
		self.B = solve(HH, Hy, sym_pos=True, overwrite_a=True, overwrite_b=True)
		# http://mathworld.wolfram.com/PositiveDefiniteMatrix.html
		
	
	def _fit_step(self, X, y, sample_weight=None, y_initial=None, **kwargs):
		
		if self.output_feedback:
			if y_initial is None:
				y_initial = np.zeros(self.output_dim)
			
			# add Gaussian noise to the outputs sent as feedback
			if self.y_noise > 0.0:
				_y = y * self.random.normal(1, self.y_noise, size=y.shape)
			else:
				_y = y
			
			# "Teacher forcing": disengages the recurrent relationship between the reservoir
			# and the readout during training. Treats output learning as a feedforward task.
			# Feeds the desired outputs y (optionally with added noise) through the feedback
			# connections, as if they had been the model's outputs at the previous step.
			_y = np.vstack([y_initial, _y[:-1]])
			X = np.hstack([X, _y])
		
		# calculate the reservoir neurons' activations matrix, given the input vectors in X
		H = self.propagate_reservoir(X, **kwargs)
		# get the reservoir's most recent state
		r = H[-1]
		# extend reservoir with the input vectors
		H = np.hstack([X, H])
		
		# calculate the auxiliary matrices from which beta will be determined
		if sample_weight is None:
			HH = H.T.dot(H)
			Hy = H.T.dot(y)
		else:
			Hw = sample_weight * H
			HH = Hw.T.dot(Hw)
			Hy = Hw.T.dot(sample_weight * y)
		
		return HH, Hy,  r, y[-1]
		
	
	def update_reservoir(self, X, r):
		"""
		Advance the reservoir's state by one time step, as a function of
		the new inputs `X`, extended with the reservoir's current state `r`.
		"""
		I = np.hstack([X, r])
		A = self.W[0] + np.dot(I, self.W[1:])
		A = self.activation_function(A)
		
		if self.leaking_rate == 1:
			# leaky integration not being used
			return A
		else:
			return (1. - self.leaking_rate) * r + self.leaking_rate * A
		
	
	def propagate_reservoir(self, X, reset_r=True, r_initial=None):
		"Advance the reservoir's state across multiple time steps."
		if reset_r:
			self.r = np.zeros(self.nr_neurons) if r_initial is None else r_initial
		
		R = []
		
		for x in X:
			self.r = self.update_reservoir(x, self.r)
			R.append(self.r)
		
		return np.array(R)
		
	
	def predict_with_feedback(self, X, reset_r=True, r_initial=None, y_initial=None, **kwargs):
		# a variant of `propagate_reservoir` that predicts outputs (`compute_readout`)
		# and feeds them back as inputs for the next step.
		
		if reset_r:
			self.r = np.zeros(self.nr_neurons) if r_initial is None else r_initial
		y = np.zeros(self.output_dim) if y_initial is None else y_initial
		
		Y = []
		for x in X:
			x = np.hstack([x, y])
			self.r = self.update_reservoir(x, self.r)
			y = self.compute_readout(x, self.r)
			Y.append(y)
		
		return np.array(Y)
		
	
	def compute_readout(self, X, R):
		"""
		Compute the linear readout, from an extended system state
		containing the inputs `X` and reservoir state `R`.
		"""
		Z = np.hstack([X, R])
		return np.dot(Z, self.B)
		
	
	def predict(self, X, **kwargs):
		assert self.B is not None, \
			'Attempt to evaluate an input with an untrained ESN.'
		
		if self.output_feedback:
			return self.predict_with_feedback(X, **kwargs)
		else:
			R = self.propagate_reservoir(X, **kwargs)
			return self.compute_readout(X, R)
		
	


##### --------------------------------------------------------------------

class ESN_ensemble(object):
	
	def __init__(self, aggregate='mean', grad_boost=False, *model_args, **model_kwargs):
		self.model_args = model_args
		self.model_kwargs = model_kwargs
		
		self.aggregate = {'mean' : np.mean, 'median' : np.median}[aggregate]
		self.grad_boost = grad_boost
		if self.grad_boost:
			self.aggregate = None
		
	
	def fit(self, X, y, nr_models=10, weighted=False, **train_args):
		self.M = []
		pred = None
		_pred = []
		_y = y
		w = None
		
		for i in trange(nr_models, leave=False, file=sys.stdout):
			
			if weighted and pred is not None:
				# weigh each time instants' predictions proportionally
				# to the ensemble's current RMSE on it
				w = np.mean((y - pred)**2, axis=1) ** 0.5
			
			m = ESN(*self.model_args, **self.model_kwargs)
			m.fit(X, _y, sample_weight=w, **train_args)
			
			if self.grad_boost:
				predX = m.predict(X)
				pred = (0 if pred is None else pred) + predX
				_y = y - pred
			elif weighted:
				predX = m.predict(X)
				_pred.append(predX)
				pred = self.aggregate(_pred, axis=0)
			
			self.M.append(m)
		
	
	def predict_with_feedback(self, X, reset_r=True, y_initial=None, **kwargs):
		assert not self.grad_boost, \
			"Using Gradient Boosting. Models can't all receive the same y."
		assert self.M[0].output_feedback, \
			"Trained model doesn't use y(t-1) as input."
		
		Y = []
		
		for x in X:
			_y = [m.predict(x[None], reset_r=reset_r, y_initial=y_initial, **kwargs)
				  for m in self.M]
			y = y_initial = self.aggregate(_y, axis=0)[0]
			Y.append(y)
			
			# after the first step, ensure `r` (reservoir neurons' states)
			# are no longer reset
			reset_r = False
		
		return np.array(Y)
		
	
	def predict(self, X, feedback_ensemble_y=False, **kwargs):
		if feedback_ensemble_y:
			return self.predict_with_feedback(X, **kwargs)
		
		predX = [m.predict(X, **kwargs) for m in self.M]
		if self.grad_boost:
			return np.sum(predX, axis=0)
		else:
			return self.aggregate(predX, axis=0)
		
	
	def model_selection(self, X, y, keep_top=10, **kwargs):
		"""
		Keep only the best `keep_top` models, as determined by the RMSEs
		on some dataset `X -> y`.
		"""
		m_eval = []
		for m in self.M:
			y_pred = m.predict(X, **kwargs)
			if not np.isnan(y_pred.max()):
				m_rmse = RMSE(y_true=y, y_pred=y_pred)
				m_eval.append((m_rmse, m))
		
		# sort by ascending order of RMSE
		m_eval.sort(key=lambda i:i[0])
		
		evals, models = list(zip(*m_eval))
		
		# keep only the `keep_top` models with lowest RMSE
		self.M = models[:keep_top]
		
		return evals, models
		
	
	def error(self, X, y, feedback_ensemble_y=False, **kwargs):
		"""
		Calculate how an ensemble of ESNs has its
		RMSE evolve with each additional model.
		"""
		if not feedback_ensemble_y:
			err = []
			pred = None
			_pred = []
			
			for m in tqdm(self.M, leave=False, file=sys.stdout):
				
				pred_m = m.predict(X, **kwargs)
				
				if self.grad_boost:
					pred = (0 if pred is None else pred) + pred_m
				else:
					_pred.append(pred_m)
					pred = self.aggregate(_pred, axis=0)
				
				e = RMSE(y_true=y, y_pred=pred)
				err.append(e)
			
			return err
		
		# if using aggregated y feedback, the contributions from each model
		# are evaluated differently
		else:
			y_initial = kwargs.pop('y_initial', np.zeros(y.shape[1]))
			y_shifted = np.vstack([y_initial, y[:-1]])
			M_pred = []
			for m in self.M:
				# obtain model m's predictions over time, as if at every stage
				# it had perfect feedback from the previous time step's output
				m_pred = np.array([
					m.predict(_x[None], reset_r=i == 0, y_initial=_y, **kwargs)
					for i,(_x, _y) in enumerate(zip(X, y_shifted))
					])
				M_pred.append(np.vstack(m_pred))
			
			return [
				RMSE(y_true=y, y_pred=self.aggregate(M_pred[:i], axis=0))
				for i in range(1, len(self.M) + 1)
				]
		
	
	def show_error(self, error):
		import matplotlib.pylab as plt
		l = 'Error after adding each model\n(final RMSE: %f)' % error[-1]
		plt.plot(list(range(1, len(self.M) + 1)), error, label=l)
		plt.legend()
		plt.xlabel('Number of models')
		plt.ylabel('RMSE')
		plt.xlim(0, len(error))
		model_str = '%d x %s' % (len(self.M), str(self.M[0]))
		plt.title('Ensemble: ' + model_str)
		
	


def RMSE(y_true, y_pred):
	"""
	Root-mean-square error (RMSE).
	Represents the sample standard deviation of the differences between
	predicted values and observed values.
	Has the same units as the quantity being estimated.
	https://en.wikipedia.org/wiki/Root-mean-square_deviation
	"""
	return np.sqrt(np.mean((y_true - y_pred) ** 2))
	

