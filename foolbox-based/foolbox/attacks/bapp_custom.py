from __future__ import print_function
from __future__ import division

import warnings
import time
import sys

from .base import Attack
from .base import call_decorator
from ..distances import MSE, Linf
import numpy as np
import math
import os

def cos_sim(x1, x2):
    cos = (x1*x2).sum() / np.sqrt( (x1**2).sum() * (x2**2).sum() )
    return cos

class BAPP_custom(Attack):
    """A powerful adversarial attack that requires neither gradients
    nor probabilities.

    Notes
    -----
    Features:
    * ability to switch between two types of distances: MSE and Linf.
    * ability to continue previous attacks by passing an instance of the
      Adversarial class
    * ability to pass an explicit starting point; especially to initialize
      a targeted attack
    * ability to pass an alternative attack used for initialization
    * ability to specify the batch size

    References
    ----------
    ..
    Code for QEBA attack. Paper arxiv link: https://arxiv.org/abs/2005.14137
    This code was modified based on the BAPP attack (now named HopSkipJumpAttack)
    Boundary Attack ++ was originally proposed by Chen and Jordan.
    It is a decision-based attack that requires access to output
    labels of a model alone.
    Paper link: https://arxiv.org/abs/1904.02144
    The implementation in Foolbox is based on Boundary Attack.
    """

    @call_decorator
    def __call__(
            self,
            input_or_adv,
            label=None,
            unpack=True,
            iterations=64,
            initial_num_evals=100,
            max_num_evals=10000,
            stepsize_search='grid_search',
            gamma=0.01,
            starting_point=None,
            batch_size=256,
            internal_dtype=np.float64,
            log_every_n_steps=1,
            verbose=False,
            rv_generator=None, atk_level=None,
            mask=None,
            save_calls=None,
            rho_ref = 0.0,
            discretize = False,
            suffix = '',
            plot_adv = True,
            ):
        """Applies QEBA Attack.

        Parameters
        ----------
        input_or_adv : `numpy.ndarray` or :class:`Adversarial`
            The original, correctly classified input. If it is a
            numpy array, label must be passed as well. If it is
            an :class:`Adversarial` instance, label must not be passed.
        label : int
            The reference label of the original input. Must be passed
            if input is a numpy array, must not be passed if input is
            an :class:`Adversarial` instance.
        unpack : bool
            If true, returns the adversarial input, otherwise returns
            the Adversarial object.
        iterations : int
            Number of iterations to run.
        initial_num_evals: int
            Initial number of evaluations for gradient estimation.
            Larger initial_num_evals increases time efficiency, but
            may decrease query efficiency.
        max_num_evals: int
            Maximum number of evaluations for gradient estimation.
        stepsize_search: str
            How to search for stepsize; choices are 'geometric_progression',
            'grid_search'. 'geometric progression' initializes the stepsize
            by ||x_t - x||_p / sqrt(iteration), and keep decreasing by half
            until reaching the target side of the boundary. 'grid_search'
            chooses the optimal epsilon over a grid, in the scale of
            ||x_t - x||_p.
        gamma: float
            The binary search threshold theta is gamma / sqrt(d) for
                   l2 attack and gamma / d for linf attack.

        starting_point : `numpy.ndarray`
            Adversarial input to use as a starting point, required
            for targeted attacks.
        batch_size : int
            Batch size for model prediction.
        internal_dtype : np.float32 or np.float64
            Higher precision might be slower but is numerically more stable.
        log_every_n_steps : int
            Determines verbositity of the logging.
        verbose : bool
            Controls verbosity of the attack.

        """

        self.initial_num_evals = initial_num_evals
        self.max_num_evals = max_num_evals
        self.stepsize_search = stepsize_search
        self.gamma = gamma
        self.batch_size = batch_size
        self.verbose = verbose
        self._starting_point = starting_point
        self.internal_dtype = internal_dtype
        self.log_every_n_steps = log_every_n_steps
        self.verbose = verbose
        self.rv_generator = rv_generator
        self.rho_ref = rho_ref
        self.discretize = discretize
        self.suffix = suffix
        self.plot_adv = plot_adv

        if mask is not None:
            print ("Generating patch")
            self.use_mask = True
            self.pert_mask = mask
            self.loss_mask = (1-mask)
        else:
            self.use_mask = False
            self.pert_mask = np.ones(input_or_adv.unperturbed.shape).astype(np.float32)
            self.loss_mask = np.ones(input_or_adv.unperturbed.shape).astype(np.float32)
        self.__mask_succeed = 0

        self.logger = []

        # Set constraint based on the distance.
        if self._default_distance == MSE:
            self.constraint = 'l2'
        elif self._default_distance == Linf:
            self.constraint = 'linf'

        # Set binary search threshold.
        self.shape = input_or_adv.unperturbed.shape
        self.fourier_basis_aux = None
        self.d = np.prod(self.shape)
        if self.constraint == 'l2':
            self.theta = self.gamma / np.sqrt(self.d)
        else:
            self.theta = self.gamma / (self.d)
        
        self.printv('Boundary Attack ++ optimized for {} distance'.format(
            self.constraint))

        #if not verbose:
        #    print('run with verbose=True to see details')

        self.save_calls = save_calls
        if save_calls is not None:
            if not os.path.isdir(save_calls):
                os.mkdir(save_calls)
            self.save_cnt = 0
            self.save_outs = []
            self.save_hashes = []

        return self.attack(
            input_or_adv,
            iterations=iterations, atk_level=atk_level)

    def gen_random_basis(self, N):
        basis = np.random.randn(N, *self.shape).astype(self.internal_dtype)
        #basis = randn_multithread(N, *self.shape).astype(self.internal_dtype)
        return basis

    def gen_custom_basis(self, N, sample, target, step, atk_level=None):
        if self.rv_generator is not None:
            basis = self.rv_generator.generate_ps(sample, N, atk_level).astype(self.internal_dtype)
        else:
            basis = self.gen_random_basis(N)
        return basis


    def attack(
            self,
            a,
            iterations, atk_level):
        """
        iterations : int
            Maximum number of iterations to run.
        """
        self.t_initial = time.time()

        # ===========================================================
        # Increase floating point precision
        # ===========================================================

        self.external_dtype = a.unperturbed.dtype

        assert self.internal_dtype in [np.float32, np.float64]
        assert self.external_dtype in [np.float32, np.float64]

        assert not (self.external_dtype == np.float64 and
                    self.internal_dtype == np.float32)

        a.set_distance_dtype(self.internal_dtype)

        # ===========================================================
        # Construct batch decision function with binary output.
        # ===========================================================
        # decision_function = lambda x: a.forward(
        #     x.astype(self.external_dtype), strict=False)[1]
        def decision_function(x):
            outs = []
            num_batchs = int(math.ceil(len(x) * 1.0 / self.batch_size))
            for j in range(num_batchs):
                current_batch = x[self.batch_size * j:
                                  self.batch_size * (j + 1)]
                current_batch = current_batch.astype(self.external_dtype)
                out = a.forward(current_batch, strict=False)[1]
                outs.append(out)
            outs = np.concatenate(outs, axis=0)

            #If save: save to self.save_calls
            if self.save_calls is not None:
                for one_x, one_out in zip(x, outs):
                    h = None
                    #h = hash(x.tostring())
                    if (h not in self.save_hashes):
                        np.save(self.save_calls+'%d.npy'%self.save_cnt, one_x)
                        self.save_outs.append(one_out)
                        #self.save_hashes.append(h)
                        self.save_cnt += 1
            return outs

        # ===========================================================
        # intialize time measurements
        # ===========================================================
        self.time_gradient_estimation = 0

        self.time_search = 0

        self.time_initialization = 0

        # ===========================================================
        # Initialize variables, constants, hyperparameters, etc.
        # ===========================================================

        # make sure repeated warnings are shown
        warnings.simplefilter('always', UserWarning)

        # get bounds
        bounds = a.bounds()
        self.clip_min, self.clip_max = bounds

        # ===========================================================
        # Find starting point
        # ===========================================================

        self.initialize_starting_point(a)

        if a.perturbed is None:
            warnings.warn(
                'Initialization failed.'
                ' it might be necessary to pass an explicit starting'
                ' point.')
            return

        self.time_initialization += time.time() - self.t_initial

        assert a.perturbed.dtype == self.external_dtype
        # get original and starting point in the right format
        original = a.unperturbed.astype(self.internal_dtype)
        perturbed = a.perturbed.astype(self.internal_dtype)

        # ===========================================================
        # Iteratively refine adversarial
        # ===========================================================
        t0 = time.time()

        # Project the initialization to the boundary.
        perturbed, dist_post_update, mask_succeed = self.binary_search_batch(
            original, np.expand_dims(perturbed, 0), decision_function)

        dist = self.compute_distance(perturbed, original)

        distance = a.distance.value
        self.time_search += time.time() - t0

        # log starting point
        self.log_step(0, distance, a=a, perturbed=perturbed)
        if mask_succeed > 0:
            self.__mask_succeed = 1
            self.log_time()
            return

        grad_gt_prev = None
        gradf_saved = []
        gradgt_saved = []
        prev_ps = [perturbed]

        ### Decision boundary direction ###
        for step in range(1, iterations + 1):
            ### Plot decision boundary ###
            #N = 20
            #plot_delta = self.select_delta(dist_post_update, step) / N * 3
            #import matplotlib
            #matplotlib.use('Agg')
            #import matplotlib.pyplot as plt
            #fig = plt.figure(figsize=(15,6))
            #for subp in range(10):
            #    print (subp)
            #    plt.subplot(2,5,subp+1)

            #    v1, v2 = sub_dirs[subp]
            #    if (subp < 2):
            #        v1 = (perturbed-original)
            #    v1 = v1 / np.linalg.norm(v1)

            #    xs = np.arange(-N,N+1) * plot_delta
            #    ys = np.arange(-N,N+1) * plot_delta
            #    vals = []
            #    for _ in range(2*N+1):
            #        query = perturbed + v1*xs[_] + v2*ys[:,None, None, None]
            #        val_cur = decision_function(query)
            #        vals.append(val_cur)
            #    plt.contourf(xs,ys,vals, levels=1)
            #    plt.axis('off')
            #fig.savefig('step%d_db_delta.png'%step)
            #plt.close(fig)
            #assert 0
            ### Plot end ###
            t0 = time.time()
            c0 = a._total_prediction_calls


            # ===========================================================
            # Gradient direction estimation.
            # ===========================================================
            # Choose delta.
            delta = self.select_delta(dist_post_update, step)

            # Choose number of evaluations.
            num_evals = int(min([self.initial_num_evals * np.sqrt(step),
                                 self.max_num_evals]))

            # approximate gradient.
            gradf, avg_val = self.approximate_gradient(decision_function, perturbed, a.unperturbed,
                                                           num_evals, delta, step=step, atk_level=atk_level)

            # Calculate auxiliary information for the exp
            import scipy as sp
            grad_gt = a._model.gradient_one(perturbed, label=a._criterion.target_class()) * self.pert_mask
            #grad_gt = np.ones_like(perturbed)
            dist_dir = original-perturbed
            if self.rv_generator is not None:
            #if False:
                #rho = self.rv_generator.calc_rho(grad_gt, perturbed).item()
                rho = self.rho_ref
            else:
                rho = 1.0
            #gradf = -grad_gt / np.linalg.norm(grad_gt) #oracle
            #cos1 = cos_sim(gradf, grad_gt)
            #rand = np.random.randn(*gradf.shape)
            #cos2 = cos_sim(grad_gt, rand)
            #self.printv("# evals: %.6f; with gt: %.6f; random with gt: %.6f"%(num_evals, cos1, cos2))
            #self.printv("\testiamted with dist: %.6f; gt with dist: %.6f"%(cos_sim(gradf, original-perturbed), cos_sim(grad_gt, original-perturbed)))


            if self.constraint == 'linf':
                update = np.sign(gradf)
            else:
                update = gradf
            t1 = time.time()
            c1 = a._total_prediction_calls
            self.time_gradient_estimation += t1 - t0

            # ===========================================================
            # Update, and binary search back to the boundary.
            # ===========================================================
            if self.stepsize_search == 'geometric_progression':
                # find step size.
                epsilon = self.geometric_progression_for_stepsize(
                    perturbed, update, dist, decision_function, step)

                # Update the sample.
                p_prev = perturbed
                perturbed = np.clip(perturbed + (epsilon * update).astype(self.internal_dtype), self.clip_min, self.clip_max)
                #actual_update = perturbed - p_prev
                #cos_actual = cos_sim(actual_update, grad_gt)
                #print ("Actual update vs. GT grad cos:", cos_actual)
                c2 = a._total_prediction_calls

                # Binary search to return to the boundary.
                perturbed, dist_post_update, mask_succeed = self.binary_search_batch(
                    original, perturbed[None], decision_function)
                c3 = a._total_prediction_calls

            elif self.stepsize_search == 'grid_search':
                # Grid search for stepsize.
                epsilons = np.logspace(-4, 0, num=20, endpoint=True) * dist
                epsilons_shape = [20] + len(self.shape) * [1]
                perturbeds = perturbed + epsilons.reshape(
                    epsilons_shape) * update
                perturbeds = np.clip(perturbeds, self.clip_min, self.clip_max)
                idx_perturbed = decision_function(perturbeds)

                if np.sum(idx_perturbed) > 0:
                    # Select the perturbation that yields the minimum
                    # distance after binary search.
                    perturbed, dist_post_update, mask_succeed = self.binary_search_batch(
                        original, perturbeds[idx_perturbed],
                        decision_function)
            t2 = time.time()

            #print (perturbed)
            #if self.discretize:
            #    perturbed = np.rint(perturbed * 255.0) / 255.0
            #print (perturbed)
            self.time_search += t2 - t1

            # compute new distance.
            dist = self.compute_distance(perturbed, original)

            # ===========================================================
            # Log the step
            # ===========================================================
            # Using foolbox definition of distance for logging.
            if self.constraint == 'l2':
                distance = dist ** 2 / self.d / \
                    (self.clip_max - self.clip_min) ** 2
            elif self.constraint == 'linf':
                distance = dist / (self.clip_max - self.clip_min)
            message = ' (took {:.5f} seconds)'.format(t2 - t0)
            self.log_step(step, distance, message, a=a, perturbed=perturbed, update=update*epsilon, aux_info=(gradf, grad_gt, dist_dir, rho))
            self.printv("Call in grad approx / geo progress / binary search: %d/%d/%d"%(c1-c0, c2-c1, c3-c2))
            sys.stdout.flush()
            a.__best_adversarial = perturbed

            if mask_succeed > 0:
                self.__mask_succeed = 1
                break


        # ===========================================================
        # Log overall runtime
        # ===========================================================
        self.log_time()

        #Save the labels
        if self.save_calls is not None:
            np.save(self.save_calls+'out.npy', self.save_outs)
            print ("Total saved calls: %d"%len(self.save_outs))

    # ===============================================================
    #
    # Other methods
    #
    # ===============================================================

    def initialize_starting_point(self, a):
        starting_point = self._starting_point

        if a.perturbed is not None:
            print(
                'Attack is applied to a previously found adversarial.'
                ' Continuing search for better adversarials.')
            if starting_point is not None:  # pragma: no cover
                warnings.warn(
                    'Ignoring starting_point parameter because the attack'
                    ' is applied to a previously found adversarial.')
            return

        if starting_point is not None:
            a.forward_one(starting_point)
            assert a.perturbed is not None, (
                'Invalid starting point provided. Please provide a starting point that is adversarial.')
            return

        """
        Apply BlendedUniformNoiseAttack if without
        initialization.
        Efficient Implementation of BlendedUniformNoiseAttack in Foolbox.
        """
        success = 0
        num_evals = 0

        while True:
            random_noise = np.random.uniform(self.clip_min, self.clip_max,
                                             size=self.shape)
            _, success = a.forward_one(
                random_noise.astype(self.external_dtype))
            num_evals += 1
            if success:
                break
            if num_evals > 1e4:
                return

        # Binary search to minimize l2 distance to the original input.
        low = 0.0
        high = 1.0
        while high - low > 0.001:
            mid = (high + low) / 2.0
            #blended = (1 - mid) * a.unperturbed + mid * random_noise
            blended = self.loss_mask * ((1 - mid) * a.unperturbed + mid * random_noise) + (1-self.loss_mask) * a.perturbed
            _, success = a.forward_one(blended.astype(self.external_dtype))
            if success:
                high = mid
            else:
                low = mid

    def compute_distance(self, x1, x2):
        if self.constraint == 'l2':
            #return np.linalg.norm(x1 - x2)
            return np.linalg.norm((x1 - x2) * self.loss_mask)
        elif self.constraint == 'linf':
            return np.max(abs(x1 - x2))

    def project(self, unperturbed, perturbed_inputs, alphas):
        """ Projection onto given l2 / linf balls in a batch. """
        alphas_shape = [len(alphas)] + [1] * len(self.shape)
        alphas = alphas.reshape(alphas_shape)
        if self.constraint == 'l2':
            #projected = (1 - alphas) * unperturbed + \
            #    alphas * perturbed_inputs
            projected = self.loss_mask * ((1 - alphas) * unperturbed + alphas * perturbed_inputs) + (1-self.loss_mask) * perturbed_inputs
            #normed = np.zeros_like(perturbed_inputs) + 0.5
            ##norm_alpha = np.sqrt(alphas)
            #norm_alpha = alphas**2
            #projected = self.loss_mask * ((1 - alphas) * unperturbed + alphas * perturbed_inputs) + (1-self.loss_mask) * ( (1-norm_alpha)*normed + norm_alpha * perturbed_inputs)
        elif self.constraint == 'linf':
            projected = np.clip(perturbed_inputs, unperturbed - alphas, unperturbed + alphas)
        return projected

    def binary_search_batch(self, unperturbed, perturbed_inputs,
                            decision_function):
        """ Binary search to approach the boundary. """

        # Compute distance between each of perturbed and unperturbed input.
        dists_post_update = np.array(
            [self.compute_distance(unperturbed,
                                   perturbed_x) for perturbed_x in
             perturbed_inputs])

        # Choose upper thresholds in binary searchs based on constraint.
        if self.constraint == 'linf':
            highs = dists_post_update
            # Stopping criteria.
            thresholds = np.minimum(dists_post_update * self.theta,
                                    self.theta)
        else:
            highs = np.ones(len(perturbed_inputs))
            thresholds = self.theta

        lows = np.zeros(len(perturbed_inputs))
        lows = lows.astype(self.internal_dtype)
        highs = highs.astype(self.internal_dtype)

        if self.use_mask:
            _mask = np.array([self.pert_mask] * len(perturbed_inputs))
            masked = perturbed_inputs * _mask + unperturbed * (1 - _mask)
            masked_decisions = decision_function(masked)
            highs[masked_decisions == 1] = 0
            succeed = (np.sum(masked_decisions) > 0)
        else:
            succeed = False
        # Call recursive function.
        while np.max((highs - lows) / thresholds) > 1:
            # projection to mids.
            mids = (highs + lows) / 2.0
            mid_inputs = self.project(unperturbed, perturbed_inputs,
                                      mids)

            # Update highs and lows based on model decisions.
            decisions = decision_function(mid_inputs)
            lows = np.where(decisions == 0, mids, lows)
            highs = np.where(decisions == 1, mids, highs)

        out_inputs = self.project(unperturbed, perturbed_inputs,
                                  highs)

        # Compute distance of the output to select the best choice.
        # (only used when stepsize_search is grid_search.)
        dists = np.array([
            self.compute_distance(
                unperturbed,
                out
            )
            for out in out_inputs])
        idx = np.argmin(dists)

        dist = dists_post_update[idx]
        out = out_inputs[idx]
        return out, dist, succeed

    def select_delta(self, dist_post_update, current_iteration):
        """
        Choose the delta at the scale of distance
        between x and perturbed sample.
        """
        if current_iteration == 1:
            delta = 0.1 * (self.clip_max - self.clip_min)
        else:
            if self.constraint == 'l2':
                delta = np.sqrt(self.d) * self.theta * dist_post_update
            elif self.constraint == 'linf':
                delta = self.d * self.theta * dist_post_update

        return delta

    def approximate_gradient(self, decision_function, sample, target,
                             num_evals, delta, step=None, atk_level=None):
        """ Gradient direction estimation """
        #import time
        #t0 = time.time()
        axis = tuple(range(1, 1 + len(self.shape)))

        # Generate random vectors.
        noise_shape = [num_evals] + list(self.shape)

        #t1 = time.time()
        rv_raw = self.gen_custom_basis(num_evals, sample=sample, target=target, step=step, atk_level=atk_level)
        #t2 = time.time()

        _mask = np.array([self.pert_mask] * num_evals)
        rv = rv_raw * _mask
        rv = rv / np.sqrt(np.sum(rv ** 2, axis=axis, keepdims=True))
        rv_print = delta * np.abs(rv)
        perturbed = sample + delta * rv
        perturbed = np.clip(perturbed, self.clip_min, self.clip_max)
        if self.discretize:
            perturbed = np.rint(perturbed*255.0)/255.0
            #sgn = np.sign(rv) #
            #perturbed[perturbed==sample] = 
        rv = (perturbed - sample) / delta
        #if self.discretize:
        #    rv[rv==0] = np.sign(rv_raw)[rv==0] / 255.0 / delta
        #    rv = rv * _mask
        #    perturbed = sample + delta * rv
        #    perturbed = np.clip(perturbed, self.clip_min, self.clip_max)
        #    rv = (perturbed - sample) / delta
        #t3 = time.time()
        #perturbed_neg = sample - delta * rv
        #print (perturbed*255.0)


        # query the model.
        decisions = decision_function(perturbed)
        #t4 = time.time()
        decision_shape = [len(decisions)] + [1] * len(self.shape)
        fval = 2 * decisions.astype(self.internal_dtype).reshape(
            decision_shape) - 1.0

        # Baseline subtraction (when fval differs)
        vals = fval if abs(np.mean(fval)) == 1.0 else fval - np.mean(fval)
        #vals = fval
        gradf = np.mean(vals * rv, axis=0)

        # Get the gradient direction.
        gradf = gradf / np.linalg.norm(gradf)
        #print (cos_sim(gradf, (target-sample)))
        #assert 0
        #t5 = time.time()
        #print ("Tot time:",t5-t0,"Detail",t1-t0,t2-t1,t3-t2,t4-t3,t5-t4)

        return gradf, np.mean(fval)

    def geometric_progression_for_stepsize(self, x, update, dist,
                                           decision_function,
                                           current_iteration):
        """ Geometric progression to search for stepsize.
          Keep decreasing stepsize by half until reaching
          the desired side of the boundary.
        """
        if self.use_mask:
            size_ratio = np.sqrt(self.pert_mask.sum() / self.pert_mask.size)
            #size_ratio = 1.0
            epsilon = dist * size_ratio / np.sqrt(current_iteration) + 0.1
            #epsilon = dist * size_ratio + 0.1
        else:
            epsilon = dist / np.sqrt(current_iteration)
        while True:
            updated = np.clip(x + epsilon * update, self.clip_min, self.clip_max)
            success = decision_function(updated[None])[0]
            if success:
                break
            else:
                epsilon = epsilon / 2.0  # pragma: no cover
                #print ("Geo progress decrease eps at %.4f"%epsilon)

        return epsilon

    def log_step(self, step, distance, message='', always=False, a=None, perturbed=None, update=None, aux_info=None):
        assert len(self.logger) == step
        if aux_info is not None:
            gradf, grad_gt, dist_dir, rho = aux_info
            cos_est = cos_sim(-gradf, grad_gt)
            cos_distpred = cos_sim(dist_dir, -gradf)
            cos_distgt = cos_sim(dist_dir, grad_gt)

            self.logger.append((a._total_prediction_calls, distance, cos_est.item(), rho, cos_distpred.item(), cos_distgt.item()))
            #cos1 = cos_sim(gradf, grad_gt)
            #rand = np.random.randn(*gradf.shape)
            #cos2 = cos_sim(grad_gt, rand)
            #print ("# evals: %.6f; with gt: %.6f; random with gt: %.6f"%(num_evals, cos1, cos2))
            #print ("\testiamted with dist: %.6f; gt with dist: %.6f"%(cos_sim(gradf, original-perturbed), cos_sim(grad_gt, original-perturbed)))
        else:
            self.logger.append((a._total_prediction_calls, distance, 0, 0, 0, 0))
        if not always and step % self.log_every_n_steps != 0:
            return
        self.printv('Step {}: {:.5e} {}'.format(
            step,
            distance,
            message))
        if aux_info is not None:
            self.printv("\tEstimated vs. GT: %.6f"%cos_est)
            self.printv("\tRho: %.6f"%rho)
            self.printv("\tEstimated vs. Distance: %.6f"%cos_distpred)
            self.printv("\tGT vs. Distance: %.6f"%cos_distgt)
        if not self.plot_adv:
            return # Dont plot
        if a is not None:
            import matplotlib.pyplot as plt
            fig = plt.figure()
            #plt.imshow(perturbed[:,:,::-1]/255)  #keras
            plt.imshow(perturbed.transpose(1,2,0))  #pytorch
            np.savez('BAPP_result/perturbed%s%d.npz' % (self.suffix, step), pert=perturbed.transpose(1,2,0),
                     info=np.array([a._total_prediction_calls, distance]))
            #plt.imshow((perturbed+1)/2)  #receipt
            #plt.imshow(perturbed/255)  #face
            #np.savez('BAPP_result/perturbed%s%d.npz' % (self.suffix, step), pert=perturbed/255,
            #         info=np.array([a._total_prediction_calls, distance]))

            plt.axis('off')
            plt.title('Call %d Distance %f'%(a._total_prediction_calls, distance))
            fig.savefig('BAPP_result/%sstep%d.png' % (self.suffix, step))
            plt.close(fig)
            if update is not None:
                #print (np.linalg.norm(update))
                fig = plt.figure()
                abs_update = (update - update.min()) / (update.max() - update.min())
                #plt.imshow(abs_update[:,:,::-1])  #keras
                plt.imshow(abs_update.transpose(1,2,0))  #pytorch
                #plt.imshow(abs_update)  #receipt
                #plt.imshow(abs_update/255)  #face
                plt.axis('off')
                plt.title('Call %d Distance %f'%(a._total_prediction_calls, distance))
                fig.savefig('BAPP_result/update%d.png'%step)
                plt.close(fig)
            #Saliency map
            #import cv2
            #saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
            #img = perturbed.transpose(1,2,0)
            #img = (img*255).astype(np.uint8)
            #fig = plt.figure()
            #(success, saliencyMap) = saliency.computeSaliency(img)
            #assert success
            #plt.imshow(saliencyMap, cmap='gray')
            #fig.savefig('BAPP_result/saliency%d.png'%step)
            #
            self.printv("Call:", a._total_prediction_calls, "Saved to", 'BAPP_result/%sstep%d.png' % (self.suffix, step))

    def log_time(self):
        t_total = time.time() - self.t_initial
        rel_initialization = self.time_initialization / t_total
        rel_gradient_estimation = self.time_gradient_estimation / t_total
        rel_search = self.time_search / t_total

        self.printv('Time since beginning: {:.5f}'.format(t_total))
        self.printv('   {:2.1f}% for initialization ({:.5f})'.format(
            rel_initialization * 100, self.time_initialization))
        self.printv('   {:2.1f}% for gradient estimation ({:.5f})'.format(
            rel_gradient_estimation * 100,
            self.time_gradient_estimation))
        self.printv('   {:2.1f}% for search ({:.5f})'.format(
            rel_search * 100, self.time_search))

    def printv(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)
