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

def cos_sim(x1, x2):
    cos = (x1*x2).sum() / np.sqrt( (x1**2).sum() * (x2**2).sum() )
    return cos

def calc_ortho_update(diff, grad_update):
    a = (diff**2).sum()
    b = (diff*grad_update).sum()
    c = (grad_update**2).sum()

    y = np.sqrt( (a*c-c*c) / (a*c-b*b) )
    x = (c-b*y) / a
    grad_ortho = x * diff + y * grad_update
    #if (grad_ortho*grad_update).sum() < 0:
    #    grad_ortho = -grad_ortho
    print (a, b, c)
    print (x, y)
    print ( cos_sim(grad_ortho, grad_update) )
    print ( cos_sim(diff, grad_update) )
    print ( cos_sim(grad_ortho, diff) )
    assert (grad_ortho*grad_update).sum() > 0
    return grad_ortho

def randn_multithread(*shape):
    assert 0, "Not working"
    N = shape[0]

    def gen_one_rv(L_rv, indices):
        print ("thread %d begins"%(indices[0]))
        for idx in indices:
            rv = np.random.randn(*shape[1:])
            #rv = np.zeros(shape[1:])
            #time.sleep(np.random.randint(1,3))
            L_rv.append(rv)
        print ("thread %d ends"%(indices[0]))

    from multiprocessing import Process, Manager
    with Manager() as manager:
        rv_list = manager.list()
        n_jobs = 64
        ps = []
        for i in range(n_jobs):
            cur_ids = range(i, N, n_jobs)
            p = Process(target=gen_one_rv, args=(rv_list, cur_ids))
            p.start()
            ps.append(p)
        for p in ps:
            p.join()
        rv_list = list(rv_list)

    rvs = np.array(rv_list)
    return rvs


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
            mask=None,):
        """Applies Boundary Attack++.

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
        print('Boundary Attack ++ optimized for {} distance'.format(
            self.constraint))

        if not verbose:
            print('run with verbose=True to see details')

        return self.attack(
            input_or_adv,
            iterations=iterations, atk_level=atk_level)

    def gen_fourier_basis(self, N):
        # Fourier basis
        if self.fourier_basis_aux is None:
            theta = 2*np.pi/self.d
            self.ortho_basis_aux = np.cos(np.arange(self.d)*theta)
            #np.random.shuffle(self.ortho_basis_aux)
        idxes = np.random.choice(self.d, N, replace=False)
        bits = (np.arange(self.d)[None]*idxes[:,None]) % self.d
        basis = self.ortho_basis_aux[bits].reshape(N, *self.shape)
        return basis

    def gen_fourier_sgn_basis(self, N):
        # Fourier basis
        if self.fourier_basis_aux is None:
            theta = 2*np.pi/self.d
            self.ortho_basis_aux = np.cos(np.arange(self.d)*theta)
            #np.random.shuffle(self.ortho_basis_aux)
        idxes = np.random.choice(self.d, N, replace=False)
        bits = (np.arange(self.d)[None]*idxes[:,None]) % self.d
        basis = self.ortho_basis_aux[bits].reshape(N, *self.shape)
        basis = np.sign(basis)
        return basis

    def gen_unit_basis(self, N):
        # unit basis
        basis = np.zeros((N, self.d))
        idxes = np.random.randint(self.d, size=(N,))
        basis[np.arange(N), idxes] = 1
        basis = basis.reshape(N, *self.shape)
        return basis

    def gen_random_basis(self, N):
        basis = np.random.randn(N, *self.shape).astype(self.internal_dtype)
        #basis = randn_multithread(N, *self.shape).astype(self.internal_dtype)
        return basis

    def gen_custom_basis(self, N, sample, target, step, atk_level=None):
        #Focus more on medium value
        #stddev = np.abs(sample-128) / 128
        #basis = np.random.randn(N, *self.shape) * stddev[None]

        #Smooth vector
        #import scipy.signal as signal
        #basis = np.random.randn(N, *self.shape)
        #basis = signal.medfilt(basis, kernel_size=(1,1,3,3))

        #Orthogonal to the distance
        #axis = tuple(range(1, len(self.shape)))
        #norm_v = (sample - target)[None]
        #basis = np.random.randn(N, *self.shape)
        #basis_proj = ((basis * norm_v).sum(axis=axis, keepdims=True)) / ((norm_v**2).sum(axis=axis, keepdims=True)) * norm_v
        #basis = basis - basis_proj

        #Generated perturbation
        #basis = np.random.randn(N, *self.shape).astype(self.internal_dtype)

        if self.rv_generator is not None:
            #if step < 70:
            #    level = 0
            #else:
            #    level = (step-20) // 50
            #    N = N*int(np.sqrt(level+1))

            #level = 0
            #basis = self.rv_generator.generate_ps(sample, N, level).astype(self.internal_dtype)

            basis = self.rv_generator.generate_ps(sample, N, atk_level).astype(self.internal_dtype)

            #if step < 9999:
            #    basis = self.rv_generator.generate_ps(sample, N).astype(self.internal_dtype)
            #else:
            #    basis = self.gen_random_basis(N)
        else:
            basis = self.gen_random_basis(N)
        #Orthogonalize
        #axis = tuple(range(1, len(self.shape)))
        #norm_v = (sample - target)[None]
        #basis_proj = ((basis * norm_v).sum(axis=axis, keepdims=True)) / ((norm_v**2).sum(axis=axis, keepdims=True)) * norm_v
        #basis = basis - basis_proj
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
        #sub_dirs = []
        #for subp in range(10):
        #    v1, v2 = np.random.randn(2, *self.shape)
        #    v1 = v1 / np.linalg.norm(v1)
        #    v2 = v2 / np.linalg.norm(v2)
        #    sub_dirs.append(((v1, v2)))

        for step in range(1, iterations + 1):
            ### Plot decision boundary ###
            #N = 10
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
            ##assert 0
            ### Plot end ###

            t0 = time.time()
            c0 = a._total_prediction_calls


            # ===========================================================
            # Gradient direction estimation.
            # ===========================================================
            # Choose delta.
            delta = self.select_delta(dist_post_update, step)
            print ("Delta:", delta)

            # Choose number of evaluations.
            num_evals = int(min([self.initial_num_evals * np.sqrt(step),
                                 self.max_num_evals]))

            # approximate gradient.
            v = a._model.forward_one(perturbed)
            print (v[a._criterion.target_class()], v[a.original_class])
            v = v[a._criterion.target_class()] - v[a.original_class]
            g = np.linalg.norm(a._model.gradient_one(perturbed, label=a._criterion.target_class()) - a._model.gradient_one(perturbed, label=a.original_class))
            print ("Value: %.4f; Grad: %.4f; approx dist: %.4f; delta: %.4f"%(v, g, v/g, delta))
            #assert 0
            ### Saliency map info ###
            #import cv2
            #saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
            #img = perturbed.transpose(1,2,0)
            #img = (img*255).astype(np.uint8)
            #(success, saliencyMap) = saliency.computeSaliency(img)
            #self.pert_mask = (saliencyMap > np.median(saliencyMap))
            #assert success
            ### Saliency end ###
            t_approx = time.time()
            current_g, avg_val = self.approximate_gradient(decision_function, perturbed, a.unperturbed,
                                                           num_evals, delta, step=step, atk_level=atk_level)
            print ("Approximate time:", time.time() - t_approx)
            #current_g, hessian, avg_val = self.approximate_gradient_and_hessian(decision_function, perturbed, a.unperturbed,
            #                                               num_evals, delta, step=step, a=a)
            ###
            ##grad_gt = a.gradient_one(perturbed, label=a._criterion.target_class())
            ##current_g = -grad_gt / np.linalg.norm(grad_gt)

            #print (hessian)
            #print (hessian.max(), hessian.min(), hessian.mean(), hessian.std())
            #print (dist)
            #print (a.forward_one(perturbed, strict=False)[0][a._criterion.target_class()])
            #print (np.linalg.norm(current_g))
            #print (a.forward_one(perturbed+current_g, strict=False)[0][a._criterion.target_class()])

            #hess_norm = np.clip(hessian+0.006, 0.001, 0.011)
            #print (hess_norm.max(), hess_norm.min(), hess_norm.mean(), hess_norm.std())
            #normed_g = current_g / hess_norm
            #normed_g = normed_g / np.linalg.norm(normed_g)
            #print (cos_sim(current_g, normed_g))
            #print (np.linalg.norm(normed_g))
            #print (a.forward_one(perturbed+normed_g, strict=False)[0][a._criterion.target_class()])
            #assert 0
            ###
            #gradf = self.approximate_gradient(decision_function, perturbed, a.unperturbed,
            #                                  num_evals, delta, chosen_dir = gradf_saved)

            #if len(gradf_saved) > 0:
            #    history_g = self.historical_gradient(decision_function, perturbed, gradf_saved, delta, avg_val, norm_v = (original-perturbed))
            #    if len(gradf_saved) >= 999:
            #        gradf = current_g + history_g
            #    else:
            #        gradf = current_g
            #    #gradf = current_g*10 + history_g*step
            #    gradf = gradf / np.linalg.norm(gradf)
            #else:
            #    gradf = current_g
            #    self.history = current_g
            #    history_g = None
            #gradf_saved.append(current_g)
            ##gradf_saved.append(gradf)
            history_g = None
            gradf = current_g

            import scipy as sp
            grad_gt = a.gradient_one(perturbed, label=a._criterion.target_class()) * self.pert_mask
            #gradf = -grad_gt / np.linalg.norm(grad_gt) #oracle
            cos1 = cos_sim(gradf, grad_gt)
            rand = np.random.randn(*gradf.shape)
            cos2 = cos_sim(grad_gt, rand)
            print ("# evals: %.6f; with gt: %.6f; random with gt: %.6f"%(num_evals, cos1, cos2))
            print ("\testiamted with dist: %.6f; gt with dist: %.6f"%(cos_sim(gradf, original-perturbed), cos_sim(grad_gt, original-perturbed)))
            if history_g is not None:
                print ("\tCurrent g with gt: %.6f; history g with gt: %.6f"%(cos_sim(current_g, grad_gt), cos_sim(history_g, grad_gt)))
                print ("\tCurrent g with dist: %.6f; history g with dist: %.6f"%(cos_sim(current_g, original-perturbed), cos_sim(history_g, original-perturbed)))
            else:
                print ("\tCurrent g with gt: %.6f"%cos_sim(current_g, grad_gt))

            #for g_p in gradf_saved:
            #    print ("\tPrevious estimated with gt: %.6f; with cur g: %.4f; with dist: %.4f"%(cos_sim(g_p, grad_gt), cos_sim(g_p, gradf), cos_sim(g_p, original-perturbed)))
            #gradgt_saved.append(grad_gt)
            #for g_gt in gradgt_saved:
            #    print ("\tPrevious gt with gt: %.6f; with cur g: %.4f; with dist: %.4f"%(cos_sim(g_gt, grad_gt), cos_sim(g_gt, gradf), cos_sim(g_gt, original-perturbed)))

            #print (hessian.shape)
            #import matplotlib.pyplot as plt
            #fig = plt.figure()
            #plt.subplot(1,3,1)
            #plt.imshow(hessian[:,:,0], cmap='hot')
            #plt.subplot(1,3,2)
            #plt.imshow(hessian[:,:,1], cmap='hot')
            #plt.subplot(1,3,3)
            #plt.imshow(hessian[:,:,2], cmap='hot')
            #plt.colorbar()  
            #plt.show()
            #assert 0

            if self.constraint == 'linf':
                update = np.sign(gradf)
            else:
                update = gradf

                #hess_norm = np.maximum(-hessian+0.005, 0) + 0.005
                ##hess_norm = np.maximum(hessian, 0) + 0.05
                #print (np.linalg.norm(gradf))
                #update = gradf / hess_norm
                #update = update / np.linalg.norm(update)
                #print (np.linalg.norm(update))
                #print (hessian.max())
                #print (hessian.min())
                ##assert 0
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

                # Clip so that grad is orthogonal *mine* #cant work: may lead the clipped perturbed to be predicted as source
                #if step > 1:
                #    diff = a.unperturbed - perturbed
                #    grad_ortho = calc_ortho_update(diff, update*epsilon)
                #    update = grad_ortho/epsilon
                #    print ("ortho with gt: %.6f"%(cos_sim(grad_ortho, grad_gt)))
                #    #print ((grad_ortho*(diff-grad_ortho)).sum())

                # Use only the orthogonal subspace *mine*
                #diff = a.unperturbed - perturbed
                #update_proj = ((update * diff).sum()) / ((diff**2).sum()) * diff
                #update = update - update_proj
                #print (cos_sim(update, diff))


                # Update the sample.
                p_prev = perturbed
                perturbed = np.clip(perturbed + (epsilon * update).astype(self.internal_dtype), self.clip_min, self.clip_max)
                actual_update = perturbed - p_prev
                cos_actual = cos_sim(actual_update, grad_gt)
                print ("Actual update vs. GT grad cos:", cos_actual)
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

            self.time_search += t2 - t1

            # compute new distance.
            dist = self.compute_distance(perturbed, original)
            #prev_ps.append(perturbed)
            #for p in prev_ps:
            #    print ("\tPrevious image with cur: %.4f"%self.compute_distance(perturbed, p))

            # ===========================================================
            # Log the step
            # ===========================================================
            # Using foolbox definition of distance for logging.
            if self.constraint == 'l2':
                distance = dist ** 2 / self.d / \
                    (self.clip_max - self.clip_min) ** 2
            elif self.constraint == 'linf':
                distance = dist / (self.clip_max - self.clip_min)
            #if self.constraint == 'l2':
            #    distance = np.linalg.norm((perturbed-original)*self.loss_mask)*(self.loss_mask.size / self.loss_mask.sum())
            #    distance = distance**2 / self.d / (self.clip_max-self.clip_min)**2
            #else:
            #    raise NotImplementedError()
            message = ' (took {:.5f} seconds)'.format(t2 - t0)
            self.log_step(step, distance, message, a=a, perturbed=perturbed, update=update*epsilon)
            print ("Call in grad approx / geo progress / binary search: %d/%d/%d"%(c1-c0, c2-c1, c3-c2))
            sys.stdout.flush()
            a.__best_adversarial = perturbed

            if mask_succeed > 0:
                self.__mask_succeed = 1
                break


        # ===========================================================
        # Log overall runtime
        # ===========================================================

        self.log_time()

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
        #while np.max((highs - lows) / thresholds) > 0.01:
            # projection to mids.
            mids = (highs + lows) / 2.0
            mid_inputs = self.project(unperturbed, perturbed_inputs,
                                      mids)

            # Update highs and lows based on model decisions.
            decisions = decision_function(mid_inputs)
            lows = np.where(decisions == 0, mids, lows)
            highs = np.where(decisions == 1, mids, highs)

        print ("Binary:", highs)
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
                             num_evals, delta, chosen_dir = None, step=None, atk_level=None):
        """ Gradient direction estimation """
        axis = tuple(range(1, 1 + len(self.shape)))

        # Generate random vectors.
        noise_shape = [num_evals] + list(self.shape)

        #rv = self.gen_random_basis(num_evals)
        #rv = self.gen_fourier_basis(num_evals) # use the orthonormal basis one by one *mine*
        #rv = self.gen_fourier_sgn_basis(num_evals) # use the orthonormal basis one by one *mine*
        #rv = self.gen_unit_basis(num_evals) # use the orthonormal basis one by one *mine*
        rv = self.gen_custom_basis(num_evals, sample=sample, target=target, step=step, atk_level=atk_level)
        #rv = np.concatenate([self.gen_fourier_basis(int(num_evals*0.5)), self.gen_unit_basis(int(num_evals*0.5))], axis=0)
        #if step > 3:
        #    rv = self.gen_random_basis(num_evals)
        #else:
        #    #rv = self.gen_unit_basis(num_evals)
        #    rv = self.gen_unit_basis(num_evals)

        #if self.constraint == 'l2':
        #    rv = np.random.randn(*noise_shape)
        #elif self.constraint == 'linf':
        #    rv = np.random.uniform(low=-1, high=1, size=noise_shape)
        if chosen_dir is not None and len(chosen_dir) > 0:
            print (rv.shape)
            print (np.array(chosen_dir).shape)
            rv = np.concatenate([rv, np.array(chosen_dir)], axis=0)
            print (rv.shape)

        _mask = np.array([self.pert_mask] * num_evals)
        rv = rv * _mask
        rv = rv / np.sqrt(np.sum(rv ** 2, axis=axis, keepdims=True))
        perturbed = sample + delta * rv
        perturbed = np.clip(perturbed, self.clip_min, self.clip_max)
        rv = (perturbed - sample) / delta
        #perturbed_neg = sample - delta * rv


        # query the model.
        decisions = decision_function(perturbed)
        #decisions_neg = decision_function(perturbed_neg)
        #decisions = (decisions.astype(self.internal_dtype) - decisions_neg.astype(self.internal_dtype))/2
        decision_shape = [len(decisions)] + [1] * len(self.shape)
        fval = 2 * decisions.astype(self.internal_dtype).reshape(
            decision_shape) - 1.0

        # Baseline subtraction (when fval differs)
        print ("Mean val:", np.mean(fval))
        vals = fval if abs(np.mean(fval)) == 1.0 else fval - np.mean(fval)
        #vals = fval
        gradf = np.mean(vals * rv, axis=0)

        # Get the gradient direction.
        gradf = gradf / np.linalg.norm(gradf)
        #print (cos_sim(gradf, (target-sample)))
        #assert 0

        return gradf, np.mean(fval)

    def approximate_gradient_and_hessian(self, decision_function, sample, target,
                             num_evals, delta, chosen_dir = None, step=None, a=None):
        #np.random.seed(1)
        """ Gradient direction estimation """
        axis = tuple(range(1, 1 + len(self.shape)))

        # Generate random vectors.
        noise_shape = [num_evals] + list(self.shape)

        rv = self.gen_random_basis(num_evals)
        #rv = self.gen_fourier_basis(num_evals) # use the orthonormal basis one by one *mine*
        #rv = self.gen_fourier_sgn_basis(num_evals) # use the orthonormal basis one by one *mine*
        #rv = self.gen_unit_basis(num_evals) # use the orthonormal basis one by one *mine*
        #rv = self.gen_custom_basis(num_evals, sample=sample, target=target)

        #rv = np.concatenate([self.gen_fourier_basis(int(num_evals*0.5)), self.gen_unit_basis(int(num_evals*0.5))], axis=0)
        #if step > 3:
        #    rv = self.gen_random_basis(num_evals)
        #else:
        #    #rv = self.gen_unit_basis(num_evals)
        #    rv = self.gen_unit_basis(num_evals)

        #if self.constraint == 'l2':
        #    rv = np.random.randn(*noise_shape)
        #elif self.constraint == 'linf':
        #    rv = np.random.uniform(low=-1, high=1, size=noise_shape)
        if chosen_dir is not None and len(chosen_dir) > 0:
            print (rv.shape)
            print (np.array(chosen_dir).shape)
            rv = np.concatenate([rv, np.array(chosen_dir)], axis=0)
            print (rv.shape)

        rv = rv / np.sqrt(np.sum(rv ** 2, axis=axis, keepdims=True))
        perturbed = sample + delta * rv
        perturbed = np.clip(perturbed, self.clip_min, self.clip_max)
        rv = (perturbed - sample) / delta

        # query the model.
        decisions = decision_function(perturbed)
        decision_shape = [len(decisions)] + [1] * len(self.shape)
        fval = 2 * decisions.astype(self.internal_dtype).reshape(
            decision_shape) - 1.0
        #print ("RV decision:", decisions)
        perturbed_neg = sample - delta*rv
        #print ("RV neg decision:", decision_function(perturbed_neg))

        # Baseline subtraction (when fval differs)
        print ("Mean val:", np.mean(fval))
        vals = fval if abs(np.mean(fval)) == 1.0 else fval - np.mean(fval)
        gradf = np.mean(vals * rv, axis=0)

        # Get the gradient direction.
        gradf = gradf / np.linalg.norm(gradf)
        #print (cos_sim(gradf, (target-sample)))
        #assert 0

        # Determine the convexity direction
        #rv_aux = np.random.randn(*rv.shape)
        #aux_proj = ((rv_aux*rv).sum(axis=axis, keepdims=True)) / ((rv**2).sum(axis=axis, keepdims=True)) * rv
        #rv_ortho = rv_aux - aux_proj   #Generate orthogonal variables
        #rv_ortho = rv_ortho / np.sqrt(np.sum(rv_ortho ** 2, axis=axis, keepdims=True))
        #perturbed_pos = sample + delta * rv_ortho
        #perturbed_neg = sample - delta * rv_ortho
        #decision_pos = decision_function(perturbed_pos)
        #decision_neg = decision_function(perturbed_neg)
        #print (delta)
        #print ("Pos decision:", decision_pos)
        #print ("Neg decision:", decision_neg)

        #deltas = np.array([0.1*delta for _ in range(num_evals)])
        #deltas_shape = [num_evals] + [1]*len(self.shape)
        #print (deltas.shape)
        #num_tries = 0
        #while (num_tries < 10):
        #    print (num_tries)
        #    num_tries += 1
        #    perturbed_pos = sample + deltas.reshape(deltas_shape) * rv_ortho
        #    perturbed_neg = sample - deltas.reshape(deltas_shape) * rv_ortho
        #    decision_pos = decision_function(perturbed_pos)
        #    decision_neg = decision_function(perturbed_neg)
        #    print ("Pos decision:", decision_pos)
        #    print ("Neg decision:", decision_neg)
        #    if (np.logical_xor(decision_pos, decision_neg).all()):
        #        break
        #    deltas = np.where(np.logical_xor(decision_pos, decision_neg), deltas, deltas*2)
        #print ()
        #print (deltas)
        #print ("Pos decision:", decision_pos)
        #print ("Neg decision:", decision_neg)
        #print ("Different:", np.logical_xor(decision_pos, decision_neg))
        #assert np.logical_xor(decision_pos, decision_neg).all()

        #rv_redir = rv * ( (decisions != decision_pos).reshape(deltas_shape)*2 - 1 )
        #print (cos_sim(rv[0], rv_redir[0]))
        #print (cos_sim(rv[1], rv_redir[1]))
        #print (cos_sim(rv[2], rv_redir[2]))
        #print (cos_sim(rv[3], rv_redir[3]))
        #print (cos_sim(rv[4], rv_redir[4]))
        #perturbed_check = sample + delta * rv_redir
        #decisions_check = decision_function(perturbed_check)
        #print ("Redirect RV decision:", decisions_check)
        #assert 0

        #perturbed_pos = sample + deltas.reshape(deltas_shape) * rv_ortho + 10.0 * rv_redir
        #print ("pos+10redir RV:", decision_function(perturbed_pos))
        #perturbed_pos = sample + deltas.reshape(deltas_shape) * rv_ortho + 30.0 * rv_redir
        #print ("pos+30redir RV:", decision_function(perturbed_pos))
        #perturbed_pos = sample + deltas.reshape(deltas_shape) * rv_ortho + 100.0 * rv_redir
        #print ("pos+100redir RV:", decision_function(perturbed_pos))
        #perturbed_pos = sample + deltas.reshape(deltas_shape) * rv_ortho + 300.0 * rv_redir
        #print ("pos+300redir RV:", decision_function(perturbed_pos))
        #perturbed_pos = sample + deltas.reshape(deltas_shape) * rv_ortho + 1000.0 * rv_redir
        #print ("pos+1000redir RV:", decision_function(perturbed_pos))
        #perturbed_neg = sample + deltas.reshape(deltas_shape) * rv_ortho - 10.0 * rv_redir
        #print ("pos-10redir RV:", decision_function(perturbed_neg))
        #assert 0

        #lows = np.zeros(num_evals)
        #highs = np.ones(num_evals) * 0.1
        #num_tries = 0
        #while (num_tries < 10):
        #    num_tries += 1
        #    dists = highs * deltas
        #    perturbed_pos = sample + deltas.reshape(deltas_shape) * rv_ortho + dists.reshape(deltas_shape) * rv_redir
        #    perturbed_neg = sample - deltas.reshape(deltas_shape) * rv_ortho - dists.reshape(deltas_shape) * rv_redir
        #    decision_new_pos = decision_function(perturbed_pos)
        #    decision_new_neg = decision_function(perturbed_neg)
        #    print ()
        #    print (highs)
        #    print (decision_new_pos)
        #    print (decision_new_neg)
        #    diff = np.logical_or(decision_new_pos != decision_pos, decision_new_neg != decision_neg)
        #    if (diff.all()):
        #        break
        #    lows = np.where(diff, lows, highs)
        #    highs = np.where(diff, highs, highs*2)
        #print (diff)
        #print (decision_new_pos)
        #print (decision_new_neg)
        #assert diff.all()
        #assert 0

        #num_tries = 0
        #while (num_tries < 10):  #Can reduce a little
        #    mids = (lows+highs)/2
        #    dists = mids * deltas
        #    num_tries += 1
        #    perturbed_pos = sample + deltas.reshape(deltas_shape) * rv_ortho + dists.reshape(deltas_shape) * rv_redir
        #    perturbed_neg = sample - deltas.reshape(deltas_shape) * rv_ortho - dists.reshape(deltas_shape) * rv_redir
        #    decision_new_pos = decision_function(perturbed_pos)
        #    decision_new_neg = decision_function(perturbed_neg)
        #    print ()
        #    print (mids)
        #    print (decision_new_pos)
        #    print (decision_new_neg)
        #    print (np.logical_xor(decision_new_pos, decision_new_neg))
        #    if (np.logical_xor(decision_new_pos, decision_new_neg).all()):
        #        break
        #    lows = np.where(np.logical_and(decision_new_pos==decision_pos, decision_new_neg==decision_neg), mids, lows)
        #    highs = np.where(np.logical_and(decision_new_pos!=decision_pos, decision_new_neg!=decision_neg), mids, highs)


        v0 = a.forward(sample[None], strict=False)[0][:,a._criterion.target_class()]
        perturbed_pos = sample + 0.01*rv
        vpos = a.forward(perturbed_pos, strict=False)[0][:,a._criterion.target_class()]
        perturbed_neg = sample - 0.01*rv
        vneg = a.forward(perturbed_neg, strict=False)[0][:,a._criterion.target_class()]
        #h = (vpos + vneg - v0*2) / (delta * np.sqrt(np.sum(rv ** 2, axis=axis)))**2
        h = vpos + vneg - v0*2
        hessian = np.mean(h.reshape(decision_shape)*rv, axis=0)
        hessian = hessian / np.linalg.norm(hessian)

        return gradf, hessian, np.mean(fval)

    def historical_gradient(self, decision_function, sample, history, delta, avg_val, norm_v):
        self.history = self.history + history[-1]
        #self.history = 0.8*self.history + history[-1]
        gradf = self.history
        #gradf = history[-1]

        #axis = tuple(range(1, 1 + len(self.shape)))
        #d = np.array(history)
        #d = d / np.sqrt(np.sum(d ** 2, axis=axis, keepdims=True))
        #perturbed = sample + delta * d
        #perturbed = np.clip(perturbed, self.clip_min, self.clip_max)
        #d = (perturbed - sample) / delta

        ## query the model.
        #decisions = decision_function(perturbed)
        #decision_shape = [len(decisions)] + [1] * len(self.shape)
        #fval = 2 * decisions.astype(self.internal_dtype).reshape(
        #    decision_shape) - 1.0
        #print (decisions)

        ### Baseline subtraction (when fval differs)
        #vals = fval - avg_val
        #gradf = np.mean(vals * d, axis=0)

        #Make ortho to dist
        print ("History vs. dist before orthogonize:", cos_sim(gradf, norm_v))
        gradf_proj = ((gradf * norm_v).sum()) / ((norm_v**2).sum()) * norm_v
        gradf = gradf - gradf_proj

        # Get the gradient direction.
        gradf = gradf / np.linalg.norm(gradf)

        return gradf

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
        print ("Initial epsilon:", epsilon)
        while True:
            updated = np.clip(x + epsilon * update, self.clip_min, self.clip_max)
            success = decision_function(updated[None])[0]
            if success:
                break
            else:
                epsilon = epsilon / 2.0  # pragma: no cover
                print ("Geo progress decrease eps at %.4f"%epsilon)

        return epsilon

    def log_step(self, step, distance, message='', always=False, a=None, perturbed=None, update=None):
        assert len(self.logger) == step
        self.logger.append((a._total_prediction_calls, distance))
        if not always and step % self.log_every_n_steps != 0:
            return
        print('Step {}: {:.5e} {}'.format(
            step,
            distance,
            message))
        if a is not None:
            import matplotlib.pyplot as plt
            fig = plt.figure()
            #plt.imshow(perturbed[:,:,::-1]/255)  #keras
            plt.imshow(perturbed.transpose(1,2,0))  #pytorch
            #plt.imshow((perturbed+1)/2)  #receipt
            plt.axis('off')
            plt.title('Call %d'%a._total_prediction_calls)
            fig.savefig('BAPP_result/step%d.png'%step)
            plt.close(fig)
            if update is not None:
                print (np.linalg.norm(update))
                fig = plt.figure()
                abs_update = (update - update.min()) / (update.max() - update.min())
                #plt.imshow(abs_update[:,:,::-1])  #keras
                plt.imshow(abs_update.transpose(1,2,0))  #pytorch
                #plt.imshow(abs_update)  #receipt
                plt.axis('off')
                plt.title('Call %d'%a._total_prediction_calls)
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
            print ("Call:", a._total_prediction_calls, "Saved to", 'BAPP_result/step%d.png'%step)

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
