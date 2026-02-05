#https://github.com/Jianbo-Lab/HSJA/blob/master/hsja.py
from __future__ import absolute_import, division, print_function
import numpy as np
import torch
from torchvision.utils import save_image
def decision_function(model, images, params,query):
    """
    Decision function output 1 on the desired side of the boundary,
    0 otherwise.
    """
    if query+len(images)>params['num_iterations']:
        return None,query
    images = clip_image(images, params['clip_min'], params['clip_max'])
    count = len(images)//100 
    yushu = len(images)%100
    probs=[]
    for i in range(count+1):
        if i==count :
            if  yushu!=0:
                prob = model(torch.tensor(images[i*100:]).float().cuda()).detach().cpu().numpy()
            else:
                break
        else:
            prob = model(torch.tensor(images[i*100:(i+1)*100]).float().cuda()).detach().cpu().numpy()
        probs.append(prob)
    prob = np.concatenate(probs,axis=0)
    query+=len(images)
    if params['target_label'] is None:
        return np.argmax(prob, axis = 1) != params['original_label'],query
    else:
        return np.argmax(prob, axis = 1) == params['target_label'],query

def clip_image(image, clip_min, clip_max):
        # Clip an image, or an image batch, with upper and lower threshold.
    return np.minimum(np.maximum(clip_min, image), clip_max) 


def compute_distance(x_ori, x_pert, constraint = 'l2'):
    # Compute the distance between two images.
    if constraint == 'l2':
        return np.linalg.norm(x_ori - x_pert)
    elif constraint == 'linf':
        return np.max(abs(x_ori - x_pert))


def approximate_gradient(model, sample, num_evals, delta, params,query,tracker=None,blacklight_threshold=25,\
                                 blacklight_count=0,blacklight_first_detect=0):
    clip_max, clip_min = params['clip_max'], params['clip_min']

    # Generate random vectors.
    noise_shape = [num_evals] + list(params['shape'])
    if params['constraint'] == 'l2':
        rv = np.random.randn(*noise_shape)
    elif params['constraint'] == 'linf':
        rv = np.random.uniform(low = -1, high = 1, size = noise_shape)

    rv = rv / np.sqrt(np.sum(rv ** 2, axis = (1,2,3), keepdims = True))
    perturbed = sample + delta * rv
    perturbed = clip_image(perturbed, clip_min, clip_max)
    rv = (perturbed - sample) / delta
    if tracker is not None:
        for imgi in perturbed:
            match_num = tracker.add_img(imgi.detach().cpu().numpy())
            if match_num>blacklight_threshold:
                blacklight_count+=1
                if blacklight_first_detect ==0:
                    blacklight_first_detect = query
                return None,query,blacklight_count,blacklight_first_detect

    # query the model.
    decisions,query = decision_function(model, perturbed, params,query)
    if decisions is None:
        return None,query,blacklight_count,blacklight_first_detect
    decision_shape = [len(decisions)] + [1] * len(params['shape'])
    fval = 2 * decisions.astype(float).reshape(decision_shape) - 1.0

    # Baseline subtraction (when fval differs)
    if np.mean(fval) == 1.0: # label changes. 
        gradf = np.mean(rv, axis = 0)
    elif np.mean(fval) == -1.0: # label not change.
        gradf = - np.mean(rv, axis = 0)
    else:
        fval -= np.mean(fval)
        gradf = np.mean(fval * rv, axis = 0) 

    # Get the gradient direction.
    gradf = gradf / np.linalg.norm(gradf)

    return gradf,query,blacklight_count,blacklight_first_detect


def project(original_image, perturbed_images, alphas, params):
    #alphas_shape = [len(alphas)] + [1] * len(params['shape'])
    alphas_shape = [1] * len(params['shape'])
    alphas = alphas.reshape(alphas_shape)
    if params['constraint'] == 'l2':
        return (1-alphas) * original_image + alphas * perturbed_images
    elif params['constraint'] == 'linf':
        out_images = clip_image(
                perturbed_images, 
                original_image - alphas, 
                original_image + alphas
                )
        return out_images


def binary_search_batch(original_image, perturbed_images, model, params,query,tracker=None,blacklight_threshold=25,
                                 blacklight_count=0,blacklight_first_detect=0):
    """ Binary search to approach the boundar. """

    # Compute distance between each of perturbed image and original image.
    dists_post_update = np.array([
                    compute_distance(
                            original_image, 
                            perturbed_image, 
                            params['constraint']
                    ) 
                    for perturbed_image in perturbed_images])

    # Choose upper thresholds in binary searchs based on constraint.
    if params['constraint'] == 'linf':
        highs = dists_post_update
            # Stopping criteria.
        thresholds = np.minimum(dists_post_update * params['theta'], params['theta'])
    else:
        highs = np.ones(len(perturbed_images))
        thresholds = params['theta']

    lows = np.zeros(len(perturbed_images))

    succ=False
    if len(perturbed_images.shape)==5:
        perturbed_images = perturbed_images[0]
    # Call recursive function. 
    while np.max((highs - lows) / thresholds) > 1:
        # projection to mids.
        mids = (highs + lows) / 2.0
        mid_images = project(original_image, perturbed_images, mids, params)

        # Update highs and lows based on model decisions.
        decisions,query = decision_function(model, mid_images, params,query)
        if tracker is not None:
            match_num = tracker.add_img(mid_images)
            if match_num>blacklight_threshold:
                blacklight_count+=1
                if blacklight_first_detect ==0:
                    blacklight_first_detect = query
                dists = np.array([
                            compute_distance(
                                original_image, 
                                out_image, 
                                params['constraint']
                            ) for out_image in mid_images])

                return None,dists,query,succ,blacklight_count,blacklight_first_detect

        if decisions is None:
                return None,None,query,False,blacklight_count,blacklight_first_detect
        lows = np.where(decisions == 0, mids, lows)
        highs = np.where(decisions == 1, mids, highs)
        if decisions==True:
            dists = np.array([
                        compute_distance(
                            original_image, 
                            out_image, 
                            params['constraint']
                        ) for out_image in mid_images])
            if dists<=params['epsilon']:
                succ=True
                return mid_images,dists,query,succ,blacklight_count,blacklight_first_detect
            
    out_images = project(original_image, perturbed_images, highs, params)

    # Compute distance of the output image to select the best choice. 
    # (only used when stepsize_search is grid_search.)
    dists = np.array([
            compute_distance(
                    original_image, 
                    out_image, 
                    params['constraint']
            ) 
            for out_image in out_images])
    idx = np.argmin(dists)

    dist = dists[idx]
    out_image = np.expand_dims(out_images[idx],0)
    return out_image, dist,query,succ,blacklight_count,blacklight_first_detect


def initialize(model, sample, params,query,tracker=None,blacklight_threshold=25,blacklight_count=0,blacklight_first_detect=0):
    """ 
    Efficient Implementation of BlendedUniformNoiseAttack in Foolbox.
    """
    success = 0
    num_evals = 0

    if params['target_image'] is None:
        # Find a misclassified random noise.
        while True:
            random_noise = np.random.uniform(params['clip_min'], 
                    params['clip_max'], size = params['shape'])
            success,query = decision_function(model,random_noise[None], params,query)
            if tracker is not None:
                match_num = tracker.add_img(random_noise)
                if match_num>blacklight_threshold:
                    blacklight_count+=1
                    if blacklight_first_detect ==0:
                        blacklight_first_detect = query
                    return None,query,blacklight_count,blacklight_first_detect


            if success is None:
                return None,query,blacklight_count,blacklight_first_detect
            num_evals += 1
            if success:
                    break
            assert num_evals < 1e5,"Initialization failed! "
            "Use a misclassified image as `target_image`" 


        # Binary search to minimize l2 distance to original image.
        low = 0.0
        high = 1.0
        while high - low > 0.001:
            mid = (high + low) / 2.0
            blended = (1 - mid) * sample + mid * random_noise 
            success,query = decision_function(model, blended[None], params,query)
            if tracker is not None:
                match_num = tracker.add_img(blended)
                if match_num>blacklight_threshold:
                    blacklight_count+=1
                    if blacklight_first_detect ==0:
                        blacklight_first_detect = query
                    return None,query,blacklight_count,blacklight_first_detect


            if success is None:
                if high<=params['epsilon']:
                    blended = (1 - high) * sample + high * random_noise
                    return blended,query,blacklight_count,blacklight_first_detect
                return None,query,blacklight_count,blacklight_first_detect
            if success:
                    high = mid
            else:
                    low = mid

        initialization = (1 - high) * sample + high * random_noise 

    else:
        initialization = params['target_image']

    return initialization,query,blacklight_count,blacklight_first_detect


def geometric_progression_for_stepsize(x, update, dist, model, params,query,tracker=None,blacklight_threshold=25,\
                                 blacklight_count=0,blacklight_first_detect=0):
    """
    Geometric progression to search for stepsize.
    Keep decreasing stepsize by half until reaching 
    the desired side of the boundary,
    """
    epsilon = dist / np.sqrt(params['cur_iter']) 

    def phi(epsilon,query,blacklight_count=0,blacklight_first_detect=0):
        new = x + epsilon * update
        success,query = decision_function(model, new, params,query)
        if tracker is not None:
            match_num = tracker.add_img(new[0])
            if match_num>blacklight_threshold:
                blacklight_count+=1
                if blacklight_first_detect ==0:
                    blacklight_first_detect = query
                return None,query,blacklight_count,blacklight_first_detect


        return success,query,blacklight_count,blacklight_first_detect
    succ,query,blacklight_count,blacklight_first_detect = phi(epsilon,query,blacklight_count,blacklight_first_detect)
    while succ==False and succ is not None:
        epsilon /= 2.0
        succ,query,blacklight_count,blacklight_first_detect  = phi(epsilon,query,blacklight_count,blacklight_first_detect)
    if succ is None:
        return None,query,blacklight_count,blacklight_first_detect
    return epsilon,query,blacklight_count,blacklight_first_detect

def select_delta(params, dist_post_update):
    """ 
    Choose the delta at the scale of distance 
    between x and perturbed sample. 

    """
    if params['cur_iter'] == 1:
        delta = 0.1 * (params['clip_max'] - params['clip_min'])
    else:
        if params['constraint'] == 'l2':
            delta = np.sqrt(params['d']) * params['theta'] * dist_post_update
        elif params['constraint'] == 'linf':
            delta = params['d'] * params['theta'] * dist_post_update        

    return delta
        

def hsja(model, 
         epsilon,
        sample, 
        clip_max = 1, 
        clip_min = 0, 
        constraint = 'l2', 
        num_iterations = 40, 
        gamma = 1.0, 
        target_label = None, 
        target_image = None, 
        stepsize_search = 'geometric_progression', 
        max_num_evals = 1e4,
        init_num_evals = 100,
        verbose = True,test=False,tracker=None,
        blacklight_threshold=25,
        blacklight_count=0,blacklight_first_detect=0):
    """
    Main algorithm for HopSkipJumpAttack.

    Inputs:
    model: the object that has predict method. 

    predict outputs probability scores.

    clip_max: upper bound of the image.

    clip_min: lower bound of the image.

    constraint: choose between [l2, linf].

    num_iterations: number of iterations.

    gamma: used to set binary search threshold theta. The binary search 
    threshold theta is gamma / d^{3/2} for l2 attack and gamma / d^2 for 
    linf attack.

    target_label: integer or None for nontargeted attack.

    target_image: an array with the same size as sample, or None. 

    stepsize_search: choose between 'geometric_progression', 'grid_search'.

    max_num_evals: maximum number of evaluations for estimating gradient (for each iteration). 
    This is not the total number of model evaluations for the entire algorithm, you need to 
    set a counter of model evaluations by yourself to get that. To increase the total number 
    of model evaluations, set a larger num_iterations. 

    init_num_evals: initial number of evaluations for estimating gradient.

    Output:
    perturbed image.
    
    """
    # Set parameters
    original_label = np.argmax(model(torch.tensor(sample).unsqueeze(0).float().cuda()).detach().cpu().numpy())
    params = {'clip_max': clip_max, 'clip_min': clip_min, 
                            'shape': sample.shape,
                            'original_label': original_label, 
                            'target_label': target_label,
                            'target_image': target_image, 
                            'constraint': constraint,
                            'num_iterations': num_iterations, 
                            'gamma': gamma, 
                            'd': int(np.prod(sample.shape)), 
                            'stepsize_search': stepsize_search,
                            'max_num_evals': max_num_evals,
                            'init_num_evals': init_num_evals,
                            'verbose': verbose,
                            'epsilon':epsilon
                            }
    succ=False
    # Set binary search threshold.
    if params['constraint'] == 'l2':
        params['theta'] = params['gamma'] / (np.sqrt(params['d']) * params['d'])
    else:
        params['theta'] = params['gamma'] / (params['d'] ** 2)
    query =0 
    # Initialize.
    perturbed,query,blacklight_count,blacklight_first_detect  = initialize(model, sample, params,query,tracker=tracker,blacklight_threshold=blacklight_threshold,\
                                 blacklight_count=blacklight_count,blacklight_first_detect=blacklight_first_detect)
    if test == True:
        return torch.tensor(perturbed).float().cuda(),True,0

    if perturbed is None:
        return None,False,query,blacklight_count,blacklight_first_detect
    # Project the initialization to the boundary.
    perturbed, dist_post_update,query,succ,blacklight_count,blacklight_first_detect = binary_search_batch(sample, \
            np.expand_dims(perturbed, 0), \
            model, \
            params,query,tracker=tracker,blacklight_threshold=blacklight_threshold,\
                                 blacklight_count=blacklight_count,blacklight_first_detect=blacklight_first_detect)
    if succ == True:
        return perturbed,succ,query,blacklight_count,blacklight_first_detect
    if perturbed is None:
        return None,False,query,blacklight_count,blacklight_first_detect
    dist = compute_distance(perturbed, sample, constraint)

    for j in np.arange(params['num_iterations']):
        params['cur_iter'] = j + 1

        # Choose delta.
        delta = select_delta(params, dist_post_update)

        # Choose number of evaluations.
        num_evals = int(params['init_num_evals'] * np.sqrt(j+1))
        num_evals = int(min([num_evals, params['max_num_evals']]))

        # approximate gradient.
        gradf,query,blacklight_count,blacklight_first_detect = approximate_gradient(model, perturbed, num_evals, 
                delta, params,query,tracker=tracker,blacklight_threshold=blacklight_threshold,\
                                 blacklight_count=blacklight_count,blacklight_first_detect=blacklight_first_detect)
        if gradf is None:
            return  None,False,query,blacklight_count,blacklight_first_detect 
        if params['constraint'] == 'linf':
            update = np.sign(gradf)
        else:
            update = gradf

        # search step size.
        if params['stepsize_search'] == 'geometric_progression':
            # find step size.
            epsilon,query,blacklight_count,blacklight_first_detect  = geometric_progression_for_stepsize(perturbed, 
                    update, dist, model, params,query,tracker=tracker,blacklight_threshold=blacklight_threshold,\
                                 blacklight_count=blacklight_count,blacklight_first_detect=blacklight_first_detect)
            if epsilon is None :
                return None,False,query,blacklight_count,blacklight_first_detect 
            # Update the sample. 
            perturbed = clip_image(perturbed + epsilon * update, 
                    clip_min, clip_max)

            # Binary search to return to the boundary. 
            perturbed, dist_post_update ,query,succ,blacklight_count,blacklight_first_detect = binary_search_batch(sample, 
                    perturbed, model, params,query,tracker=tracker,blacklight_threshold=blacklight_threshold,\
                                 blacklight_count=blacklight_count,blacklight_first_detect=blacklight_first_detect)
            if succ == True:
                return perturbed,succ,query,blacklight_count,blacklight_first_detect 
            if dist_post_update is None:
                return None,False,query,blacklight_count,blacklight_first_detect 

        elif params['stepsize_search'] == 'grid_search':
            # Grid search for stepsize.
            epsilons = np.logspace(-4, 0, num=20, endpoint = True) * dist
            epsilons_shape = [20] + len(params['shape']) * [1]
            perturbeds = perturbed + epsilons.reshape(epsilons_shape) * update
            perturbeds = clip_image(perturbeds, params['clip_min'], params['clip_max'])
            idx_perturbed,query = decision_function(model, perturbeds, params,query)

            if np.sum(idx_perturbed) > 0:
                    # Select the perturbation that yields the minimum distance # after binary search.
                perturbed, dist_post_update,query,succ,blacklight_count,blacklight_first_detect = binary_search_batch(sample, 
                        perturbeds[idx_perturbed], model, params,query,tracker=tracker,blacklight_threshold=blacklight_threshold,\
                            blacklight_count=blacklight_count,blacklight_first_detect=blacklight_first_detect)
        # compute new distance.
        dist = compute_distance(perturbed, sample, constraint)
        if verbose:
            print('iteration: {:d}, {:s} distance {:.4E}'.format(j+1, constraint, dist))

    return perturbed,succ,query,blacklight_count,blacklight_first_detect 
