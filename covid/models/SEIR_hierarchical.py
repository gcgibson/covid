import jax
import jax.numpy as np

import numpyro
import numpyro.distributions as dist

from functools import partial
from ..compartment import SEIRDModel
from ..glm import glm, GLM, log_link, logit_link, Gamma, Beta

from .util import observe, ExponentialRandomWalk, get_future_data


"""
************************************************************
SEIRD hierarchical
************************************************************
"""


def SEIR_dynamics_hierarchical(T, params, x0, obs = None,  death=None, use_rw = True, suffix=""):
    '''Run SEIR dynamics for T time steps
    
    Uses SEIRModel.run to run dynamics with pre-determined parameters.
    '''
        
    beta, sigma, gamma, det_rate, det_noise_scale, rw_loc, rw_scale, drift, death_prob,death_rate,det_prob_d = params

    num_places, T_minus_1 = beta.shape
    assert(T_minus_1 == T-1)
    
    # prep for broadcasting over time
    sigma = sigma[:,None]
    gamma = gamma[:,None]
    det_rate = det_rate[:,None]
    death_prob = death_prob[:,None]
    death_rate = death_rate[:,None]
    det_prob_d = det_prob_d[:,None]

    if use_rw:
        with numpyro.plate("places", num_places):
            rw = numpyro.sample("rw" + suffix,
                                ExponentialRandomWalk(loc = rw_loc,
                                                      scale = rw_scale,
                                                      drift = drift, 
                                                      num_steps = T-1))
    else:
        rw = rw_loc

    beta *= rw

    # Run ODE
    apply_model = lambda x0, beta, sigma, gamma, death_prob,death_rate: SEIRDModel.run(T, x0, (beta, sigma, gamma,death_prob,death_rate))
    x = jax.vmap(apply_model)(x0, beta, sigma, gamma, death_prob,death_rate)

    x = x[:,1:,:] # drop first time step from result (duplicates initial value)
    numpyro.deterministic("x" + suffix, x)
   
    # Noisy observations
    y = observe("y" + suffix, x[:,:,6], det_rate, det_noise_scale, obs = obs)
    z = observe("z" + suffix, x[:,:,5], det_prob_d, det_noise_scale, obs = death)


    return rw, x, y, z


def SEIR_hierarchical(data = None,
                      place_data = None,
                      T_future = 0,
                      E_duration_est = 4.5,
                      I_duration_est = 3.0,
                      R0_est = 4.5,
                      det_rate_est = 0.3,
                      det_rate_conc = 100,
                      use_rw = True,
                      rw_scale = 1e-1,
                      det_noise_scale = 0.2,
                      drift_scale = None,
                      use_obs = False):

    '''
    Stochastic SEIR model. Draws random parameters and runs dynamics.
    '''
    
    num_places, _ = place_data.shape
    
    '''Generate R0'''
    ## TODO: forecasting with splines not yet supported b/c patsy will not evaluate
    ## splines outside of the outermost knots. Look into workaround/fix for this
    R0_glm = GLM("1 + C(state, OneHot)  + shelter_in_place +standardize(popdensity) ", 
                 data, 
                 log_link,
                 partial(Gamma, var=0.1),
                 prior = dist.Normal(0, 0.1),
                 guess=R0_est,
                 name="R0")
    
    R0 = R0_glm.sample(shape=(num_places,-1))[0]
    
    '''Generate E_duration'''
    E_duration = glm('1 + C(state, OneHot)', 
                     place_data,
                     log_link,
                     partial(Gamma, var=0.05),
                     prior = dist.Normal(0, 0.05),
                     guess=E_duration_est,
                     name="E_duration")[0]
    
    '''Generate I_duration'''
    I_duration = glm('1 + C(state, OneHot)', 
                     place_data,
                     log_link,
                     partial(Gamma, var=0.05),
                     prior=dist.Normal(0, 0.05),
                     guess=I_duration_est,
                     name="I_duration")[0]


    '''Generate det_rate'''
    det_rate = glm('1 + C(state, OneHot)',
                   place_data,
                   logit_link,
                   partial(Beta, conc=det_rate_conc),
                   prior=dist.Normal(0, 0.025),
                   guess=det_rate_est,
                   name="det_rate")[0]
    
    # Broadcast to correct size
    N = np.array(place_data['totalpop'])
    
    R0 = R0.reshape((num_places, -1))
    _, T = R0.shape
    
    det_rate = det_rate
    sigma = 1/E_duration
    gamma = 1/I_duration
    beta0 = R0 * gamma[:,None]
    print (beta0.shape)
    #with numpyro.plate("places", num_places):
    betas1 = numpyro.sample("beta1" ,
                      ExponentialRandomWalk(loc=beta0[0,:], scale=1e-2, drift=0, num_steps=T))
    betas2 = numpyro.sample("beta2" ,
                      ExponentialRandomWalk(loc=beta0[1,:], scale=1e-2, drift=0, num_steps=T))
    beta = np.concatenate((betas1,betas2),axis=0).reshape(num_places,-1)
    print (beta.shape)
    #beta = beta[:,:-1] # truncate to T-1 timesteps (transitions)
    
    with numpyro.plate("num_places", num_places): 
        I0 = numpyro.sample("I0", dist.Uniform(0, 0.02*N))
        E0 = numpyro.sample("E0", dist.Uniform(0, 0.02*N))
        H0 = numpyro.sample("H0", dist.Uniform(0, 0.02*N))
        D0 = numpyro.sample("D0", dist.Uniform(0, 0.001*N))
        death_prob = numpyro.sample("death_prob", 
                                    dist.Beta(.1 * 100,
                                              (1-.1) * 100))

        death_rate = numpyro.sample("death_rate", 
                                    dist.Gamma(10, 10 * 10))
        
        det_prob_d = numpyro.sample("det_prob_d", 
                                    dist.Beta(.9 * 100,
                                              (1-.9) * 100))

    if use_obs:
        pos = np.array(data['positive']).reshape(num_places, T)
        death = np.array(data['death']).reshape(num_places, T)

        obs0, obs = pos[:,0], pos[:,1:]
        death0, death = death[:,0], death[:,1:]
    else:
        obs0, obs = None, None
        death0, death = None, None
    
    '''
    Run model for each place
    '''
    x0 = jax.vmap(SEIRDModel.seed)(N, I0, E0, H0, D0)
    numpyro.deterministic("x0", x0)
    
    # Split observations into first and rest
    
    # First observation
    y0 = observe("y0", x0[:,6], det_rate, det_noise_scale, obs=obs0)
    z0 = observe("z0", x0[:,5], det_rate, det_noise_scale, obs=death0)

    # Run dynamics
    drift = 0.
    rw_loc = 1.
    params = (beta[:,:-1], sigma, gamma, det_rate, det_noise_scale, rw_loc, rw_scale, drift,
             death_prob,death_rate,det_prob_d)
    rw, x, y, z = SEIR_dynamics_hierarchical(T, params, x0, 
                                          use_rw = use_rw, 
                                          obs = obs,
                                            death=death)
    
    x = np.concatenate((x0[:,None,:], x), axis=1)
    z = np.concatenate((z0[:,None], z), axis=1)

    y = np.concatenate((y0[:,None], y), axis=1)
    
    if T_future > 0:
        
        future_data = get_future_data(data, T_future-1)
        
        R0_future = R0_glm.sample(future_data, name="R0_future", shape=(num_places,-1))[0]

        beta_future = R0_future * gamma[:, None]
        beta_future = np.concatenate((beta[:,-1,None], beta_future), axis=1)
        
        rw_loc = rw[:,-1,None] if use_rw else 1.
        
        params = (beta_future, sigma, gamma, det_rate, det_noise_scale, rw_loc, rw_scale, drift,
                 death_prob,death_rate,det_prob_d)
        
        _, x_f, y_f, z_f = SEIR_dynamics_hierarchical(T_future+1, 
                                                 params, 
                                                 x[:,-1,:], 
                                                 use_rw = use_rw,
                                                 suffix="_future")
        
        x = np.concatenate((x, x_f), axis=1)
        y = np.concatenate((y, y_f), axis=1)
        z = np.concatenate((z, z_f), axis=1)

        
    return beta, x, y, z,  det_rate