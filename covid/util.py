import sys

from . import jhu
from . import covidtracking
from . import states

import covid.models.SEIRD

import pandas as pd
import matplotlib.pyplot as plt

import numpy as onp

import jax
import jax.numpy as np
from jax.random import PRNGKey

import numpyro
from numpyro.infer import MCMC, NUTS, Predictive

from pathlib import Path

import cachetools


"""
************************************************************
Data
************************************************************
"""

@cachetools.func.ttl_cache(ttl=600)
def load_world_data():
    # world data
    world = jhu.load_world()
    world = world.loc[:,(slice(None), 'tot', slice(None))] # only country totals

    country_names = world.columns.unique(level=0)
    world_pop_data = pd.read_csv('https://s3.amazonaws.com/rawstore.datahub.io/630580e802a621887384f99527b68f59.csv')
    world_pop_data = world_pop_data.set_index("Country")
        
    country_names_valid = set(country_names) & set(world_pop_data.index) 
    world_data = {
        k: {'data' : world[k].tot, 
            'pop' : world_pop_data.loc[k]['Year_2016'],
            'name' : k}
        for k in country_names_valid
    }

    
    return world_data


def load_state_data(source="jhu"):

    # US state data
    if source=="covidtracker":
        US = covidtracking.load_us()
    if source=="jhu":
        US = jhu.load_us()
    
    traits = states.uga_traits()

    state_set = set(traits.index) & set(US.columns.unique(level=0))

    state_data = {
        k : {'data': US[k], 
             'pop': traits.totalpop[k],
             'name': traits.NAME[k]
            }
        for k in state_set
    }
    
    return state_data


def load_data(us_source="jhu"):
    state_data = load_state_data(source=us_source)
    world_data = load_world_data()
    return dict(world_data, **state_data)


def load_state_Xy(which=None):
    X_place = states.uga_traits().drop('DC') # incomplete data for DC
    X = states.uga_interventions()
    y = covidtracking.load_us_flat()
    
    Xy = y.join(X, how='inner').sort_index()
    
    # Remove dates without enough data
    date = Xy.index.get_level_values(1)
    counts = Xy.groupby(date).apply(lambda x: len(x))
    good_dates = counts.index[counts == counts.max()]
    Xy = Xy.loc[date.isin(good_dates)]
        
    # Add integer time column
    start = Xy.index.unique(level=1).min()
    Xy['t'] = (Xy['date']-start)/pd.Timedelta("1d")
            
    # Select requested states
    if which is not None:
        Xy = Xy.loc[which,:]
        X_place = X_place.loc[which,:]
        
    return Xy, X_place



"""
************************************************************
Plotting
************************************************************
"""
    
def plot_R0(mcmc_samples, start):

    fig = plt.figure(figsize=(5,3))
    
    # Compute average R0 over time
    if 'gamma' in mcmc_samples.keys():
        gamma = mcmc_samples['gamma'][:,None]
    else:
        gamma = .5
    beta = mcmc_samples['beta']
    t = pd.date_range(start=start, periods=beta.shape[1], freq='D')
    R0 = beta/gamma

    pi = np.percentile(R0, (10, 90), axis=0)
    df = pd.DataFrame(index=t, data={'R0': np.median(R0, axis=0)})
    df.plot(style='-o')
    plt.fill_between(t, pi[0,:], pi[1,:], alpha=0.1)

    plt.axhline(1, linestyle='--')
    
    #plt.tight_layout()

    return fig





"""
************************************************************
Running
************************************************************
"""


def run_place(data, 
              place, 
              model_type=covid.models.SEIRD.SEIRD,
              start = '2020-03-04',
              end = None,
              save = True,
              num_warmup = 1000,
              num_samples = 1000,
              num_chains = 1,
              num_prior_samples = 0,              
              T_future=4*7,
              prefix = "results",
              **kwargs):


    print(f"Running {place} (start={start}, end={end})")
    
    place_data = data[place]['data'][start:end]
    T = len(place_data)

    model = model_type(
        data = place_data,
        T = T,
        N = data[place]['pop'],
        **kwargs
    )
    
    print(" * running MCMC")
    mcmc_samples = model.infer(num_warmup=num_warmup, 
                               num_samples=num_samples)

    # Prior samples
    prior_samples = None
    if num_prior_samples > 0:
        print(" * collecting prior samples")
        prior_samples = model.prior(num_samples=num_prior_samples)

    # In-sample posterior predictive samples (don't condition on observations)
    print(" * collecting predictive samples")
    post_pred_samples = model.predictive()

    # Forecasting posterior predictive (do condition on observations)
    print(" * collecting forecast samples")
    forecast_samples = model.forecast(T_future=T_future)
        
    if save:

        # Save samples
        path = Path(prefix) / 'samples'
        path.mkdir(parents=True, exist_ok=True)
        filename = path / f'{place}.npz'
        
        save_samples(filename,
                     prior_samples,
                     mcmc_samples, 
                     post_pred_samples,
                     forecast_samples)
        
        path = Path(prefix) / 'summary'
        path.mkdir(parents=True, exist_ok=True)
        filename = path / f'{place}.txt'
        
        write_summary(filename, model.mcmc)

        
def save_samples(filename, 
                 prior_samples,
                 mcmc_samples, 
                 post_pred_samples,
                 forecast_samples):
    
    np.savez(filename, 
             prior_samples = prior_samples,
             mcmc_samples = mcmc_samples, 
             post_pred_samples = post_pred_samples,
             forecast_samples = forecast_samples)


def write_summary(filename, mcmc):
    # Write diagnostics to file
    orig_stdout = sys.stdout
    with open(filename, 'w') as f:
        sys.stdout = f
        mcmc.print_summary()
    sys.stdout = orig_stdout

    
def load_samples(filename):

    x = np.load(filename, allow_pickle=True)
    
    prior_samples = x['prior_samples'].item()
    mcmc_samples = x['mcmc_samples'].item()
    post_pred_samples = x['post_pred_samples'].item()
    forecast_samples = x['forecast_samples'].item()
    
    return prior_samples, mcmc_samples, post_pred_samples, forecast_samples


def gen_forecasts(data, 
                  place, 
                  model_type=covid.models.SEIRD.SEIRD,
                  start = '2020-03-04', 
                  end=None,
                  save = True,
                  show = True, 
                  prefix='results',
                  **kwargs):
    

    # Deal with paths
    samples_path = Path(prefix) / 'samples'
    vis_path = Path(prefix) / 'vis'
    vis_path.mkdir(parents=True, exist_ok=True)
    
    model = model_type()

    confirmed = data[place]['data'].confirmed[start:end]
    death = data[place]['data'].death[start:end]

    T = len(confirmed)
    N = data[place]['pop']

    filename = samples_path / f'{place}.npz'   
    _, mcmc_samples, post_pred_samples, forecast_samples = load_samples(filename)
        
    for daily in [False, True]:
        for scale in ['log', 'lin']:
            for T in [14, 28]:

                fig, axes = plt.subplots(nrows = 2, figsize=(8,12), sharex=True)    

                if daily:
                    variables = ['dy', 'dz']
                    observations = [confirmed.diff(), death.diff()]
                else:
                    variables = ['y', 'z']
                    observations= [confirmed, death]

                for variable, obs, ax in zip(variables, observations, axes):
                    model.plot_forecast(variable,
                                        post_pred_samples,
                                        forecast_samples,
                                        start,
                                        T_future=T,
                                        obs=obs,
                                        ax=ax,
                                        scale=scale)

                name = data[place]['name']
                plt.suptitle(f'{name} {T} days ')
                plt.tight_layout()

                if save:
                    filename = vis_path / f'{place}_scale_{scale}_daily_{daily}_T_{T}.png'
                    plt.savefig(filename)

                if show:
                    plt.show()
            
    fig = plot_R0(mcmc_samples, start)    
    plt.title(place)
    plt.tight_layout()
    
    if save:
        filename = vis_path / f'{place}_R0.png'
        plt.savefig(filename)

    if show:
        plt.show()   
        
        
        
def score_forecasts(start,
                    place,
                    data,
                    prefix="results",
                    model_type=covid.models.SEIRD.SEIRD,
                    eval_date=None,
                    method="mae"):

    model = model_type()
    
    filename = Path(prefix) / 'samples' / f'{place}.npz'
    prior_samples, mcmc_samples, post_pred_samples, forecast_samples = \
        load_samples(filename)

    # cumulative deaths 
    death = data[place]['data'][start:eval_date].death
    end = death.index.max()

    obs = death[start:]

    T = len(obs)
    z = model.get(forecast_samples, 'z', forecast=True)[:,:T]
    df = pd.DataFrame(index=obs.index, data=z.T)
 

    if method == "ls":
         
          ls = []

          for index, row in df.iterrows():
               ls.append(sum((round(df.loc[index,:]) < (round(obs.loc[index]) + 100) )&
          (round(df.loc[index,:]) > (round(obs.loc[index]) - 100))))

          err = np.mean(np.clip(np.log(np.array(ls)),-10))
          #err_plot = err.plot(style='0')     
    else:
         point_forecast = df.median(axis=1)
         
         errs = (obs-point_forecast).rename('errs')

         err = errs[eval_date] if eval_date is not None else errs.values[-1]

         #err_plot = err.plot(style='o')
         #err = err.abs().mean()

    return err
