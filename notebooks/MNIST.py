from importlib import reload
import abstention
reload(abstention)
reload(abstention.calibration)
reload(abstention.label_shift)
reload(abstention.figure_making_utils)
from abstention.calibration import (
    TempScaling, VectorScaling, NoBiasVectorScaling, softmax)
from abstention.label_shift import (EMImbalanceAdapter,
     BBSEImbalanceAdapter,
     RLLSImbalanceAdapter,
     ShiftWeightFromImbalanceAdapter)
import glob
import gzip
import numpy as np
from collections import defaultdict, OrderedDict
from histnetlabelshift import HistNetAdapter
import cifarandmnist
from keras.datasets import mnist
from keras import backend as K

import warnings
warnings.filterwarnings("ignore")
#para quitar los warnings del cvxpy


np.set_printoptions(suppress=True)

        
def read_preds(fh):
    return np.array([[float(x) for x in y.rstrip().split("\t")]
                     for y in fh])

test_labels = read_preds(open("test_labels.txt"))
valid_labels = read_preds(open("valid_labels.txt"))


#Load test data
(_, _), (x_test, y_test) = mnist.load_data()
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
x_test = x_test.astype('float32')
x_test /= 255

imbalanceadaptername_to_imbalanceadapter = {
    'histnet': HistNetAdapter(model_path="/media/nas/pgonzalez/labelshiftexperiments/notebooks/obtaining_predictions/mnist/model_quant_mnist_mse_set-30000_mse_seed-0.h5"),
    'em': EMImbalanceAdapter(),
    'bbse-hard': BBSEImbalanceAdapter(soft=False),
    'bbse-soft': BBSEImbalanceAdapter(soft=True),
    'rlls-hard': RLLSImbalanceAdapter(soft=False),
    'rlls-soft': RLLSImbalanceAdapter(soft=True),}

calibname_to_calibfactory = OrderedDict([
    ('None', abstention.calibration.Softmax()),
    ('TS', TempScaling(verbose=False)),
    ('NBVS', NoBiasVectorScaling(verbose=False)),
    ('BCTS', TempScaling(verbose=False,
                         bias_positions='all')),
    ('VS', VectorScaling(verbose=False))
])

adaptncalib_pairs = [
    ('histnet', 'None'),
    ('bbse-hard', 'None'),
    ('bbse-soft', 'None'),
    ('bbse-soft', 'TS'),
    ('bbse-soft', 'NBVS'),
    ('bbse-soft', 'BCTS'),
    ('bbse-soft', 'VS'),
    ('bbse-soft', 'best-ece'),
    ('bbse-soft', 'best-jsdiv'),
    ('bbse-soft', 'best-nll'),

    ('rlls-hard', 'None'),
    ('rlls-soft', 'None'),
    ('rlls-soft', 'TS'),
    ('rlls-soft', 'NBVS'),
    ('rlls-soft', 'BCTS'),
    ('rlls-soft', 'VS'),
    ('rlls-soft', 'best-ece'),
    ('rlls-soft', 'best-jsdiv'),
    ('rlls-soft', 'best-nll'),

    ('em', 'None'),
    ('em', 'TS'),
    ('em', 'NBVS'),
    ('em', 'BCTS'),
    ('em', 'VS'),
    ('em', 'best-ece'),
    ('em', 'best-jsdiv'),
    ('em', 'best-nll')
]

num_trials = 10
seeds = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
dirichlet_alphas_and_samplesize = [
    (0.1,2000), (1.0,2000), (5,2000), (10,2000),
    (0.1,4000), (1.0,4000), (5,4000), (10,4000),
    (0.1,8000), (1.0,8000), (5,8000), (10,8000)
]
tweakone_alphas_and_samplesize = [
    (0.01,2000), (0.9,2000),
    (0.01,4000), (0.9,4000),
    (0.01,8000), (0.9,8000)
]

print("Dirichlet shift")

(dirichlet_alpha_to_samplesize_to_adaptername_to_metric_to_vals,      
 dirichlet_alpha_to_samplesize_to_baselineacc,                        
 metric_to_samplesize_to_calibname_to_unshiftedvals) =\
  cifarandmnist.run_experiments(
    num_trials=num_trials,
    seeds=seeds,
    alphas_and_samplesize = dirichlet_alphas_and_samplesize,
    shifttype='dirichlet',
    calibname_to_calibfactory=calibname_to_calibfactory,
    imbalanceadaptername_to_imbalanceadapter=
      imbalanceadaptername_to_imbalanceadapter,
    adaptncalib_pairs=adaptncalib_pairs,
    validglobprefix="validpreacts_model_mnist_set-30000_seed-",
    testglobprefix="testpreacts_model_mnist_set-30000_seed-",
    valid_labels=valid_labels,
    test_labels=test_labels,
    x_test=x_test) 

print("Tweak one shift")

(tweakone_alpha_to_samplesize_to_adaptername_to_metric_to_vals,      
 tweakone_alpha_to_samplesize_to_baselineacc,                        
 _) = cifarandmnist.run_experiments(
    num_trials=num_trials,
    seeds=seeds,
    alphas_and_samplesize = tweakone_alphas_and_samplesize,
    shifttype='tweakone',
    calibname_to_calibfactory=calibname_to_calibfactory,
    imbalanceadaptername_to_imbalanceadapter=
      imbalanceadaptername_to_imbalanceadapter,
    adaptncalib_pairs=adaptncalib_pairs,
    validglobprefix="validpreacts_model_mnist_set-30000_seed-",
    testglobprefix="testpreacts_model_mnist_set-30000_seed-",
    valid_labels=valid_labels,
    test_labels=test_labels,
    x_test=x_test)


import json
import os
file_out = "mnist_label_shift_adaptation_results.json"
dict_to_write = {
    "dirichlet_alpha_to_samplesize_to_adaptername_to_metric_to_vals":
     dirichlet_alpha_to_samplesize_to_adaptername_to_metric_to_vals,
    "dirichlet_alpha_to_samplesize_to_baselineacc":
     dirichlet_alpha_to_samplesize_to_baselineacc,
    "metric_to_samplesize_to_calibname_to_unshiftedvals":
     metric_to_samplesize_to_calibname_to_unshiftedvals,
    "tweakone_alpha_to_samplesize_to_adaptername_to_metric_to_vals":
     tweakone_alpha_to_samplesize_to_adaptername_to_metric_to_vals,
    "tweakone_alpha_to_samplesize_to_baselineacc":
     tweakone_alpha_to_samplesize_to_baselineacc
}
open(file_out, 'w').write(
    json.dumps(dict_to_write,
               sort_keys=True, indent=4, separators=(',', ': ')))
os.system("gzip -f "+file_out)

import gzip
import json
loaded_dicts = json.loads(gzip.open("mnist_label_shift_adaptation_results.json.gz").read())
dirichlet_alpha_to_samplesize_to_adaptername_to_metric_to_vals =\
    loaded_dicts['dirichlet_alpha_to_samplesize_to_adaptername_to_metric_to_vals']
dirichlet_alpha_to_samplesize_to_baselineacc =\
    loaded_dicts['dirichlet_alpha_to_samplesize_to_baselineacc']
tweakone_alpha_to_samplesize_to_adaptername_to_metric_to_vals =\
    loaded_dicts['tweakone_alpha_to_samplesize_to_adaptername_to_metric_to_vals']
tweakone_alpha_to_samplesize_to_baselineacc =\
    loaded_dicts['tweakone_alpha_to_samplesize_to_baselineacc']
metric_to_samplesize_to_calibname_to_unshiftedvals =\
    loaded_dicts['metric_to_samplesize_to_calibname_to_unshiftedvals']

from importlib import reload
import abstention
import numpy as np

from maketable import render_calibration_table

metricname_to_nicename = {'nll': 'NLL', 'jsdiv': 'jsdiv', 'ece': 'ECE'}
calibname_to_nicename = {'None': "None", "TS": "TS",
                         "VS":"VS", "NBVS": "NBVS", "BCTS": "BCTS"}
  
from scipy.stats import norm
N = len(dirichlet_alpha_to_samplesize_to_adaptername_to_metric_to_vals['1.0']['2000']['em:TS']['jsdiv'])
#Using the normal approximation at N=100;
# variance from https://en.wikipedia.org/wiki/Wilcoxon_signed-rank_test
#Note that T = ((N+1)*N/2 - W)/2            
ustat_threshold = ((N*(N+1))/2 - norm.ppf(0.99)*np.sqrt(N*(N+1)*(2*N+1)/6.0))/2.0
  
print(render_calibration_table(
        metric_to_samplesize_to_calibname_to_unshiftedvals=
          metric_to_samplesize_to_calibname_to_unshiftedvals,
        #threshold of 8 comes from table https://www.oreilly.com/library/view/nonparametric-statistics-a/9781118840429/bapp02.xhtml
        #for one-tailed alpha=0.025 and n=10            
        ustat_threshold=ustat_threshold,
        metrics_in_table=['nll', 'ece'],
        samplesizes_in_table=['2000', '4000', '8000'],
        calibnames_in_table=['None', 'TS', 'NBVS', 'BCTS', 'VS'],
        metricname_to_nicename=metricname_to_nicename,
        calibname_to_nicename=calibname_to_nicename,
        caption="MNIST Calibration metric differences", label="MNIST_calibrationcomparison",
        applyunderline=False))


from maketable import render_adaptation_table
from collections import OrderedDict
    
samplesizes_in_table = ['2000', '4000', '8000']
adaptname_to_nicename = {'histnet':'HistNet',
                         'em': 'EM',
                         'bbse-soft': 'BBSL-soft',
                         'bbse-hard': 'BBSL-hard',
                         'rlls-soft': 'RLLS-soft',
                         'rlls-hard': 'RLLS-hard'}
calibname_to_nicename = {'None': 'None',
                           'TS': 'TS',
                           'NBVS': 'NBVS',
                           'BCTS': 'BCTS',
                           'VS': 'VS',
                           'best-nll':'Best NLL',
                           'best-jsdiv':'Best JS Div',
                           'best-ece':'Best ECE'}

dirichlet_alphas_in_table = ['0.1', '1.0', '10']
tweakone_alphas_in_table = ['0.01', '0.9']

methodgroups_all = OrderedDict([('all', [
                      'histnet:None','em:None', 'em:TS', 'em:NBVS', 'em:BCTS', 'em:VS',
                      'bbse-hard:None', 'bbse-soft:None', 'bbse-soft:TS', 'bbse-soft:NBVS', 'bbse-soft:BCTS', 'bbse-soft:VS',
                      'rlls-hard:None', 'rlls-soft:None', 'rlls-soft:TS', 'rlls-soft:NBVS', 'rlls-soft:BCTS', 'rlls-soft:VS'
                      ])])
methodgroups_all_fordeltacc = OrderedDict([('all', [
                      'histnet:None','em:None', 'em:TS', 'em:NBVS', 'em:BCTS', 'em:VS',
                      'bbse-soft:TS', 'bbse-soft:NBVS', 'bbse-soft:BCTS', 'bbse-soft:VS',
                      'rlls-soft:TS', 'rlls-soft:NBVS', 'rlls-soft:BCTS', 'rlls-soft:VS'
                      ])])

methodgroups_em = OrderedDict([('em', ['em:None', 'em:TS',
                                       'em:NBVS', 'em:BCTS', 'em:VS'])])
methodgroups_bbserlls = OrderedDict([('bbse',
                                      ['bbse-soft:None', 'bbse-soft:TS',
                                       'bbse-soft:NBVS', 'bbse-soft:BCTS',
                                       'bbse-soft:VS']),
                                     ('rlls',
                                      ['rlls-soft:None', 'rlls-soft:TS',
                                       'rlls-soft:NBVS', 'rlls-soft:BCTS',
                                       'rlls-soft:VS'])])

#Accuracy improvement with EM
methodgroups = methodgroups_em
print(render_adaptation_table(
    alpha_to_samplesize_to_adaptncalib_to_metric_to_vals=dirichlet_alpha_to_samplesize_to_adaptername_to_metric_to_vals,
    ustat_threshold=ustat_threshold,
    valmultiplier=100,
    adaptname_to_nicename=adaptname_to_nicename,
    calibname_to_nicename=calibname_to_nicename,
    methodgroups=methodgroups,
    metric='delta_acc',
    largerisbetter=True,
    alphas_in_table=dirichlet_alphas_in_table,
    samplesizes_in_table=samplesizes_in_table,
    caption=("MNIST EM Metric: $\\Delta$\\%Accuracy, dirichlet shift."),
    label="MNIST_em_deltaacc_dirichletshift",
    applyunderline=False))
print(render_adaptation_table(
    alpha_to_samplesize_to_adaptncalib_to_metric_to_vals=tweakone_alpha_to_samplesize_to_adaptername_to_metric_to_vals,
    ustat_threshold=ustat_threshold,
    valmultiplier=100,
    adaptname_to_nicename=adaptname_to_nicename,
    calibname_to_nicename=calibname_to_nicename,
    methodgroups=methodgroups,
    metric='delta_acc',
    largerisbetter=True,
    alphas_in_table=tweakone_alphas_in_table,
    samplesizes_in_table=samplesizes_in_table,
    caption=("MNIST EM Metric: $\\Delta$\\%Accuracy, tweakone shift"),
    label="MNIST_em_deltaacc_tweakoneshift",
    applyunderline=False,
    symbol='\\rho'))

#JSDiv and MSE for all
for metric, nicemetricname, decimals in [('jsdiv', 'JS Divergence', 3),
                                         ('mseweights_even', 'MSE Across Classes', 5)]:
  methodgroups = methodgroups_all
  print(render_adaptation_table(
      alpha_to_samplesize_to_adaptncalib_to_metric_to_vals=dirichlet_alpha_to_samplesize_to_adaptername_to_metric_to_vals,
      ustat_threshold=ustat_threshold,
      valmultiplier=1.0,
      adaptname_to_nicename=adaptname_to_nicename,
      calibname_to_nicename=calibname_to_nicename,
      methodgroups=methodgroups,
      metric=metric,
      largerisbetter=False,
      alphas_in_table=dirichlet_alphas_in_table,
      samplesizes_in_table=samplesizes_in_table,
      caption="MNIST All Metric: "+nicemetricname+", dirichlet shift",
      label="MNIST_all_"+metric+"_dirichletshift",
      applyunderline=False,
      decimals=decimals))
  print(render_adaptation_table(
      alpha_to_samplesize_to_adaptncalib_to_metric_to_vals=tweakone_alpha_to_samplesize_to_adaptername_to_metric_to_vals,
      ustat_threshold=ustat_threshold,
      valmultiplier=1.0,
      adaptname_to_nicename=adaptname_to_nicename,
      calibname_to_nicename=calibname_to_nicename,
      methodgroups=methodgroups,
      metric=metric,
      largerisbetter=False,
      alphas_in_table=tweakone_alphas_in_table,
      samplesizes_in_table=samplesizes_in_table,
      caption="MNIST All Metric: "+nicemetricname+", tweakone shift",
      label="MNIST_all_"+metric+"_tweakoneshift",
      applyunderline=False,
      symbol='\\rho',
      decimals=decimals))

#Shift estimates from BBSE/RLLS vs. EM
methodgroups_bycalib = OrderedDict([
  ('None', ['histnet:None', 'em:None', 'bbse-hard:None', 'bbse-soft:None', 'rlls-hard:None', 'rlls-soft:None']),
  ('TS', ['em:TS', 'bbse-soft:TS', 'rlls-soft:TS']),
  ('NBVS', ['em:NBVS', 'bbse-soft:NBVS', 'rlls-soft:NBVS']),
  ('BCTS', ['em:BCTS', 'bbse-soft:BCTS', 'rlls-soft:BCTS']),
  ('VS', ['em:VS', 'bbse-soft:VS', 'rlls-soft:VS'])])
for (metric,nicemetricname,
     valmultiplier,largerisbetter,
     adaptname_to_nicename_touse,
     methodgroups_touse,
     decimals) in [('jsdiv', 'JS Divergence', 1.0, False, adaptname_to_nicename, methodgroups_bycalib, 3),
                   ('mseweights_even', 'MSE Across Classes', 1.0, False, adaptname_to_nicename, methodgroups_bycalib, 5)]:
  print(render_adaptation_table(
      alpha_to_samplesize_to_adaptncalib_to_metric_to_vals=dirichlet_alpha_to_samplesize_to_adaptername_to_metric_to_vals,
      ustat_threshold=ustat_threshold,
      valmultiplier=valmultiplier,
      adaptname_to_nicename=adaptname_to_nicename_touse,
      calibname_to_nicename=calibname_to_nicename,
      methodgroups=methodgroups_touse,
      metric=metric,
      largerisbetter=largerisbetter,
      alphas_in_table=dirichlet_alphas_in_table,
      samplesizes_in_table=samplesizes_in_table,
      caption="MNIST, EM vs BBSL vs RLLS, Evaluation Metric: "+nicemetricname+" dirichlet",
      label="MNIST_emvsbbsevsrlls_"+metric+"_dirichlet",
      applyunderline=False,
      symbol="\\alpha",
      decimals=decimals))

  print(render_adaptation_table(
      alpha_to_samplesize_to_adaptncalib_to_metric_to_vals=tweakone_alpha_to_samplesize_to_adaptername_to_metric_to_vals,
      ustat_threshold=ustat_threshold,
      valmultiplier=valmultiplier,
      adaptname_to_nicename=adaptname_to_nicename_touse,
      calibname_to_nicename=calibname_to_nicename,
      methodgroups=methodgroups_touse,
      metric=metric,
      largerisbetter=largerisbetter,
      alphas_in_table=tweakone_alphas_in_table,
      samplesizes_in_table=samplesizes_in_table,
      caption="MNIST, EM vs BBSL vs RLLS, Evaluation Metric: "+nicemetricname+", tweakone",
      label="MNIST_emvsbbsevsrlls_"+metric+"_tweakone",
      applyunderline=False,
      symbol="\\rho",
      decimals=decimals))

methodgroups = OrderedDict([
  ('em-calib', ['em:best-nll', 'em:best-ece'])])

print(render_adaptation_table(
    alpha_to_samplesize_to_adaptncalib_to_metric_to_vals=dirichlet_alpha_to_samplesize_to_adaptername_to_metric_to_vals,
    ustat_threshold=ustat_threshold,
    valmultiplier=100,
    adaptname_to_nicename=adaptname_to_nicename,
    calibname_to_nicename=calibname_to_nicename,
    methodgroups=methodgroups,
    metric='delta_acc',
    largerisbetter=True,
    alphas_in_table=dirichlet_alphas_in_table,
    samplesizes_in_table=samplesizes_in_table,
    caption="\\textbf{MNIST: NLL vs ECE, $\\Delta$\\%Accuracy, dirichlet shift.}",
    label="mnist_nllvsece_deltaacc_dirichletshift",
    applyunderline=False))

print(render_adaptation_table(
    alpha_to_samplesize_to_adaptncalib_to_metric_to_vals=tweakone_alpha_to_samplesize_to_adaptername_to_metric_to_vals,
    ustat_threshold=ustat_threshold,
    valmultiplier=100,
    adaptname_to_nicename=adaptname_to_nicename,
    calibname_to_nicename=calibname_to_nicename,
    methodgroups=methodgroups,
    metric='delta_acc',
    largerisbetter=True,
    alphas_in_table=tweakone_alphas_in_table,
    samplesizes_in_table=samplesizes_in_table,
    caption="\\textbf{MNIST: NLL vs ECE, $\\Delta$\\%Accuracy ``tweak-one'' shift.}",
    label="mnist_nllvsece_deltaacc_tweakoneshift",
    applyunderline=False,
    symbol='\\rho'))

methodgroups = OrderedDict([
  ('em-calib', ['em:best-nll', 'em:best-ece']),
  ('bbse-calib', ['bbse-soft:best-nll', 'bbse-soft:best-ece']),
  ('rlls-calib', ['rlls-soft:best-nll', 'rlls-soft:best-ece'])])

for metric, nicemetricname, decimals in [('jsdiv', 'JS Divergence', 3),
                                         ('mseweights_even', 'MSE Across Classes', 5)]:
  
  print(render_adaptation_table(
      alpha_to_samplesize_to_adaptncalib_to_metric_to_vals=dirichlet_alpha_to_samplesize_to_adaptername_to_metric_to_vals,
      ustat_threshold=ustat_threshold,
      valmultiplier=100,
      adaptname_to_nicename=adaptname_to_nicename,
      calibname_to_nicename=calibname_to_nicename,
      methodgroups=methodgroups,
      metric=metric,
      largerisbetter=False,
      alphas_in_table=dirichlet_alphas_in_table,
      samplesizes_in_table=samplesizes_in_table,
      caption="\\textbf{MNIST: NLL vs ECE, "+nicemetricname+", dirichlet shift.}",
      label="mnist_nllvsece_"+metric+"_dirichletshift",
      applyunderline=False,
      decimals=decimals))
  
  print(render_adaptation_table(
      alpha_to_samplesize_to_adaptncalib_to_metric_to_vals=tweakone_alpha_to_samplesize_to_adaptername_to_metric_to_vals,
      ustat_threshold=ustat_threshold,
      valmultiplier=100,
      adaptname_to_nicename=adaptname_to_nicename,
      calibname_to_nicename=calibname_to_nicename,
      methodgroups=methodgroups,
      metric=metric,
      largerisbetter=False,
      alphas_in_table=tweakone_alphas_in_table,
      samplesizes_in_table=samplesizes_in_table,
      caption="\\textbf{MNIST NLL vs ECE: "+nicemetricname+", ``tweak-one'' shift.}",
      label="mnist_nllvsece_"+metric+"_tweakoneshift",
      applyunderline=False,
      symbol='\\rho',
      decimals=decimals))

  