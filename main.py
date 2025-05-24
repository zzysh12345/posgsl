import argparse
import os
import sys
import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='IMDB-BINARY', help='dataset')
parser.add_argument('--method', type=str, default='posgsl', help="Select methods")
parser.add_argument('--config', type=str, default=None)
parser.add_argument('--debug', action='store_false')
parser.add_argument('--gpu', type=str, default='0', help="Visible GPU")
parser.add_argument('--n_splits', type=int, default=None)
parser.add_argument('--n_runs', type=int, default=1)
parser.add_argument('--cv', type=int, default=10)
parser.add_argument('--attack', type=float, default=0)
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
# os.environ["OMP_NUM_THREADS"] = '1'

from opengsl.config import load_conf
from opengsl.data import Dataset
from opengsl import ExpManager
from opengsl.module import *
from model.posgsl import POSGSLSolver
import torch
# torch.autograd.set_detect_anomaly(True)

conf_path = args.config
if conf_path is None:
    conf_path = 'config_public/{}_{}.yaml'.format(args.method, args.data)
conf = load_conf(conf_path)
print(conf)

dataset = Dataset(args.data, **conf.dataset)
method = eval('{}Solver(conf, dataset)'.format(args.method.upper()))
exp = ExpManager(method)
acc_save, std_save = exp.run(n_runs=args.n_runs, n_splits=args.n_splits, debug=args.debug)
text = '{:.2f} ± {:.2f}'.format(acc_save, std_save)

if not os.path.exists('results'):
    os.makedirs('results')
if os.path.exists('results/performance.csv'):
    records = pd.read_csv('results/performance.csv')
    records.loc[len(records)] = {'method':args.method, 'data':args.data, 'acc':text}
    records.to_csv('results/performance.csv', index=False)
else:
    records = pd.DataFrame([[args.method, args.data, text]], columns=['method', 'data', 'acc'])
    records.to_csv('results/performance.csv', index=False)

print(torch.cuda.max_memory_allocated()/1048576)