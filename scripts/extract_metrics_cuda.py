#!/usr/bin/env python3
import sys
import ncu_report
import argparse
import os
import glob
import pandas as pd

METRIC_LIST = {
  'sm_throughput': 'sm__throughput.avg.pct_of_peak_sustained_elapsed',
  'memory_throughput': 'gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed',
  'occupancy': 'sm__warps_active.avg.pct_of_peak_sustained_active',
  'l1_throughput': 'l1tex__throughput.avg.pct_of_peak_sustained_active',
  'l2_throughput': 'lts__throughput.avg.pct_of_peak_sustained_elapsed',
  'warps_active': 'sm__warps_active.avg.per_cycle_active',
  'warps_active_peak': 'sm__warps_active.avg.pct_of_peak_sustained_active',
  'registers': 'launch__registers_per_thread',
  'l1_hitrate': 'l1tex__t_sector_hit_rate.pct',
  'l2_hitrate': 'lts__t_sector_hit_rate.pct',
  'local_memory': 'smsp__sass_thread_inst_executed_op_memory_lg_local',
  'active_cycles': 'sm__active_cycles',
  'atomics': 'smsp__inst_executed_op_global_red.sum',
}

AGG_KEYS = [
  'max',
  'min',
  'avg',
]

def is_valid_action(ac, framework=None):
  # if 'workgroup_mapped_advance_kernel' not in ac.name() and 'block_mapped_kernel' not in ac.name() and ac.name() != 'kernel':
  #   return False
  return True

class Metric:
  def __init__(self, name):
    self.name = name
    self._occurrences = []
    
  def check_na(func):
    def wrapper(self, *args, **kwargs):
      # remove all NA values
      self._occurrences = [x for x in self._occurrences if x is not pd.NA]
      if not self._occurrences:
        return pd.NA
      return func(self, *args, **kwargs)
    return wrapper
  
  def add(self, value):
    self._occurrences.append(value)
  
  @check_na
  def max(self):
    return max(self._occurrences) if self._occurrences else 0
  
  @check_na
  def min(self):
    return min(self._occurrences) if self._occurrences else 0
  
  @check_na
  def avg(self):
    return sum(self._occurrences) / len(self._occurrences) if self._occurrences else 0

def get_metrics(fname):
  ctx = ncu_report.load_report(fname)
  rng = ctx.range_by_idx(0)

  acts = [rng.action_by_idx(i) for i in range(rng.num_actions())]

  metrics = {}
  
  for name, metric in METRIC_LIST.items():
    metrics[name] = Metric(metric)

  for ac in acts:
    if not is_valid_action(ac):
      continue
    
    for name, metric in METRIC_LIST.items():
      try:
        value = ac[metric].value()
        if value is not None:
          metrics[name].add(value)
        else:
          metrics[name].add(pd.NA)
      except KeyError:
        metrics[name].add(pd.NA)
        continue

  return metrics

def navigate_fs(path: str):
  if os.path.isfile(path):
    if path.endswith('.ncu-rep'):
      yield path
    else:
      return
  for el in glob.glob(os.path.join(path, '*')):
    el = os.path.abspath(el)
    if os.path.isdir(el):
      yield from navigate_fs(el)
    elif el.endswith('.ncu-rep'):
      yield el



if __name__ == '__main__':
  
  
  parser = argparse.ArgumentParser(description='Retrieve hit rate from NCU report')
  parser.add_argument('dir', type=str, help='Directory to NCU report files')
  parser.add_argument('out', nargs='?', type=str, help='Output file')
  parser.add_argument('--aggregate', '-a', nargs='+', choices=AGG_KEYS, default=['max'], help='Aggregation functions to apply')
  args = parser.parse_args()

  aggregation_list = args.aggregate

  df = pd.DataFrame(columns=['framework', 'dataset'] + [f'{x}_{y}' for x in METRIC_LIST.keys() for y in aggregation_list])
  
  for file in navigate_fs(args.dir):
    try:
      framework, dataset = [x for x in os.path.basename(file).split('.')[0].split('_')]
      metrics = get_metrics(file)
      vals = []
      for name in METRIC_LIST.keys():
        for agg in aggregation_list:
          vals.append(getattr(metrics[name], agg)())
      df.loc[len(df)] = [framework, dataset, *vals]
    except Exception as e:
      print(f'Error processing {file}:', e, file=sys.stderr)
      continue
  
  df.sort_values(by=['framework', 'dataset'], inplace=True)
  if args.out is None:
    print(df.to_string())
  else:
    df.to_csv(args.out, index=False)
