from src.parser import *
from src.folderconstants import *


def resolve_dataset_profile(dataset_name):
	threshold_profiles = {
		'SMD': [(0.99995, 1.04), (0.99995, 1.06)],
		'synthetic': [(0.999, 1), (0.999, 1)],
		'SWaT': [(0.993, 1), (0.993, 1)],
		'UCR': [(0.993, 1), (0.99935, 1)],
		'NAB': [(0.991, 1), (0.99, 1)],
		'SMAP': [(0.98, 1), (0.98, 1)],
		'MSL': [(0.97, 1), (0.999, 1.04)],
		'WADI': [(0.99, 1), (0.999, 1)],
		'MSDS': [(0.91, 1), (0.9, 1.04)],
		'MBA': [(0.87, 1), (0.93, 1.04)],
	}
	lr_profiles = {
		'SMD': 0.0001,
		'synthetic': 0.0001,
		'SWaT': 0.008,
		'SMAP': 0.001,
		'MSL': 0.002,
		'WADI': 0.0001,
		'MSDS': 0.001,
		'UCR': 0.006,
		'NAB': 0.009,
		'MBA': 0.001,
	}
	percentile_profiles = {
		'SMD': (98, 2000),
		'synthetic': (95, 10),
		'SWaT': (95, 10),
		'SMAP': (97, 5000),
		'MSL': (97, 150),
		'WADI': (99, 1200),
		'MSDS': (96, 30),
		'UCR': (98, 2),
		'NAB': (98, 2),
		'MBA': (99, 2),
	}

	if dataset_name in threshold_profiles:
		return (
			threshold_profiles[dataset_name],
			lr_profiles[dataset_name],
			percentile_profiles[dataset_name],
		)

	if str(dataset_name).startswith(('NASA_RANDOM_CHARGE_', 'NASA_RANDOMRANDOM_DISCHARGE_', 'NASA_', 'BMS_')):
		return threshold_profiles['SMAP'], lr_profiles['SMAP'], percentile_profiles['SMAP']

	raise KeyError(f"Unsupported TranAD dataset profile: {dataset_name}")


lm_profile, lr, percentile_profile = resolve_dataset_profile(args.dataset)
lm = lm_profile[1 if 'TranAD' in args.model else 0]
percentile_merlin = percentile_profile[0]
cvp = percentile_profile[1]
preds = []
debug = 9
