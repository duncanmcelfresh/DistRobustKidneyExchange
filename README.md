# Distributionally Robust Kidney Exchange

Experiments testing various robust and non-robust kidney exchange matching algorithms. Written for Python 2.7 and Gurobi 8+. Not guaranteed to work for Python 3 (but it should).

Example usage:

```
python robust_kex_experiment.py  --num-weight-measurements=5  --output-dir ./output_directory --graph-type CMU --input-dir ./intput_directory
```

Will produce an output file that looks like this (the first line contains experiment parameters):

```
Namespace(chain_cap=4, cycle_cap=3, graph_type='CMU', input_dir='./intput_directory/', num_weight_measurements=5, num_weight_realizations=10, output_dir='./output_directory/', protection_level=0.001, seed=0, verbose=False)
     graph_name ,realization_num ,      cycle_cap ,      chain_cap ,protection_level ,nonrobust_score ,edge_weight_robust_score 
unos_bimodal_apd_v64_i3_maxcard.input ,              0 ,              3 ,              4 ,       1.00e-03 ,      6.700e+01 ,      7.000e+01 
unos_bimodal_apd_v64_i3_maxcard.input ,              1 ,              3 ,              4 ,       1.00e-03 ,      4.600e+01 ,      4.900e+01 
unos_bimodal_apd_v64_i3_maxcard.input ,              2 ,              3 ,              4 ,       1.00e-03 ,      4.700e+01 ,      4.400e+01 
unos_bimodal_apd_v64_i3_maxcard.input ,              3 ,              3 ,              4 ,       1.00e-03 ,      4.500e+01 ,      4.200e+01 
unos_bimodal_apd_v64_i3_maxcard.input ,              4 ,              3 ,              4 ,       1.00e-03 ,      5.600e+01 ,      5.600e+01 
unos_bimodal_apd_v64_i3_maxcard.input ,              5 ,              3 ,              4 ,       1.00e-03 ,      4.600e+01 ,      4.900e+01 
unos_bimodal_apd_v64_i3_maxcard.input ,              6 ,              3 ,              4 ,       1.00e-03 ,      4.500e+01 ,      4.000e+01 
unos_bimodal_apd_v64_i3_maxcard.input ,              7 ,              3 ,              4 ,       1.00e-03 ,      4.600e+01 ,      4.900e+01 
unos_bimodal_apd_v64_i3_maxcard.input ,              8 ,              3 ,              4 ,       1.00e-03 ,      5.500e+01 ,      5.700e+01 
unos_bimodal_apd_v64_i3_maxcard.input ,              9 ,              3 ,              4 ,       1.00e-03 ,      4.900e+01 ,      5.100e+01 
unos_bimodal_apd_v64_i25_maxcard.input ,              0 ,              3 ,              4 ,       1.00e-03 ,      5.500e+01 ,      5.600e+01 
unos_bimodal_apd_v64_i25_maxcard.input ,              1 ,              3 ,              4 ,       1.00e-03 ,      8.700e+01 ,      5.700e+01 
unos_bimodal_apd_v64_i25_maxcard.input ,              2 ,              3 ,              4 ,       1.00e-03 ,      8.200e+01 ,      6.300e+01 
unos_bimodal_apd_v64_i25_maxcard.input ,              3 ,              3 ,              4 ,       1.00e-03 ,      7.800e+01 ,      5.900e+01 
unos_bimodal_apd_v64_i25_maxcard.input ,              4 ,              3 ,              4 ,       1.00e-03 ,      7.100e+01 ,      7.100e+01 
unos_bimodal_apd_v64_i25_maxcard.input ,              5 ,              3 ,              4 ,       1.00e-03 ,      8.200e+01 ,      7.400e+01 
unos_bimodal_apd_v64_i25_maxcard.input ,              6 ,              3 ,              4 ,       1.00e-03 ,      6.700e+01 ,      6.900e+01 
unos_bimodal_apd_v64_i25_maxcard.input ,              7 ,              3 ,              4 ,       1.00e-03 ,      7.500e+01 ,      7.300e+01 
...
```
