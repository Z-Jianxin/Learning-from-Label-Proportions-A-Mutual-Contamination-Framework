# Learning-from-Label-Proportions
0. Please cite *Clayton Scott and Jianxin Zhang, "Learning from Label Proportions: A Mutual Contamination Framework"*
1. Dependencies:
    * for LMMCM:
        * download and install Anaconda 
        * create an environment and install the dependencies: `conda env create -f LLP.yml` 
        * activate the new environment: `conda activate LLP`
    * for InvCal and pSVM:
        * follow the readme file in the psvm folder
2. Generate LLP data:
    * open the folder `LMMCM`
    * usage: `python make_data.py loader dataset_name path_to_save_mat path_to_save_binary lower_bound upper_bound train_size test_size [number_of_bags]`
        * to re-generate the Adult and Magic data in the main paper and save them to the folder `./experiments/`, run:
            * `python make_data.py load_adult adult0 ./experiments/ ./experiments/ 0 0.5 8192 3000`
            * `python make_data.py load_adult adult1 ./experiments/ ./experiments/ 0.5 1 8192 3000`
            * `python make_data.py load_magic magic0 ./experiments/ ./experiments/ 0 0.5 5120 1400`
            * `python make_data.py load_magic magic1 ./experiments/ ./experiments/ 0.5 1 5120 1400`
        * to re-generate the Adult and Magic data in the supplement and save them to the folder `./experiments/`, run:
            * `python make_data.py load_adult supp_adult0 ./experiments/ ./experiments/ 0 0.5 -1 3000 16`
            * `python make_data.py load_adult supp_adult1 ./experiments/ ./experiments/ 0.5 1 -1 3000 16`
            * `python make_data.py load_magic supp_magic0 ./experiments/ ./experiments/ 0 0.5 -1 1400 12`
            * `python make_data.py load_magic supp_magic1 ./experiments/ ./experiments/ 0.5 1 -1 1400 12`
        * the created dataset will be named as `[dataset name]_[bag size]_[trial id]`, trial id is an integer from 0 to 4
    * the program samples five subsets with repeatition for each bag size in [8, 16, 32, 64, 128, 256, 512] and saves the created dataset in a binary 
    * the program saves a copy of the created dataset in `.mat` format
3. Reproduce the LMMCM resuls:
    * open the folder `Learning-from-Label-Proportions`
    * usage: `python run_experiment.py data_path path_to_save_results`
    * example: `python run_experiment.py ./experiments/adult0_8_0 ./experiments/adult0_8_0_res`
4. Reproduce InvCal and pSVM results:
    * open the folder `pSVM`
    * run InvCal: `run_exp(@InvCal_cv, data_path, ouptput_path);`
    * run pSVM `run_exp(@alterPSVM_cv, data_path, ouptput_path);`
    * the code is modified from [Fexlix Yu's implementation of pSVM and InvCal](https://github.com/felixyu/pSVM)
