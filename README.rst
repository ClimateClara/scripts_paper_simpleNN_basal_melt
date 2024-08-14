
Scripts used for the publication "Emulating present and future simulations of melt rates at the base of Antarctic ice shelves with neural networks."
===================================================================================================================================================

These are the scripts that were developed and used for the publication: Burgard, C., Jourdain, N. C., Mathiot, P., Smith, R.S., Schäfer, R., Caillet, J., Finn, T. S. and Johnson, J.E.: "Emulating present and future simulations of melt rates at the base of Antarctic ice shelves with neural networks." Journal of Advances in Modeling Earth Systems, https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2023MS003829, JAMES, 2023.


Useful functions are grouped in the package ``nn_funcs``. To install them and use them in further scripts, don't forget to run 

.. code-block:: bash

  pip install .
  
The scripts to format the data and produce the figures can be found in the folder ``scripts_and_notebooks``.

*Note - In the scripts, the NEMO runs are called 'OPM+number'. Here are the corresponding names given in the manuscript: OPM006=HIGHGETZ, OPM016=WARMROSS, OPM018=COLDAMU and OPM021=REALISTIC. Also 'bf663' is the REPEAT1970 run and 'bi646' is the 4xCO2 run*


Initial data formatting (from raw NEMO output to interesting variables gridded on stereographic grid)
-----------------------------------------------------------------------------------------------------

The scripts for the initial formatting and of the data, prepare the ice-shelf masks, the box and plume characteristics, and the temperature and salinity profiles can be found in ``scripts_and_notebooks/data_formatting``. 

The training data is the same as used in Burgard et al. (2022). See: https://github.com/ClimateClara/scripts_paper_assessment_basal_melt_param

To format the testing data from Smith et al. (2021): Start with ``data_formatting_smith.sh``, then move to ``custom_lsmask_Smith.ipynb`` and finally to ``regridding_vars_cdoSmith.ipynb``. At this point you have the relevant NEMO fields on a stereographic grid.

``isf_mask_NEMO_Smith.ipynb``  prepare masks of ice shelves, and plume and box characteristics, on the NEMO grid respectively. 

``prepare_reference_melt_file_Smith.ipynb`` prepares 2D and 1D metrics of the melt in NEMO for future comparison to the results of the parameterisations and neural network.

``T_S_profile_formatting_with_conversion_Smith.ipynb`` converts the 3D fields from absolute salinity to practical salinity.

``T_S_profiles_front_Smith.ipynb`` prepares the average temperature and salinity profiles in front of the ice shelf.

Conduct the preprocessing for the neural network
------------------------------------------------

The script for the preprocessing of the data to be fed as input for neural networks. To run in the following order:

    - ``prepare_2D_T_S_trainingruns.ipynb``, ``compute_bedrock_slope.ipynb``, ``prepare_mean_std_hydroinput.ipynb``
    - ``prepare_input_csv_extrap_chunks.ipynb``, ``prepare_inputdata_crossval.ipynb`` or ``prepare_indata_parallely.py``
    - ``prepare_inputdata_whole_dataset.ipynb``
    - ``prepare_2D_T_S_Smith.ipynb``, ``compute_bedrock_slope_Smith.ipynb``, ``prepare_mean_std_hydroinput_Smith.ipynb``
    - ``prepare_input_csv_Smith.ipynb``
    - ``shuffle_variables_Smith.ipynb``

Conduct the training of the neural networks
-------------------------------------------
The scripts to conduct the cross-validation, the best-estimate tuning and the tuning on different bootstrap samples can be found in ``scripts_and_notebooks/training``. 

``run_cross_validation_NN_experiments.ipynb`` for cross validation, ``run_training_whole_dataset.ipynb`` for the "final" training used in the test.



Apply the neural network
------------------------
The scripts to run the neural networks can be found in ``scripts_and_notebooks/postprocessing``. 

    - ``compute_1D_evalmetrics_directly_experiments.ipynb``: Evaluation metrics for cross validation
    - ``compute_2D_NN_experiments_CV.ipynb``: 2D output for cross-validation
    - ``compute_1D_NN_Smith_deepensemble.ipynb``: Evaluation metrics testing dataset
    - ``compute_2D_deepensemble_Smith.py``: 2D output testing dataset
    - ``script_to_apply_classic_param_Smith.ipynb``: Application of classic basal melt parameterisations to testing dataset
    - ``compute_2D_evalmetrics_shuffling_deepensemble_Smith.py``: 2D output for analysis of permute-and-predict
    - ``compute_1D_evalmetrics_shuffling_deepensemble_Smith.py``: Evaluation metrics for analysis of permute-and-predict

Final analysis and figures
--------------------------
The scripts to finalise the figures can be found in ``scripts_and_notebooks/figures``. 
