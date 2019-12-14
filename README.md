# Hybrid Analysis of Psychiatric Disorders in a Pediatric Dataset

This project is part of the Data Science Lab (Fall Semester 2019) at ETH.


| Name  | Email |
| ------------- | ------------- |
| Adamos Solomou  | solomoua@student.ethz.ch  |
| Anagnostidis Sotiris  | sanagnos@student.ethz.ch  |
| Vasilakopoulos George  | gvasilak@student.ethz.ch  |

## Project structure

    .
    ├── classification_clustering           # main code used for the classificion and clustering results
      ├── autoencode_predict.py             # model that reconstructs original data (as a simple autoencode) but also predicts according to the multilabel problem from the same latent space
      ├── autoencode_seq.py                 # autoencoder that also generates a sequence from the same latent space
      ├── autoencode_sequence_predict.py    # autoencoder that also generates a sequence and predicts according to the multilabel problem from the same latent space
      ├── utils.py                          # helping functions
      ├── classification_behavioral_and_combinations.ipynb  # classify users based on only the bahavioral data, as well by combining different data categories                                            
      ├── clustering_users.ipynb            # cluster users to new disorders based on the behavioral data                                    
      ├── TODO (eeg)
      └── TODO (mri)
    ├── data_driven_clusters                # final "new" disorders obtained, received from the clustering
    ├── preprocessing                       # includes useful statistics extracted from the data originally - many of the results and methods here not displayed to ensure data transparency
      └── imputation_methods.ipynb          # different techniques explored to fill missing values of the final cleaned version of the data provided
    ├── reports                       
      ├── poster.pdf
      └── report.pdf


## Getting Started

TODO

## Results

TODO
