# code-clinic

Code to be shared with the University of Sheffield Code Clinic.

## Session 11/05/2023

1. `python synthetic_data_generation.py` 
    - to generate synthetic data
    - runs ~ 40 seconds
2. `python id_to_entity_id_list_encoder.py --embedding_method all-mpnet-base-v2` 
    - to generate embeddings
    - runs ~ 2 minutes 30 seconds
3. `python main_nosweep.py --embedding_method all-mpnet-base-v2` 
    - to train the model
    - runs ~ 30 seconds per epoch (10 epochs ~ 5 minutes)