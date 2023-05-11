#!/bin/bash

################ 
# PARAMETERS
################

unused_embedding_methods="bert-base-nli-mean-tokens sentence-t5-xxl"
embedding_methods="all-mpnet-base-v2 all-MiniLM-L6-v2"
data_types="hired interviewed"

################ 
# INITIALISE
################

abs_start=`date +%s`
printf "Process started at %s\n" "$(date +'%Y-%m-%d %H:%M:%S')\n" > log.txt

source ../miniconda3/bin/activate job-status-prediction 

################ 
# MAIN
################

for embedding_method in $embedding_methods

do

    start=`date +%s`

    printf "Initialising encoding protocol using ***$embedding_method***\n" >> log.txt

    for data_type in $data_types

    do

        printf "Encoding $data_type data\n" >> log.txt

        python id_to_entity_id_list_encoder.py --data $data_type --embedding_method $embedding_method

    done

    end=`date +%s`

    printf "Encoding Complete. Process ended at %s\n\n" "$(date +'%Y-%m-%d %H:%M:%S')\n" >> log.txt
    printf "Total time taken: %s" "$(date -u -d @$(($end-$start)) +'%H:%M:%S')\n" >> log.txt

    printf "Beginning sweep on ***$embedding_method*** data. Process started at %s\n" "$(date +'%Y-%m-%d %H:%M:%S')\n" >> log.txt

    for data_type in $data_types

    do

        start=`date +%s`

        printf "Beginning sweep on $data_type data\n" >> log.txt

        python main_sweep.py --data $data_type --embedding_method $embedding_method 

        end=`date +%s`
        
        printf "$data_type sweep complete. Process ended at %s\n\n" "$(date +'%Y-%m-%d %H:%M:%S')\n" >> log.txt
        printf "Total time taken: %s" "$(date -u -d @$(($end-$start)) +'%H:%M:%S')\n" >> log.txt

    done

done