#!/bin/bash

# cd reevo
# conda activate ReEvo
# sh test_csi300.sh

python main.py \
    problem=csi300 \
    init_pop_size=10 \
    pop_size=10 \
    max_fe=200 \
    object_n=5 \
    timeout=20 \
    algorithm=TreEvo \
    llm_client=deepseek \
    llm_client.api_key='sk-fbe0a3a999b64026944664c70185c539'