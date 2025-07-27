#!/bin/bash

# cd reevo
# conda activate ReEvo
# sh test_ndx.sh

python main.py \
    problem=ndx \
    init_pop_size=4 \
    pop_size=4 \
    max_fe=100 \
    timeout=20 \
    algorithm=eoh \
    llm_client=qwen3_api \
    llm_client.api_key='sk-fb4917a77b7d4a2b88369204d7435aba'