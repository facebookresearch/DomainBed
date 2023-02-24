#!/bin/bash

CUDA_VISIBLE_DEVICES=2 python -m domainbed.scripts.collect_results\
	--input_dir=/pub2/tmp
