#!/bin/bash
python compute_stats_for_all_parts.py $1 $2 2 1000
python compute_stats_for_all_parts.py $1 $2 1 1000
