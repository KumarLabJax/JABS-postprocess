#!/bin/bash

usage_str="
Usage:
This runscript will call generate_behavior_tables.py
singularity run this_image.sif <args_to_python_script>
"

if [ $# -lt 2 ]; then
	echo ${usage_str}
else
	python3 /JABS-postprocess/generate_behavior_tables.py $@
fi