#!/bin/sh
python gan.py --mode=process_data --data_path=$1
python gan.py --mode=train --n_epochs=500
python gan.py --mode=inference --model_path=result/models/160800/

