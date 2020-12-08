#!/bin/bash

trap 'kill $train_pid $board_pid; exit' SIGINT SIGTERM

rm -rf ./Result
python3 main.py &
train_pid=$!
tensorboard --logdir=./Result --reload_interval=15 &
board_pid=$!
wait $train_pid
kill $board_pid
