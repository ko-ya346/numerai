#!/bin/bash

LOG_FILE="/home/kouya-takahashi/numerai/submissions/cron.log"

echo "Execution started at: $(date)" >> "$LOG_FILE"

/usr/local/bin/python3 /home/kouya-takahashi/numerai/submissions/main.py --config config.yaml >> "$LOG_FILE" 2>&1


echo "Execution ended at: $(date)" >> "$LOG_FILE"
echo "---------------------------------" >> "$LOG_FILE"
