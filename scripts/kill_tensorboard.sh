kill -9 $(ps aux | grep 'tensorboard' | awk '{print $2}')
