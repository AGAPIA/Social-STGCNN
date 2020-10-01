import subprocess
import os
import shlex
import time
import signal

cmd_line = "python test_flas.py --model_path /cluster/vol1/ciprian/Work/Social-STGCNN/checkpoint/custom_waymo --external 1"
args = shlex.split(cmd_line)
print(args)


for i in range(10):
	print("Test " , i)
	p = subprocess.Popen(args)

	print("Ok now i will sleep")
	time.sleep(3)
	print("Wake up and kill child")
	p.send_signal(signal.SIGINT)
	print("Message sent..should be good now")
