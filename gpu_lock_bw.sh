rm -rf gpu_bandwidth.txt

for n in `seq 250000000 250000000 1000000000`
	do
		./gpu_lock_bw $n 0 >> gpu_bandwidth.txt
	done