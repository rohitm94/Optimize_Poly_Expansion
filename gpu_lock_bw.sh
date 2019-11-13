rm -rf gpu_bandwidth.txt

for n in `seq 250000000 100000000 1999999999`
	do
		./gpu_lock_bw $n 0 >> gpu_bandwidth.txt
	done