rm -rf pipelined_bandwidth.txt

for deg in  `seq 10 10 100` 
	do
		for n in  `seq 100 100 999` \
			  `seq 1000 1000 9999` \
			  `seq 10000 10000 99999` \
			  `seq 100000 100000 999999` \
			  `seq 1000000 1000000 9999999` \
			  `seq 10000000 10000000 99999999` \
			  `seq 100000000 100000000 999999999` \
			  `seq 1000000000 1000000000 10000000000`
        	do
            	    ./gpu_multi_stream $n $deg >> pipelined_bandwidth.txt
        	done
	done
