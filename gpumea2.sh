rm -rf bandwidth.txt

for n in `seq 250 250 999` \
	 `seq 2500 2500 9999` \
	 `seq 25000 25000 99999` \
	 `seq 250000 250000 999999` \
	 `seq 2500000 2500000 9999999` \
	 `seq 25000000 25000000 99999999`\
         `seq 250000000 100000000 1999999999`
	do
		./measure2 $n 1 >> bandwidth.txt
	done
