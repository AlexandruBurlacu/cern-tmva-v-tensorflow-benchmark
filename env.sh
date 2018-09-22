export KMP_AFFINITY=granularity=fine,compact,1,0
export vCPUs=`cat /proc/cpuinfo | grep processor | wc -l`
export OMP_NUM_THREADS=$((vCPUs / 2))
export KMP_BLOCKTIME=0
