
# Create IID Partitions
# for num_clients in 20 50 80 110 140 170 200
# do
# 	echo ${num_clients}
# 	python3 data/utils/run.py -d tiny_imagenet -cn ${num_clients} --iid 1 --name iid-${num_clients}
# done

# Create Non-IID Shard Partitions
# for num_shards in 50 100 200 300 500
# do
# 	echo ${num_shards}
# 	python3 data/utils/run.py -d tiny_imagenet -cn 100 --shards ${num_shards} --name shards-${num_shards}
# done

# Create Non-IID Drichlet Partitions
# for alpha in 0.25 0.5 1.0 2.0 4.0
# do
#   echo ${alpha}
#   python3 data/utils/run.py -d tiny_imagenet -cn 100 --alpha ${alpha} --name alpha-${alpha}
# done

for std in 8.0 16.0 # 0.5 1.0 2.0 4.0 5.0
do
  echo ${std}
  python3 data/utils/run.py -d tiny_imagenet -cn 100 -cm 20 -cs ${std} --name classes-20-${std}
done
