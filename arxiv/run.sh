for k in 20 50 100
do
	echo "python train.py --k=$k"
	python train.py --k=$k
done
