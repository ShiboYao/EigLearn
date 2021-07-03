for k in 5 10 15 20 30 40 50 100 150 200 500 1000
do
    echo "python train.py --k=$k --data=cora"
    for s in {1..20}
    do
        for i in {1..10}
        do
            python train.py --k=$k --seed_data=$s --seed_train=$i
        done
    done
done
