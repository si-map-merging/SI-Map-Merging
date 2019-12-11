for i in {1..20}
do
    echo
    echo "new_experiments"
    cd process_g2o
    python3 convert_to_multi.py ../datasets/manhattanOlson3500.g2o output2.g2o --max_inter_lc 9 --random_inter_lc 15
    cd ..
    python3 multi_robot_optimization.py process_g2o/output2.g2o
done

for i in {1..20}
do
    echo
    echo "new_experiments"
    cd process_g2o
    python3 convert_to_multi.py ../datasets/manhattanOlson3500.g2o output2.g2o --max_inter_lc 6 --random_inter_lc 18
    cd ..
    python3 multi_robot_optimization.py process_g2o/output2.g2o
done
