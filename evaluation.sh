for i in {1..20}
do
    echo
    echo "new_experiments"
    cd process_g2o
    python3 convert_to_multi.py ../datasets/manhattanOlson3500.g2o output1.g2o --max_inter_lc 15 --random_inter_lc 9
    cd ..
    python3 multi_robot_optimization.py process_g2o/output1.g2o
done

for i in {1..20}
do
    echo
    echo "new_experiments"
    cd process_g2o
    python3 convert_to_multi.py ../datasets/manhattanOlson3500.g2o output1.g2o --max_inter_lc 12 --random_inter_lc 12 output_fpath
    cd ..
    python3 multi_robot_optimization.py process_g2o/output1.g2o
done
