cd process_g2o
python3 convert_to_multi.py ../datasets/manhattanOlson3500.g2o
cd ..
python3 multi_robot_optimization.py process_g2o/output.g2o