test_exp () {
echo @@@@@@@@@@@@@@@ Testing $1 @@@@@@@@@@@@@@@@@@
python main.py --mode test --name $1 --data data/$2
echo Saving results ...
echo Computing Errors ...
python computeErrors.py --cnn results/$1_predictions.npy --data data/$2 --nz 8
echo End of Test $1
echo @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
echo @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
}

test_exp bareteeth bareteeth
test_exp cheeks_in cheeks_in
test_exp eyebrow eyebrow
test_exp high_smile high_smile
test_exp lips_back lips_back
test_exp lips_up lips_up
test_exp mouth_down mouth_down
test_exp mouth_extreme mouth_extreme
test_exp mouth_middle mouth_middle
test_exp mouth_open mouth_open
test_exp mouth_side mouth_side
test_exp mouth_up mouth_up
