import os, time, sys
from subprocess import Popen, PIPE

"""
we want this script to be able to automate running the regex matching to make it easier to compare results and what not"""

__author__ = "Akshar Patel & Michael Gathara"
__email__ = "akshar20@uab.edu & mikegtr@uab.edu"

original_rules = []
lc = 0
with open("../data/SMS/rules.save.config", mode="r", encoding="utf-8") as original:
    # spam starts at line 5
    for line in original:
        if lc < 5:
            lc += 1
            continue
        else:
            original_rules.append(line)

"""original rules should hold the original rules"""

ham = """[ham]
$ * // Thanks. It was only from tescos but quite nice. All gone now. Speak soon
[spam]
"""

epochs = 1

if len(sys.argv) > 1:
    epochs = int(sys.argv[1])

if epochs > len(original_rules) - 1:
    epochs = len(original_rules) - 1

for i in range(epochs):
    # new_dir = f"results/{hash(time.time())}"
    new_dir = f"results/{i + 1}-rules"
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)

    print(f"\n\n\nRunning {i}th epoch, outputting to {new_dir}\n\n\n")
    
    with open("../data/SMS/rules.config", mode="w", encoding="utf-8") as out:
        out.write(ham)
        for line in original_rules[:i + 1]:
            out.write(line)
    
    command = f"cp ../data/SMS/rules.config {new_dir}/"
    os.system(command)
    
    # prepare data
    command = f"python3 data.py --dataset SMS"
    os.system(command)

    command = f"python create_automata.py --dataset SMS --automata_name all.pkl --reversed 0"
    os.system(command)

    command = f"python create_automata.py --dataset SMS --automata_name all.pkl --reversed 1"
    os.system(command)

    command = f"python main.py --model_type Onehot --dataset SMS --only_probe 1 --wfa_type viterbi --epoch 0 --automata_path_forward all.pkl --seed 0"
    os.system(command)

    command = f"python decompose_automata.py --dataset SMS --automata_name all.pkl --rank 150 --init svd"
    os.system(command)
    command = f"python main.py --dataset ATIS --run save_dir --model_type FSARNN --beta 0.9 \
    --wfa_type forward --seed 0 --lr 0.001 --bidirection 0 --farnn 0 --random 0"

    # command = f"cd ../src_simple/"
    # os.system(command)

    # run the fa -> rnn
    # ../../../
    print("Actually running main.py")
    output_file_path = os.path.join(new_dir, "out.txt")
    error_file_path = os.path.join(new_dir, "err.txt")
    with open(output_file_path, 'w') as output_file, open(error_file_path, 'w') as error_file:
        p = Popen(['python3', 'main.py', '--dataset', 'SMS', '--model_type', 'FSARNN', '--bidirection', '0', '--wfa_type', 'forward', '--automata_path_forward', 'automata.150.0.0000.0.pkl', '--lr', '0.001', '--random', '0'], stdout=output_file, stderr=error_file, stdin=PIPE)
        p.wait()
        
    automata_list = [
        '../data/SMS/automata/automata.150.0.0000.0.pkl',
        '../data/SMS/automata/automata.150.0.0000.1.pkl',
        '../data/SMS/automata/automata.150.0.0000.2.pkl',
        '../data/SMS/automata/automata.150.0.0000.3.pkl',
    ]
    for x in automata_list:
        command = f"rm -rf {x}"
        os.system(command)