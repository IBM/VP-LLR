#! /bin/bash
declare -a StringArray=("CIFAR10")  # Oter datasets: "SVHN" "GTSRB" "Melanoma" "CIFAR10" "OxfordIIITPet" "CIFAR100" "Flowers102" "DTD" "Food101" "EuroSAT" "UCF101" "FMoW"
for val in ${StringArray[@]}; do
    touch Evidence_log_$val.txt
    > Evidence_log_$val.txt # truncate the file
    declare -a StringArray2=("resnet18" "ig_resnext101_32x8d" "vit_b_16" "swin_t" "clip" )
    for val2 in ${StringArray2[@]}; do
        echo $val
        echo $val2

        # Linear probing
        python3 Evidence.py --dataset $val --datapath "/DATAPATH/$val" --mode "lp" --pretrained $val2 | tee -a Evidence_log_$val.txt
        # Without prompts
        python3 Evidence.py --dataset $val --datapath "/DATAPATH/$val" --mode "no_prompt" --pretrained $val2 | tee -a Evidence_log_$val.txt
        
        if [ "$val" = "SVHN" ]; then
            # Gaussian prompts
            python3 Evidence.py --dataset $val --datapath "/DATAPATH/$val" --pretrained $val2  --mode "gaussain" --img_scale 1.0 --mean 0.0 --std 10.0 | tee -a Evidence_log_$val.txt
            # Gradient prompts
            python3 Evidence.py --dataset $val --datapath "/DATAPATH/$val" --pretrained $val2  --mode "grad" --img_scale 1.0 --mean 0.0 --std 1.0 | tee -a Evidence_log_$val.txt
            # Mini-finetune 1 run
            python3 Evidence.py --dataset $val --datapath "/DATAPATH/$val" --pretrained $val2  --mode "mini_finetune" --runs 1 --img_scale 1.0 --mean 0.0 --std 0.1 | tee -a Evidence_log_$val.txt
            # Mini-finetune 5 runs
            python3 Evidence.py --dataset $val --datapath "/DATAPATH/$val" --pretrained $val2  --mode "mini_finetune" --runs 5 --img_scale 1.0 --mean 0.0 --std 0.1 | tee -a Evidence_log_$val.txt
        else
            # Gaussian prompts
            python3 Evidence.py --dataset $val --datapath "/DATAPATH/$val" --pretrained $val2  --mode "gaussain" --img_scale 1.5 --mean 0.0 --std 10.0 | tee -a Evidence_log_$val.txt
            # Gradient prompts
            python3 Evidence.py --dataset $val --datapath "/DATAPATH/$val" --pretrained $val2  --mode "grad" --img_scale 1.5 --mean 0.0 --std 1.0 | tee -a Evidence_log_$val.txt
            # Mini-finetune 1 run
            python3 Evidence.py --dataset $val --datapath "/DATAPATH/$val" --pretrained $val2  --mode "mini_finetune" --runs 1 --img_scale 1.5 --mean 0.0 --std 0.1 | tee -a Evidence_log_$val.txt
            # Mini-finetune 5 runs
            python3 Evidence.py --dataset $val --datapath "/DATAPATH/$val" --pretrained $val2  --mode "mini_finetune" --runs 5 --img_scale 1.5 --mean 0.0 --std 0.1 | tee -a Evidence_log_$val.txt
        fi

        # Trained prompts
        python3 Evidence.py --dataset $val --datapath "/DATAPATH/$val" --mode "from_file" --pretrained $val2 --ckpt_file "/CKPTPATH/${val}_last.pth" | tee -a Evidence_log_$val.txt
    done
done