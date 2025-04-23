@ ZIYAN
TUNING:
ALGORITHMS - [PPO, SAC, DESTA_SAC]
GAMMA - [0.99, 1.]
GRADIENT_STEPS - [1, 5, 15]
TAU - [0.005, 0.01]
TRAIN_FREQUENCY - [4, 8, 32, 64]
INTERVENTION_COST = [0, 1e-4, 1e-3, 1e-2, 1e-1]
STANDARD_AGENT_LEARNING_RATE - [1e-4, 1e-3]
SAFE_AGENT_LEARNING_RATE - [1e-4, 1e-3]
INTERVENTION_AGENT_LEARNING_RATE - [1e-4, 1e-3]



for gamma in 0.99 1.
do
    for gradient_steps in 1 5
    do
        for batch_size in 32 64
        do
            for train_frequency in 4 8
            do
                # echo "${gamma}_${gradient_steps}_${batch_size}_${train_frequency}"
                echo gamma=$gamma gradient_steps=$gradient_steps batch_size=$batch_size train_frequency=$train_frequency
                CUDA_VISIBLE_DEVICES=0 python3 -m safety_main --seed 0 --experiment "${gamma}_${gradient_steps}_${batch_size}_${train_frequency}"  --gamma $gamma --gradient_steps $gradient_steps --batch_size $batch_size --train_frequency $train_frequency &
                CUDA_VISIBLE_DEVICES=1 python3 -m safety_main --seed 10 --experiment "${gamma}_${gradient_steps}_${batch_size}_${train_frequency}"  --gamma $gamma --gradient_steps $gradient_steps --batch_size $batch_size --train_frequency $train_frequency &
                wait
            done
        done
    done
done






# CUDA_VISIBLE_DEVICES=0 python3 -m safety_main --experiment gamma_1 --gamma 0.99 --gradient_steps 1 batch_size 32 --tau 0.005 --train_frequency 4 --standard_agent_learning_rate 1e-4 &
# CUDA_VISIBLE_DEVICES=0 python3 -m safety_main --experiment gamma_2 --gamma 0.99 --gradient_steps 1 batch_size 32 --tau 0.005 --train_frequency 4 --standard_agent_learning_rate 1e-4 &
