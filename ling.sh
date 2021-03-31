date

python smc.py -d 4 \
    -s examples/linguistic_case_study/folha.txt.bkp \
    -f /home/arthur/Documents/Neuromat/projects/SMC/results/python/bp \
    --num_cores 7 \
    --split \> \
    -p 4 \
    bic \
    --penalty_interval 0 400

python smc.py -d 4 \
    -s examples/linguistic_case_study/publico.txt.bkp \
    -f /home/arthur/Documents/Neuromat/projects/SMC/results/python/ep \
    -p 4 \
    --num_cores 7 \
    --split \> \
    bic \
    --penalty_interval 0 400

date
