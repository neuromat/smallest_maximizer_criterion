date

python smc.py -d 4 \
    -s examples/linguistic_case_study/folha.txt.bkp \
    -f /home/arthur/Documents/Neuromat/projects/SMC/results/python_pl_compat/bp \
    --perl_compatible 1 \
    --num_cores 7 \
    --split \> \
    -p 4 \
    bic \
    --penalty_interval 0 400

python smc.py -d 4 \
    -s examples/linguistic_case_study/publico.txt.bkp \
    -f /home/arthur/Documents/Neuromat/projects/SMC/results/python_pl_compat/ep \
    --perl_compatible 1 \
    -p 4 \
    --num_cores 7 \
    --split \> \
    bic \
    --penalty_interval 0 400

date
