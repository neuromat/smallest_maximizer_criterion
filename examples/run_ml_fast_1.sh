conda activate py37 && python simulation_study.py model1_5000_ml \
	--model model1 \
	--df csizar_and_talata \
	--resamples 200 \
	--sample_size 5000 \
	--num_cores 20 \
	--penalty_interval 0 500 \
	--scan_offset 0

conda activate py37 && python simulation_study.py model1_10000_ml \
	--model model1 \
	--df csizar_and_talata \
	--resamples 200 \
	--sample_size 10000 \
	--num_cores 20 \
	--penalty_interval 0 500 \
	--scan_offset 0

conda activate py37 && python simulation_study.py model1_20000_ml \
	--model model1 \
	--df csizar_and_talata \
	--resamples 200 \
	--sample_size 20000 \
	--num_cores 20 \
	--penalty_interval 0 500 \
	--scan_offset 0
