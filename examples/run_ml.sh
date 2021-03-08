python simulation_study.py model1_5000_ml \
	--model model1 \
	--df csizar_and_talata \
	--resamples 200 \
	--sample_size 10000 \
	--num_cores 6 \
	--penalty_interval 0 100 \
	--scan_offset 0 \

python simulation_study.py model1_10000_ml \
	--model model1 \
	--df csizar_and_talata \
	--resamples 200 \
	--sample_size 10000 \
	--num_cores 6 \
	--penalty_interval 0 100 \
	--scan_offset 0 \

python simulation_study.py model1_20000_ml \
	--model model1 \
	--df csizar_and_talata \
	--resamples 200 \
	--sample_size 20000 \
	--num_cores 6 \
	--penalty_interval 0 100 \
	--scan_offset 0 \

python simulation_study.py model2_5000_ml \
	--model model2 \
	--df csizar_and_talata \
	--resamples 200 \
	--sample_size 10000 \
	--num_cores 6 \
	--penalty_interval 0 100 \
	--scan_offset 0 \

python simulation_study.py model2_10000_ml \
	--model model2 \
	--df csizar_and_talata \
	--resamples 200 \
	--sample_size 10000 \
	--num_cores 6 \
	--penalty_interval 0 100 \
	--scan_offset 0 \

python simulation_study.py model2_20000_ml \
	--model model2 \
	--df csizar_and_talata \
	--resamples 200 \
	--sample_size 20000 \
	--num_cores 6 \
	--penalty_interval 0 100 \
	--scan_offset 0 \
