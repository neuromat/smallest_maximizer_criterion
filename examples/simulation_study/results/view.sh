total_seq=$(cat SeqROCTM/model1_5000.csv | grep ',0 1' | wc -l)
found_seq=$(cat SeqROCTM/model1_5000.csv | grep ',000 1 10 100' | wc -l)
echo "SeqROCTM: ($total_seq)"
bc <<<"scale=2; $found_seq / $total_seq"
