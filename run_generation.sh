for MODE in ltr rnd
do
    for REF in gibbs xent none
    do
        python run_generation.py \
            --model_type xlnet \
            --model_name_or_path xlnet-base-cased \
            --refine $REF \
            --mode $MODE \
            --num_gen 5000
    done
done
