for top_args in "--top_k 0 --top_p 1." # "--top_k 8 --top_p 0.9"
do
    for LENGTH in 60 # 100
    do
        for MODE in rnd #ltr rnd_perm
        do
            for REF in gibbs xent none
            do
                echo sbatch --gres=gpu:1 --mem=4GB ./run.sh run_generation.py \
                    --model_type xlnet \
                    --model_name_or_path xlnet-base-cased \
                    --refine $REF \
                    --mode $MODE \
                    --length $LENGTH \
                    --num_gen 1000 \
                    $top_args &
            done
        done
    done
done
