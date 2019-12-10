FILES=exp_shawn/*
mkdir exp_shawn/results
for FILE in $FILES
do
    for METRIC in FID bert_score rouge
    do
        echo $FILE
        echo $METRIC
        python run_evaluation.py \
            --evaluation $METRIC \
            --input-file $FILE \
            --max-real-samples 1000 \
            > "exp_shawn/results/$(basename $FILE)${METRIC}.txt"
    done
done
