FILES=exp/*
for FILE in $FILES
do
    for metric in FID bert_score rouge
    do
        echo $FILE
        echo $METRIC
        python run_evaluation.py \
            --evaluation $METRIC \
            --input-file $FILE \
            --max-real-samples 5000 \
            > $FILE$METRIC.txt
    done
done
