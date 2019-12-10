FILES=exp_shawn/*
for FILE in $FILES
do
    for METRIC in FID bert_score rouge
    do
        FILENAME=$(echo $FILE | rev | cut -c5- | rev)_$METRIC.txt
        echo $FILENAME
        python run_evaluation.py \
            --evaluation $METRIC \
            --input-file $FILE \
            --max-real-samples 1000 \
            > $FILENAME
    done
done
