echo "starting reddit models"
python3 evaluate_mlm.py --model_path="../fine-tuned-models/reddit-left" --eval_path='political_compass.jsonl' --output_file="political_compass/reddit-left-eval.csv"
python3 evaluate_mlm.py --model_path="../fine-tuned-models/reddit-right" --eval_path='political_compass.jsonl' --output_file="political_compass/reddit-right-eval.csv"
python3 evaluate_mlm.py --model_path="../fine-tuned-models/reddit-center" --eval_path='political_compass.jsonl' --output_file="political_compass/reddit-center-eval.csv"

echo "news models"
python3 evaluate_mlm.py --model_path="../fine-tuned-models/news-left" --eval_path='political_compass.jsonl' --output_file="political_compass/news-left-eval.csv"
python3 evaluate_mlm.py --model_path="../fine-tuned-models/news-right" --eval_path='political_compass.jsonl' --output_file="political_compass/news-right-eval.csv"
python3 evaluate_mlm.py --model_path="../fine-tuned-models/news-center" --eval_path='political_compass.jsonl' --output_file="political_compass/news-center-eval.csv"

echo "roberta base"
python3 evaluate_mlm.py --model_path="distilbert/distilroberta-base" --eval_path='political_compass.jsonl' --output_file="political_compass/roberta-base-eval.csv"