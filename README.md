# Research on Fundamental Deep Learning

## Installing requirements

Simple `python3 -m pip install -r requirements.txt` should work. Be sure to use a virtualenv.

## Usage
Run a single local experiment with `python3 bert_train.py TESTING`. The flag `TESTING` will disable ClearML, and run a smaller model.
If you use ClearML, you must have clearml config in your home directory. Note that if you don't use srun.sbatch on entropy, you won't have access to GPU.

To run a single experiment on entropy, write your shell script based on `run_train.sh`, and run it either by:
`srun --partition=common --qos=24gpu7d --gres=gpu:titanv:1 run_time.sh`
or `sbatch run_time.sh`.

To run multiple experiment, modify gen_run_trains.py and run:
`python3 gen_run_trains.py --prefix=SWPB --real --sleep=70`

To just generate configs without scheduling jobs, run:
`python3 gen_run_trains.py --prefix=SWP --test --sleep=0`

## Code description (chaotic)

* true core
  * bert.py - główne warstwy
    * to powinno być zesplitowane w jakiś główny bert oraz bert-research; ew. research oddzielnie na różne warstwy
  * misc.py - różne warstwy, Trax-inspired
  * wikibookdata.py - data processing, etc
* support
  * metrics.py - logowanie metryk
  * ash.py - shape-checking
  * profile.py - do profilowania kodu w czasie działania
* research - sideprojects
  * initialization.py - to był mój poboczny projekt, robiłem inne warstwy z tym
* training
  * to wszystko powino być zrefactorowane
  * bert_train.py - mój główny projekt
    * powinien być zesplitowany
  * initbert_train.py - fork powyższego do podprojektu ze zmianą inicjalizacji
  * bert_time.py - mierzenie czasu, do profilowania (nie przejmujemy się)
* grid-search/entropy
  * run_train.sh / run_time.sh - pojedyncze skrypty do odpalania treningu
  * gen_run_trains.py / gen_initrun_trains.py - skrypty do odpalenia grid_searchu/wielu treningów naraz
* testy
  * test_utils.py - moduł do łatwiejszego robienia unittestów
  * test_*.py - unittesty do modułów, ofc
