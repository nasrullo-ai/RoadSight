PY ?= python
VIDEOS ?= data/samples
PRED ?= outputs/predictions_samples.json
GT ?= data/dev_labels.json

.PHONY: setup run eval validate test lint format demo render eda tune docker

setup:            ## install runtime + dev dependencies
	$(PY) -m pip install -r requirements-dev.txt

run:              ## run the harness on the sample videos
	$(PY) run_submission.py --videos $(VIDEOS) --out $(PRED)

validate:         ## format check only
	$(PY) evaluate.py --pred $(PRED) --validate-only

eval: validate    ## score against our dev labels
	$(PY) evaluate.py --pred $(PRED) --gt $(GT)

test:             ## pytest: interface, format, determinism, causality, offline, rules
	$(PY) -m pytest tests -q

lint:
	$(PY) -m ruff check roadsight tools tests web solution.py
	$(PY) -m black --check roadsight tools tests web solution.py

format:
	$(PY) -m black roadsight tools tests web solution.py

render:           ## annotated videos + timelines for the website
	$(PY) tools/render.py --videos $(VIDEOS) --out web/static/media --gt $(GT)

eda:              ## EDA stats and plots for the website
	$(PY) tools/eda.py --videos $(VIDEOS) --out web/static/eda

tune:             ## grid-search thresholds against dev labels
	ROADSIGHT_CACHE_DIR=.cache $(PY) tools/tune.py --videos $(VIDEOS) --gt $(GT)

demo:             ## local website + live demo on http://localhost:7860
	$(PY) -m uvicorn web.app:app --host 0.0.0.0 --port 7860

docker:           ## clean-machine check (no network)
	docker build -t roadsight .
	docker run --gpus all --network none -v $(PWD)/$(VIDEOS):/videos -v $(PWD)/outputs:/out roadsight
