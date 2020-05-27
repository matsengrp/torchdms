default:

install:
	pip install -e .

test: torchdms/data/_ignore/test_df.prepped.pkl
	cd torchdms/data; tdms go --config test_config.json
	rm torchdms/data/_ignore/test_df.prepped.pkl
	pytest

format:
	black torchdms
	docformatter --in-place torchdms/*py

lint:
	pylint torchdms && echo "LINTING PASS"

torchdms/data/_ignore/test_df.prepped.pkl: torchdms/data/test_df.pkl
	mkdir -p torchdms/data/_ignore
	tdms prep --per-stratum-variants-for-test 10 --skip-stratum-if-count-is-smaller-than 30 torchdms/data/test_df.pkl torchdms/data/_ignore/test_df.prepped affinity_score

.PHONY: install test format lint
