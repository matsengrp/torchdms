default:

install:
	pip install -e .

test: torchdms/data/_ignore/test_df.prepped.pkl
	cd torchdms/data; tdms go --config test_config.json
	cp torchdms/data/_ignore/test_df.prepped.pkl torchdms/data/test_df.prepped.pkl
	pytest
	rm torchdms/data/test_df.prepped.pkl torchdms/data/run.model

datatest: torchdms/data/_ignore/test_df.prepped.pkl
	tdms validate torchdms/data/_ignore/test_df.prepped.pkl
	tdms summarize --out-prefix torchdms/data/_ignore/test_df.summary torchdms/data/_ignore/test_df.prepped.pkl

format:
	black torchdms
	docformatter --in-place torchdms/*py

lint:
	pylint **/[^_]*.py && echo "LINTING PASS"

docs:
	make -C docs html

deploy:
	make docs
	git checkout gh-pages
	cp -a docs/_build/html/* .
	git add .
	git commit --amend -av -m "update docs"
	git push -f

torchdms/data/_ignore/test_df.prepped.pkl: torchdms/data/test_df.pkl
	mkdir -p torchdms/data/_ignore
	tdms prep --per-stratum-variants-for-test 10 --skip-stratum-if-count-is-smaller-than 30 torchdms/data/test_df.pkl torchdms/data/_ignore/test_df.prepped affinity_score

.PHONY: install test datatest format lint deploy docs
