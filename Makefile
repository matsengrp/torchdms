default:

install:
	pip install -r requirements.txt
	pip install -e .

test: prep_1d
	# Full tdms go on vanilla
	cd torchdms/data; tdms go --config test_config.json
	pytest -vv
	rm torchdms/data/_ignore/test_df.prepped.pkl

test2d: prep_2d
	cd torchdms/data; tdms go --config test2d_config.json
	pytest -vv
	rm torchdms/data/_ignore/test_df_2d.prepped.pkl

testescape: prep_escape
	cd torchdms/data; tdms go --config test_escape_config.json
	pytest
	rm torchdms/data/_ignore/test_escape_df.prepped.pkl

datatest: prep_1d
	tdms validate torchdms/data/_ignore/test_df.prepped.pkl
	tdms summarize --out-prefix torchdms/data/_ignore/test_df.summary torchdms/data/_ignore/test_df.prepped.pkl

format:
	black .
	docformatter --in-place torchdms/*py

lint:
	pylint **/[^_]*.py && echo "PYLINTING PASS"
	# stop the build if there are Python syntax errors or undefined names
	flake8 **/[^_]*.py --count --select=E9,F63,F7,F82 --show-source --statistics
	# exit-zero treats all errors as warnings. The GitHub editor is 127 chars wide
	flake8 **/[^_]*.py --count --exit-zero --max-complexity=15 --max-line-length=127 --statistics

docs:
	make -C docs html

deploy:
	make docs
	git checkout gh-pages
	cp -a docs/_build/html/* .
	git add .
	git commit --amend -av -m "update docs"
	git push -f

prep_1d: torchdms/data/test_df.pkl
	mkdir -p torchdms/data/_ignore
	tdms prep --per-stratum-variants-for-test 10 --skip-stratum-if-count-is-smaller-than 30 torchdms/data/test_df.pkl torchdms/data/_ignore/test_df.prepped affinity_score

prep_2d: torchdms/data/test_df_2d.pkl
	mkdir -p torchdms/data/_ignore
	tdms prep --per-stratum-variants-for-test 10 --skip-stratum-if-count-is-smaller-than 30 torchdms/data/test_df_2d.pkl torchdms/data/_ignore/test_df_2d.prepped func_score func_score2

prep_escape: torchdms/data/test_escape_df.pkl
	mkdir -p torchdms/data/_ignore
	tdms prep --per-stratum-variants-for-test 10 --skip-stratum-if-count-is-smaller-than 30 torchdms/data/test_escape_df.pkl torchdms/data/_ignore/test_escape_df.prepped prob_escape

.PHONY: install test datatest format lint deploy docs
