.PHONY: install test verify run ml deep nested windows boss context figures
install:
	python -m pip install -e '.[test,deep]'
test:
	python -m pytest -q
verify:
	python -m tpehg.cli verify
run:
	bash scripts/run_all.sh
ml:
	python -m tpehg.cli evaluate
deep:
	python -m tpehg.cli deep --epochs 10
nested:
	python scripts/run_advanced.py --folds 10
windows:
	python scripts/run_windows.py --folds 10
figures:
	python -m tpehg.cli figures
	python scripts/plot_protocols.py
	python scripts/make_report.py

boss:
	python scripts/run_boss.py --folds 10

context:
	bash scripts/run_context_only.sh

protocol-search:
	bash scripts/run_protocol_search.sh

.PHONY: design-search
design-search:
	bash scripts/run_design_search.sh
