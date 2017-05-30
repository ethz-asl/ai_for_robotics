.PHONY: generate_data, train, eval, run_tests, clean, clean_models, clean_data, distclean

generate_data:
	python generate_data.py

train:
	python train_classification.py --plot --verbosity info

eval:
	python test_classification.py --plot

run_tests:
	python unit_tests.py

clean:
	rm -rf *.pyc && rm -rf *~
	
clean_models:
	rm -rf models
	
clean_data:
	rm -rf data

distclean:
	$(MAKE) clean 
	$(MAKE) clean_models
	$(MAKE) clean_data

