EXEC = HarrisDetector
DIRS = bin


all: directory

directory:
	@for dir in $(DIRS); \
	do \
		printf "Checking $$dir..."; \
		if [ ! -d $$dir ]; then \
			mkdir $$dir; \
			echo "Creating directory \"$$dir\"..."; \
		else \
			echo "Directory exists."; \
		fi; \
	done
	@cd src; make -f make.src --no-print-directory EXEC=$(EXEC)
	@ln -fs bin/$(EXEC) $(EXEC)

.PHONY: clean

clean:
	@cd src; make -f make.src --no-print-directory EXEC=$(EXEC) clean;
