#!/bin/bash
/usr/bin/time -o 05.fix.public.time.txt python ./spellchecker.py --input no_fix.public.submission.csv --output 05.fix.public.submission.csv --coeff 0.5 --cache
sleep 5
/usr/bin/time -o 10.fix.public.time.txt python ./spellchecker.py --input no_fix.public.submission.csv --output 10.fix.public.submission.csv --coeff 1.0 --cache
sleep 5
/usr/bin/time -o 10.fix.time.txt python ./spellchecker.py --input no_fix.submission.csv --output 10.fix.submission.csv --coeff 1.0 --cache
sleep 5
/usr/bin/time -o 07.fix.time.txt python ./spellchecker.py --input no_fix.submission.csv --output 07.fix.submission.csv --coeff 0.7 --cache
sleep 5
/usr/bin/time -o 08.fix.time.txt python ./spellchecker.py --input no_fix.submission.csv --output 08.fix.submission.csv --coeff 0.8 --cache
sleep 5
/usr/bin/time -o 09.fix.time.txt python ./spellchecker.py --input no_fix.submission.csv --output 09.fix.submission.csv --coeff 0.9 --cache
