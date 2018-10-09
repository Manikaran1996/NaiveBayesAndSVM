#!/bin/sh

if [ $1 -eq "1" ]; then
	if [ $2 -eq "1" ]; then
		python3 test.py 1  "$3" "$4"

	elif [ $2 -eq "2" ]; then
		python3 test.py 2 "$3" "$4"

	elif [ $2 -eq "3" ]; then
		python3 test.py 3 "$3" "$4"
	fi
else
	if [ $2 -eq "1" ]; then
		python3 test2.py 1 "$3" "$4"

	elif [ $2 -eq "2" ]; then
		python3 test2.py 2 "$3" "$4"

	elif [ $2 -eq "3" ];then
		python3 test2.py 3 "$3" "$4"
	fi
fi
