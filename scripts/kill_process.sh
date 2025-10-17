#!/bin/bash
ProcessID=$1

childID=$(ps -o pid= --ppid $ProcessID)

if [[ $childID ]] ; then
	#sh ./kill_process childID
	$0 $childID
else
	echo $ProcessID "has no childs"
fi

echo "Killing " $ProcessID
kill -9 $ProcessID