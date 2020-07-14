for i in *.tsv
do
	echo $i
	j=`basename $i .tsv`;
	cat $i | sed 's/	/ /g' > $j".txt"
done
