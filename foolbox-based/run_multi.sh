for i in `seq 0 6`
do
	echo python main_pytorch.py --suffix unet20inc_lvl$i --atk_level $i
	python main_pytorch.py --suffix unet20inc_lvl$i --atk_level $i
done
