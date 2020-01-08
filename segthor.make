CC = python3.7

# CFLAGS = -O
# DEBUG = --debug
EPC = 150
# EPC = 5


# G_RGX = (\d+_Case\d+_\d+)_\d+
G_RGX = (\d+_\w+__\d+)_\d+
NET = ResidualUNet
B_DATA = [('img', png_transform, False), ('gt', gt_transform, True)]

SIZES = results/segthor/sizeloss_e results/segthor/sizeloss_r
TRN = results/segthor/sizeloss_e \
   	  results/segthor/fs  \
	  results/segthor/sizeloss_r \
	# results/segthor/fs results/segthor/partial results/segthor/presize \
	# results/segthor/sizeloss_r \
	# results/segthor/presize_upper
	# results/segthor/3d_sizeloss



GRAPH = results/segthor/val_dice.png results/segthor/tra_loss.png \
		results/segthor/val_batch_dice.png results/segthor/tra_dice.png
HIST =  results/segthor/val_dice_hist.png results/segthor/tra_loss_hist.png \
		results/segthor/val_batch_dice_hist.png
BOXPLOT = results/segthor/val_batch_dice_boxplot.png results/segthor/val_dice_boxplot.png
PLT = $(GRAPH) $(HIST) $(BOXPLOT)


REPO = $(shell basename `git rev-parse --show-toplevel`)
DATE = $(shell date +"%y%m%d")
HASH = $(shell git rev-parse --short HEAD)
HOSTNAME = $(shell hostname)
PBASE = archives
PACK = $(PBASE)/$(REPO)-$(DATE)-$(HASH)-$(HOSTNAME)-segthor.tar.gz

all: pack

plot: $(PLT)

pack: $(PACK) report
$(PACK): $(PLT) $(TRN)
	mkdir -p $(@D)
	# tar -zc -f $@ $^  # Use if pigz is not available
	tar cf - $^ | pigz > $@
	chmod -w $@


# Extraction and slicing
data/SEGTHOR/train/gt data/SEGTHOR/val/gt: data/SEGTHOR
data/SEGTHOR: data/thor
	rm -rf $@_tmp
	$(CC) $(CFLAGS) slice_segthor.py --source_dir $< --dest_dir $@_tmp --n_augment=0
	mv $@_tmp $@
data/thor: data/segthor.lineage data/segthor.zip
	md5sum -c $<
	rm -rf $@_tmp
	unzip -q $(word 2, $^) -d $@_tmp
	mv $@_tmp $@


# Weak labels generation
weaks = data/SEGTHOR/train/centroid data/SEGTHOR/val/centroid\
		data/SEGTHOR/train/erosion data/SEGTHOR/val/erosion\
		data/SEGTHOR/train/random data/SEGTHOR/val/random
weak: $(weaks)

data/SEGTHOR/train/centroid data/SEGTHOR/val/centroid: OPT = --seed=0 --width=4 --r=0 --strategy=centroid_strat
data/SEGTHOR/train/erosion data/SEGTHOR/val/erosion: OPT = --seed=0 --strategy=erosion_strat
data/SEGTHOR/train/random data/SEGTHOR/val/random: OPT = --seed=0 --width=4 --r=0 --strategy=random_strat

$(weaks): data/SEGTHOR
	rm -rf $@_tmp
	$(CC) $(CFLAGS) gen_weak.py --selected_class 1 --filling 1 --base_folder=$(@D) --save_subfolder=$(@F)_tmp $(OPT)
	mv $@_tmp $@


data/SEGTHOR-Aug/train/gt data/SEGTHOR-Aug/val/gt: data/SEGTHOR-Aug
data/SEGTHOR-Aug/train/centroid data/SEGTHOR-Aug/val/centroid: data/SEGTHOR-Aug
data/SEGTHOR-Aug/train/erosion data/SEGTHOR-Aug/val/erosion: data/SEGTHOR-Aug
data/SEGTHOR-Aug/train/random data/SEGTHOR-Aug/val/random: data/SEGTHOR-Aug
data/SEGTHOR-Aug: data/SEGTHOR $(weaks)
	rm -rf $@ $@_tmp
	$(CC) $(CFLAGS) augment.py --n_aug 4 --root_dir $</train --dest_dir $@_tmp/train
	$(CC) $(CFLAGS) augment.py --n_aug 0 --root_dir $</val --dest_dir $@_tmp/val  # Naming scheme for consistency
	mv $@_tmp $@

data/SEGTHOR-aug-tiny: data/SEGTHOR-Aug
	rm -rf $@ $@_tmp
	cp -r $< $@_tmp
	for f in `ls $@_tmp/train` ; do \
		mogrify -resize '128x128!' $@_tmp/train/$$f/*.png ; \
	done
	for f in `ls $@_tmp/val` ; do \
		mogrify -resize '128x128!' $@_tmp/val/$$f/*.png ; \
	done
	mv $@_tmp $@



# Training
$(SIZES): OPT = --losses="[('CrossEntropy', {'idc': [1]}, None, None, None, 1), ('DiceLoss', {'idc': [0, 1]}, None, None, None, 1e-1),\
	('NaivePenalty', {'idc': [1]}, 'TagBounds', {'values': {1: [230, 7800]}, 'idc': [1]}, 'soft_size', 1e-2)]"
# -losses: List of list of (loss_name, loss_params, bounds_name, bounds_params, fn, weight)"
# --tentative of common bounds
# Idc is for filtering, I don't quite get why we would want it but It's everywhere so..
# From what i get from l.113 in main.py, 1e-2, the weight, is the lambda of the expression.
# --Full supervision
results/segthor/fs: OPT = --losses="[('CrossEntropy', {'idc': [0, 1]}, None, None, None, 1), ('DiceLoss', {'idc': [0, 1]}, None, None, None, 1e-1)]"
results/segthor/partial: OPT = --losses="[('CrossEntropy', {'idc': [1]}, None, None, None, 1)]"
results/segthor/presize: OPT = --losses="[('CrossEntropy', {'idc': [1]}, None, None, None, 1),\
	('NaivePenalty', {'idc': [1]}, 'PreciseBounds', {'margin': 0.10, 'mode': 'percentage'}, 'soft_size', 1e-2)]"
results/segthor/presize_upper: OPT = --losses="[('CrossEntropy', {'idc': [1]}, None, None, None, 1),\
	('NaivePenalty', {'idc': [1]}, 'PreciseUpper', {'margin': 0.10, 'mode': 'percentage'}, 'soft_size', 1e-2)]"
results/segthor/loose: OPT = --losses="[('CrossEntropy', {'idc': [1]}, None, None, None, 1),\
	('NaivePenalty', {'idc': [1]}, 'TagBounds', {'values': {1: [1, 65000]}, 'idc': [1]}, 'soft_size', 1e-2)]"
results/segthor/3d_sizeloss: OPT = --losses="[('CrossEntropy', {'idc': [1]}, None, None, None, 1), \
	('BatchNaivePenalty', {'idc': [1], 'margin': 0}, 'PreciseBounds', {'margin': 0, 'mode': 'percentage'}, 'soft_size', 1e-2)]" \
	--group_train
results/segthor/3d_sizeloss: NET = ENet

results/segthor/fs: data/SEGTHOR-Aug/train/gt data/SEGTHOR-Aug/val/gt
results/segthor/fs: DATA = --folders="$(B_DATA)+[('gt', gt_transform, True)]"

results/segthor/sizeloss_e results/segthor/neg: data/SEGTHOR-Aug/train/erosion data/SEGTHOR-Aug/val/erosion
results/segthor/sizeloss_e results/segthor/neg:  DATA = --folders="$(B_DATA)+[('erosion', gt_transform, True), ('erosion', gt_transform, True)]"

results/segthor/partial: data/SEGTHOR-Aug/train/random data/SEGTHOR-Aug/val/random
results/segthor/partial:  DATA = --folders="$(B_DATA)+[('random', gt_transform, True)]"

results/segthor/sizeloss_c: data/SEGTHOR-Aug/train/centroid data/SEGTHOR-Aug/val/centroid
results/segthor/sizeloss_c: DATA = --folders="$(B_DATA)+[('centroid', gt_transform, True), ('centroid', gt_transform, True)]"

results/segthor/sizeloss_r: data/SEGTHOR-Aug/train/random data/SEGTHOR-Aug/val/random
results/segthor/sizeloss_r: DATA = --folders="$(B_DATA)+[('random', gt_transform, True), ('random', gt_transform, True)]"

results/segthor/presize: data/SEGTHOR-Aug/train/random data/SEGTHOR-Aug/val/random
results/segthor/presize: DATA = --folders="$(B_DATA)+[('random', gt_transform, True), ('random', gt_transform, True)]"

results/segthor/presize_upper: data/SEGTHOR-Aug/train/random data/SEGTHOR-Aug/val/random
results/segthor/presize_upper: DATA = --folders="$(B_DATA)+[('random', gt_transform, True), ('random', gt_transform, True)]"

results/segthor/loose: data/SEGTHOR-Aug/train/random data/SEGTHOR-Aug/val/random
results/segthor/loose: DATA = --folders="$(B_DATA)+[('random', gt_transform, True), ('random', gt_transform, True)]"

results/segthor/3d_sizeloss: data/SEGTHOR-Aug/train/random data/SEGTHOR-Aug/val/random
results/segthor/3d_sizeloss: DATA = --folders="$(B_DATA)+[('random', gt_transform, True), ('random', gt_transform, True)]"


$(TRN):
	rm -rf $@_tmp
	$(CC) $(CFLAGS) main.py --dataset=$(dir $(<D)) --batch_size=4 --group --schedule \
		--n_epoch=$(EPC) --workdir=$@_tmp --csv=metrics.csv --n_class=2 --metric_axis=1 \
		--grp_regex="$(G_RGX)" --network=$(NET) $(OPT) $(DATA) $(DEBUG)
	mv $@_tmp $@


# Inference
INFR = results/segthor/inference/fs_sanity results/segthor/inference/size_sanity
results/segthor/inference/fs_sanity: results/segthor/fs data/SEGTHOR/test
results/segthor/inference/size_sanity: results/segthor/sizeloss_e data/SEGTHOR/test
$(INFR):
	$(CC) inference.py --save_folder $@_tmp --model_weights $</best.pkl --data_folder $(word 2, $^)/img --num_classes 2 $(OPT)
	$(CC) metrics.py --pred_folder $@_tmp/iter000 --gt_folder $(word 2, $^)/gt --save_folder $@_tmp  \
		--grp_regex="$(G_RGX)" --num_classes=2
	mv $@_tmp $@


# Plotting
results/segthor/val_batch_dice.png results/segthor/val_dice.png results/segthor/tra_dice.png: COLS = 1
results/segthor/tra_loss.png: COLS = 0
results/segthor/val_dice.png results/segthor/tra_loss.png results/segthor/val_batch_dice.png: plot.py $(TRN)
results/segthor/tra_dice.png: plot.py $(TRN)

results/segthor/val_haussdorf.png: COLS = 1
results/segthor/val_haussdorf.png: OPT = --ylim 0 7 --min
results/segthor/val_haussdorf.png: plot.py $(TRN)

results/segthor/val_batch_dice_hist.png results/segthor/val_dice_hist.png: COLS = 1
results/segthor/tra_loss_hist.png: COLS = 0
results/segthor/val_dice_hist.png results/segthor/tra_loss_hist.png results/segthor/val_batch_dice_hist.png: hist.py $(TRN)

results/segthor/val_batch_dice_boxplot.png results/segthor/val_dice_boxplot.png: COLS = 1
results/segthor/val_batch_dice_boxplot.png results/segthor/val_dice_boxplot.png: moustache.py $(TRN)

$(GRAPH) $(HIST) $(BOXPLOT):
	$(eval metric:=$(subst _hist,,$(@F)))
	$(eval metric:=$(subst _boxplot,,$(metric)))
	$(eval metric:=$(subst .png,.npy,$(metric)))
	$(CC) $(CFLAGS) $< --filename $(metric) --folders $(filter-out $<,$^) --columns $(COLS) \
		--savefig=$@ --headless --epc 199 $(OPT)

# Viewing
view: $(TRN)
	viewer -n 3 --img_source data/SEGTHOR/val/img data/SEGTHOR/val/gt $(addsuffix /best_epoch/val, $^) --crop 10 \
		--display_names gt $(notdir $^)

report: $(TRN)
	$(CC) $(CFLAGS) report.py --folders $(TRN) --metrics val_batch_dice val_dice --axises 1
