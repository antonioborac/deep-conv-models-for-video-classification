glob_av_s = GlobalAveragePooling3D()(stem5_slow)
glob_av_f = GlobalAveragePooling3D()(stem5_fast)

conc = Concatenate(axis=-1)([glob_av_s,glob_av_f])
do = Dropout(dropout)(conc)
fc = Dense(_CLASSES, use_bias=True, activation="softmax")(do)