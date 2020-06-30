self.preprared_aug = []
for i in range(self.file_count):
    self.preprared_aug.append(self.augumentation.get_random_transform(self.target_shape))