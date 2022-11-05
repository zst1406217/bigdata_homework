# 添加特征life表示部件最大工作时长
for i in range(data_length):
    data[i][0]['life'] = data[i][0]['部件工作时长'].max()

train_set = data[0:800]
test_set = data[800:]
print('train set length:', len(train_set))
print('test set length:', len(test_set))
ratio_list = [0.45, 0.55, 0.63, 0.75, 0.85]
name_list = data[0][0].columns.values.tolist()
