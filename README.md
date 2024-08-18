```
pip install tqdm
```

### 执行成员推断攻击

```python
python main.py -target_model cnn -d CIFAR10 -s 2024
```

### 参数介绍 

1. **`-target_model` 参数**：指定要使用的目标模型类型。
2. **`-d` 参数**：定义用于训练的数据集。
3. **`-s` 参数**：设置随机种子以确保结果的可重复性。
4. **`--save_model` 参数**：指示是否保存训练好的模型。
5. **`--save_data` 参数**：指定是否保存训练过程中使用的数据。
6. **`--target_learning_rate` 参数**：目标模型的学习率。
7. **`--target_batch_size` 参数**：目标模型训练时使用的批大小。
8. **`--target_fc_dim_hidden` 参数**：目标模型中全连接层的隐藏单元数。
9. **`--target_epochs` 参数**：目标模型的训练轮数。
10. **`--n_shadow` 参数**：用于攻击模型训练的影子模型数量。
11. **`--attack_model` 参数**：指定要使用的攻击模型类型。
12. **`--attack_learning_rate` 参数**：攻击模型的学习率。
13. **`--attack_batch_size` 参数**：攻击模型训练时使用的批大小。
14. **`--attack_fc_dim_hidden` 参数**：攻击模型中全连接层的隐藏单元数。
15. **`--attack_epochs` 参数**：攻击模型的训练轮数。

#### 如果你想添加新的数据集，需要修改以下两处：

1. 在`get_data.py`函数中增加数据集导入过程
2. 在`utils.py`文件中的`get_more_args`函数中增加对应的数据集划分参数

#### 如果你想添加新的模型，需要在`model.py`中进行相应的添加

#### 如果在联邦学习场景下，把`train_target_model`函数修改为联邦学习训练即可
