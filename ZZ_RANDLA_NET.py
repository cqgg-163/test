


########源码摘抄
#网络部分：
# 参数
# 开始构建网络
with tf.variable_scope('inputs'):
    self.inputs = dict()    # 读取数据集输入
    num_layers = self.config.num_layers
    self.inputs['xyz'] = flat_inputs[:num_layers]
    self.inputs['neigh_idx'] = flat_inputs[num_layers: 2 * num_layers]
    self.inputs['sub_idx'] = flat_inputs[2 * num_layers:3 * num_layers]
    self.inputs['interp_idx'] = flat_inputs[3 * num_layers:4 * num_layers]
    self.inputs['features'] = flat_inputs[4 * num_layers]
    self.inputs['labels'] = flat_inputs[4 * num_layers + 1]
    self.inputs['input_inds'] = flat_inputs[4 * num_layers + 2]
    self.inputs['cloud_inds'] = flat_inputs[4 * num_layers + 3]
    self.labels = self.inputs['labels']     # 初始化参数、变量等杂项
    self.is_training = tf.placeholder(tf.bool, shape=())
    self.training_step = 1
    self.training_epoch = 0
    self.correct_prediction = 0
    self.accuracy = 0
    self.mIou_list = [0]
    self.class_weights = DP.get_class_weights(dataset.name)
    self.Log_file = open('log_train_' + dataset.name + str(dataset.val_split) + '.txt', 'a')

# ###########核心结构########################
with tf.variable_scope('layers'):
    self.logits = self.inference(self.inputs, self.is_training)
# 函数inference()中，就是网络的核心结构,
# 首先使用一个全连接层将特征数据变换为B*N*1*8
# 随后叠加上多层（超参数 num_layers=5）的dilated_res_block()，每层dilated_res_block()之后还进行random_sample()。



####Encoder
def inference(self, inputs, is_training):
    d_out = self.config.d_out
    feature = inputs['features']  # B*N*3(RGB)
    feature = tf.layers.dense(feature, 8, activation=None, name='fc0')  # B*N*8
    feature = tf.nn.leaky_relu(tf.layers.batch_normalization(feature, -1, 0.99, 1e-6, training=is_training))
    feature = tf.expand_dims(feature, axis=2)  # B*N*1*8



###上为编码部分，主要是一个dilated_res_block()的堆叠，也就调用的流程。预定义的代码块在下方。
    #上面接受输入特征后，全连接层将3维RGB特征扩展到8维，并添加了一个空维度，为后续2D卷积做准备

    # ###########################Encoder############################
    f_encoder_list = []
    for i in range(self.config.num_layers):
        #  # 残差块处理

        f_encoder_i = self.dilated_res_block(feature, inputs['xyz'][i], inputs['neigh_idx'][i], d_out[i],
                                             'Encoder_layer_' + str(i), is_training)

        # # 下采样
        f_sampled_i = self.random_sample(f_encoder_i, inputs['sub_idx'][i])
        feature = f_sampled_i

        # 保存特征用于跳跃连接
        if i == 0:
            f_encoder_list.append(f_encoder_i)# 原始分辨率特征
        f_encoder_list.append(f_sampled_i)# 下采样后特征




    # ###########################Encoder############################
    feature = helper_tf_util.conv2d(f_encoder_list[-1], f_encoder_list[-1].get_shape()[3].value, [1, 1],
                                    'decoder_0',
                                    [1, 1], 'VALID', True, is_training)  # B*N*1*1024

########################################################在编码器输出和解码器输入之间添加过渡层，精炼最深层的特征表示。

    def dilated_res_block(self, feature, xyz, neigh_idx, d_out, name, is_training):
        f_pc = helper_tf_util.conv2d(feature, d_out // 2, [1, 1], name + 'mlp1', [1, 1], 'VALID', True,
                                     is_training)  # B*N*1*d/2
        #特征维度减半，方便后去输入，为局部特征聚合准备

        #  # 局部特征聚合块
        f_pc = self.building_block(xyz, f_pc, neigh_idx, d_out, name + 'LFA', is_training)

        #
        f_pc = helper_tf_util.conv2d(f_pc, d_out * 2, [1, 1], name + 'mlp2', [1, 1], 'VALID', True, is_training,
                                     activation_fn=None)
        # # 捷径连接
        #     shortcut = helper_tf_util.conv2d(
        #         feature, d_out * 2, [1, 1], name + 'shortcut', [1, 1], 'VALID', activation_fn=None,
        #         bn=True, is_training=is_training
        #     )  # B*N*1*(d_out*2)
        #
        #     # 残差连接

        ###############?什么意思与两者作用
        shortcut = helper_tf_util.conv2d(feature, d_out * 2, [1, 1], name + 'shortcut', [1, 1], 'VALID',
                                         activation_fn=None, bn=True, is_training=is_training)
        return tf.nn.leaky_relu(f_pc + shortcut)


    #  # 局部特征聚合块
    def building_block(self, xyz, feature, neigh_idx, d_out, name, is_training):
        d_in = feature.get_shape()[-1].value  # B*N*1*d/2-输入特征维度

        # 1相对位置编码
        f_xyz = self.relative_pos_encoding(xyz, neigh_idx)  # B*N*16*10 对点的相关位置信息进行编码

        #2位置特征提取
        f_xyz = helper_tf_util.conv2d(f_xyz, d_in, [1, 1], name + 'mlp1', [1, 1], 'VALID', True,
                                      is_training)  # B*N*16*d/2
        #3邻居特征聚合
        f_neighbours = self.gather_neighbour(tf.squeeze(feature, axis=2), neigh_idx)  # B*N*16*d/2 获取点的相关颜色信息
        #4特征拼接
        f_concat = tf.concat([f_neighbours, f_xyz], axis=-1)  # B*N*16*d 连接颜色信息和位置信息
        #5自注意力池化
        # 对相关信息进行基于自注意力的池化，
        f_pc_agg = self.att_pooling(f_concat, d_out // 2, name + 'att_pooling_1',
                                    is_training)  # B*N*1*d/2 对相关信息进行基于自注意力的池化
        # 重复2345
        f_xyz = helper_tf_util.conv2d(f_xyz, d_out // 2, [1, 1], name + 'mlp2', [1, 1], 'VALID', True,
                                      is_training)  # B*N*16*d/2
        f_neighbours = self.gather_neighbour(tf.squeeze(f_pc_agg, axis=2), neigh_idx)  # B*N*16*d/2 再次获取点的相关特征信息
        f_concat = tf.concat([f_neighbours, f_xyz], axis=-1)  # B*N*16*d 连接特征信息和位置信息
        f_pc_agg = self.att_pooling(f_concat, d_out, name + 'att_pooling_2', is_training)  # B*N*1*d 注意力池化
        return f_pc_agg  # B*N*1*d


    # 相对位置编码
    def relative_pos_encoding(self, xyz, neigh_idx):  # 编码相关点的距离，方位，原始点坐标和相关点坐标
        #  # 获取邻居点坐标 B*N*16*3
        neighbor_xyz = self.gather_neighbour(xyz, neigh_idx)  # B*N*16*3
        #  # 复制当前点坐标 B*N*16*3
        xyz_tile = tf.tile(tf.expand_dims(xyz, axis=2), [1, 1, tf.shape(neigh_idx)[-1], 1])  # B*N*16*3
        #     # 计算相对坐标 B*N*16*3
        relative_xyz = xyz_tile - neighbor_xyz  # 计算每个点相关点的相对坐标 B*N*16*3
        #     # 计算欧氏距离 B*N*16*1
        relative_dis = tf.sqrt(tf.reduce_sum(tf.square(relative_xyz), axis=-1, keepdims=True))  # 距离 B*N*16*1
        # # 拼接所有几何信息 B*N*16*10
        relative_feature = tf.concat([relative_dis, relative_xyz, xyz_tile, neighbor_xyz], axis=-1)
        return relative_feature  # B*N*16*10=距离B*N*16*1+相关点方位B*N*16*3+点坐标B*N*16*3+相关点坐标B*N*16*3


    # 检索邻居特征
    @staticmethod
    def gather_neighbour(pc, neighbor_idx):
        # gather the coordinates or features of neighboring points  获取索引的坐标或特征
        batch_size = tf.shape(pc)[0]
        num_points = tf.shape(pc)[1]
        d = pc.get_shape()[2].value
        index_input = tf.reshape(neighbor_idx, shape=[batch_size, -1])  # batch_size * num_point*16
        features = tf.batch_gather(pc, index_input)  # 进行索引
        features = tf.reshape(features, [batch_size, num_points, tf.shape(neighbor_idx)[-1], d])
        return features  # B*N*16*d
    #

    # 为每个采样点进行自注意力池化，选16个邻居最大池化
    @staticmethod
    def random_sample(feature, pool_idx):
        """
        :param feature: [B, N, d] input features matrix
        :param pool_idx: [B, N', max_num] N' < N, N' is the selected position after pooling
        :return: pool_features = [B, N', d] pooled features matrix
        """
        feature = tf.squeeze(feature, axis=2)   # B*N*d
        num_neigh = tf.shape(pool_idx)[-1]  # 16
        d = feature.get_shape()[-1]
        batch_size = tf.shape(pool_idx)[0]
        pool_idx = tf.reshape(pool_idx, [batch_size, -1])
        pool_features = tf.batch_gather(feature, pool_idx)
        pool_features = tf.reshape(pool_features, [batch_size, -1, num_neigh, d])   # B*N'*16*d
        pool_features = tf.reduce_max(pool_features, axis=2, keepdims=True)
        return pool_features    # B*N'*1*d




    # ###########################Decoder############################
    f_decoder_list = []
    for j in range(self.config.num_layers):
        # 1上采样特插值
        f_interp_i = self.nearest_interpolation(feature, inputs['interp_idx'][-j - 1])  # 特征向上索引 B*N+*1*1024
        #特征拼接：编码器特征加上采样特征
        #反卷积特征精炼
        f_decoder_i = helper_tf_util.conv2d_transpose(tf.concat([f_encoder_list[-j - 2], f_interp_i], axis=3),  # B*N*1*1.5d
                                                      f_encoder_list[-j - 2].get_shape()[-1].value, [1, 1],
                                                      'Decoder_layer_' + str(j), [1, 1], 'VALID', bn=True,
                                                      is_training=is_training)      # 反卷积 B*N*1*0.5d
        feature = f_decoder_i
        f_decoder_list.append(f_decoder_i)
    # ###########################Decoder############################

    # 网络分割模块是由3个mlp+dropout组成
    # 3个全连接层，最后一个没有激活函数
    # 先将特征维度从1024降到64，再降到32，最后输出13个类别
    # 这里的特征维度是指每个点的特征维度
    # 这里的输出是B*N*1*13，最后一维是类别数
    #上面就是完整的解码过程，以下都是输出层的内容，
    # 特征精炼；特征压缩；正则化；分类输出；维度调整
    f_layer_fc1 = helper_tf_util.conv2d(f_decoder_list[-1], 64, [1, 1], 'fc1', [1, 1], 'VALID', True, is_training)  # B*N*1*64
    f_layer_fc2 = helper_tf_util.conv2d(f_layer_fc1, 32, [1, 1], 'fc2', [1, 1], 'VALID', True, is_training) # B*N*1*32
    f_layer_drop = helper_tf_util.dropout(f_layer_fc2, keep_prob=0.5, is_training=is_training, scope='dp1')
    f_layer_fc3 = helper_tf_util.conv2d(f_layer_drop, self.config.num_classes, [1, 1], 'fc', [1, 1], 'VALID', False,
                                        is_training, activation_fn=None)    # B*N*1*13
    f_out = tf.squeeze(f_layer_fc3, [2])
    return f_out    # B*N*class









#####################################################################
# Ignore the invalid point (unlabeled) when calculating the loss ########################
#####################################################################
with tf.variable_scope('loss'):
    self.logits = tf.reshape(self.logits, [-1, config.num_classes])
    self.labels = tf.reshape(self.labels, [-1])
    # Boolean mask of points that should be ignored
    ignored_bool = tf.zeros_like(self.labels, dtype=tf.bool)
    for ign_label in self.config.ignored_label_inds:
        ignored_bool = tf.logical_or(ignored_bool, tf.equal(self.labels, ign_label))
    # Collect logits and labels that are not ignored
    valid_idx = tf.squeeze(tf.where(tf.logical_not(ignored_bool)))
    valid_logits = tf.gather(self.logits, valid_idx, axis=0)
    valid_labels_init = tf.gather(self.labels, valid_idx, axis=0)
    # Reduce label values in the range of logit shape
    reducing_list = tf.range(self.config.num_classes, dtype=tf.int32)
    inserted_value = tf.zeros((1,), dtype=tf.int32)
    for ign_label in self.config.ignored_label_inds:
        reducing_list = tf.concat([reducing_list[:ign_label], inserted_value, reducing_list[ign_label:]], 0)
    valid_labels = tf.gather(reducing_list, valid_labels_init)
    self.loss = self.get_loss(valid_logits, valid_labels, self.class_weights)









def train(self, dataset):
    log_out('****EPOCH {}****'.format(self.training_epoch), self.Log_file)
    self.sess.run(dataset.train_init_op)
    while self.training_epoch < self.config.max_epoch:
        t_start = time.time()
        try:
            ops = [self.train_op,
                   self.extra_update_ops,
                   self.merged,
                   self.loss,
                   self.logits,
                   self.labels,
                   self.accuracy]
            _, _, summary, l_out, probs, labels, acc = self.sess.run(ops, {self.is_training: True})
            self.train_writer.add_summary(summary, self.training_step)
            t_end = time.time()
            if self.training_step % 50 == 0:
                message = 'Step {:08d} L_out={:5.3f} Acc={:4.2f} ''---{:8.2f} ms/batch'
                log_out(message.format(self.training_step, l_out, acc, 1000 * (t_end - t_start)), self.Log_file)
            self.training_step += 1

        except tf.errors.OutOfRangeError:

            m_iou = self.evaluate(dataset)
            if m_iou > np.max(self.mIou_list):
                # Save the best model
                snapshot_directory = join(self.saving_path, 'snapshots')
                makedirs(snapshot_directory) if not exists(snapshot_directory) else None
                self.saver.save(self.sess, snapshot_directory + './snap', global_step=self.training_step)
            self.mIou_list.append(m_iou)
            log_out('Best m_IoU is: {:5.3f}'.format(max(self.mIou_list)), self.Log_file)

            self.training_epoch += 1
            self.sess.run(dataset.train_init_op)
            # Update learning rate
            op = self.learning_rate.assign(tf.multiply(self.learning_rate,
                                                       self.config.lr_decays[self.training_epoch]))
            self.sess.run(op)
            log_out('****EPOCH {}****'.format(self.training_epoch), self.Log_file)

        except tf.errors.InvalidArgumentError as e:

            print('Caught a NaN error :')
            print(e.error_code)
            print(e.message)
            print(e.op)
            print(e.op.name)
            print([t.name for t in e.op.inputs])
            print([t.name for t in e.op.outputs])

            a = 1 / 0

    print('finished')
    self.sess.close()




################################################不同版本的完整核心代码块
    def inference(self, inputs, is_training):
        """
        RandLANet前向操作
        :param inputs:输入
        :param is_training:是否训练
        :return:
        """
        d_out = self.config.d_out  # 每一层输出特征通道数
        # 原始特征，带颜色为6(x,y,z,r,g,b)，不带颜色为3(x,y,z)
        feature = inputs['features']
        # 先通过一个mlp将特征通道数增加到8
        feature = tf.layers.dense(feature, 8, activation=None, name='fc0')
        feature = tf.nn.leaky_relu(tf.layers.batch_normalization(feature, -1, 0.99, 1e-6, training=is_training))
        feature = tf.expand_dims(feature, axis=2)

        # ###########################Encoder############################
        # 编码器
        f_encoder_list = []-
        for i in range(self.config.num_layers):  # layers为5
            # 执行LFA模块
            f_encoder_i = self.dilated_res_block(feature, inputs['xyz'][i], inputs['neigh_idx'][i], d_out[i],
                                                 'Encoder_layer_' + str(i), is_training)
            # 随机下采样，在dataset中已经提前获取了下采样后的点的坐标id
            f_sampled_i = self.random_sample(f_encoder_i, inputs['sub_idx'][i])
            # 将下采样后的点的特征作为下一层的输入特征
            feature = f_sampled_i
            if i == 0:
                f_encoder_list.append(f_encoder_i)
            f_encoder_list.append(f_sampled_i)
        # ###########################Encoder############################

        feature = helper_tf_util.conv2d(f_encoder_list[-1], f_encoder_list[-1].get_shape()[3].value, [1, 1],
                                        'decoder_0',
                                        [1, 1], 'VALID', True, is_training)

        # ###########################Decoder############################
        # 解码器
        f_decoder_list = []
        for j in range(self.config.num_layers):
            # 使用最邻近插值上采样
            f_interp_i = self.nearest_interpolation(feature, inputs['interp_idx'][-j - 1])
            # 卷积
            f_decoder_i = helper_tf_util.conv2d_transpose(tf.concat([f_encoder_list[-j - 2], f_interp_i], axis=3),
                                                          f_encoder_list[-j - 2].get_shape()[-1].value, [1, 1],
                                                          'Decoder_layer_' + str(j), [1, 1], 'VALID', bn=True,
                                                          is_training=is_training)
            feature = f_decoder_i
            f_decoder_list.append(f_decoder_i)
        # ###########################Decoder############################
        # 网络分割模块是由3个mlp+dropout组成
        f_layer_fc1 = helper_tf_util.conv2d(f_decoder_list[-1], 64, [1, 1], 'fc1', [1, 1], 'VALID', True, is_training)
        f_layer_fc2 = helper_tf_util.conv2d(f_layer_fc1, 32, [1, 1], 'fc2', [1, 1], 'VALID', True, is_training)
        f_layer_drop = helper_tf_util.dropout(f_layer_fc2, keep_prob=0.5, is_training=is_training, scope='dp1')
        f_layer_fc3 = helper_tf_util.conv2d(f_layer_drop, self.config.num_classes, [1, 1], 'fc', [1, 1], 'VALID', False,
                                            is_training, activation_fn=None)
        f_out = tf.squeeze(f_layer_fc3, [2])
        return f_out