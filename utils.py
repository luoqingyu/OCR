# -*- coding: UTF-8 -*-
import os
import numpy as np
import tensorflow as tf
from skimage import io
from skimage import transform




maxPrintLen = 100


tf.app.flags.DEFINE_string('checkpoint_dir', './checkpoint/lstm_3', 'the checkpoint dir')
tf.app.flags.DEFINE_integer('rnn_layers', 3 ,'number of rnn layers')
tf.app.flags.DEFINE_string('gpu_idex', '2' ,'index of gpu' )
tf.app.flags.DEFINE_string('model', 'lstm' , 'name of the rnn part')
tf.app.flags.DEFINE_string('log_dir', './log/lstm_3', 'the logging dir')


tf.app.flags.DEFINE_string('infer_dir', './data/infer/', 'the infer data dir')
tf.app.flags.DEFINE_boolean('restore', True, 'whether to restore from the latest checkpoint')
tf.app.flags.DEFINE_float('initial_learning_rate', 1e-3, 'inital lr')
tf.app.flags.DEFINE_integer('image_height', 32, 'image height')
tf.app.flags.DEFINE_integer('image_width', 256, 'image width')
tf.app.flags.DEFINE_integer('image_channel', 1, 'image channels as input')
tf.app.flags.DEFINE_integer('max_stepsize', 16, 'max stepsize in lstm, as well as '                                             'the output channels of last layer in CNN')
tf.app.flags.DEFINE_integer('num_hidden', 256, 'number of hidden units in lstm')
tf.app.flags.DEFINE_integer('num_epochs', 10000, 'maximum epochs')
tf.app.flags.DEFINE_integer('batch_size',32, 'the batch_size')
tf.app.flags.DEFINE_integer('save_steps', 50, 'the step to save checkpoint')
tf.app.flags.DEFINE_integer('validation_steps', 50, 'the step to validation')
tf.app.flags.DEFINE_float('decay_rate', 0.98, 'the lr decay rate')
tf.app.flags.DEFINE_float('beta1', 0.9, 'parameter of adam optimizer beta1')
tf.app.flags.DEFINE_float('beta2', 0.999, 'adam parameter beta2')
tf.app.flags.DEFINE_integer('decay_steps', 10000, 'the lr decay_step for optimizer')
tf.app.flags.DEFINE_float('momentum', 0.9, 'the momentum')

tf.app.flags.DEFINE_string('train_dir','./data/train/', 'the train data dir')
tf.app.flags.DEFINE_string('val_dir','./data/test/', 'the val data dir')

tf.app.flags.DEFINE_string('mode', 'train', 'train, val or infer')
tf.app.flags.DEFINE_integer('num_gpus', 1, 'num of gpus')

FLAGS = tf.app.flags.FLAGS

# num_batches_per_epoch = int(num_train_samples/FLAGS.batch_size)

encode_maps = {}
decode_maps = {}

with open("./dic.txt") as f:
    i = 1
    for line in f.readlines():
        line = line.replace('\n','')

        #line[1] = int(line[1])
        line =  unicode(line, 'utf-8')
        encode_maps[line] = i
        decode_maps[i] = line

        i += 1
    print  encode_maps
        
encode_maps[''] = 0
decode_maps[0] = ''

# 所有类 + blank + space
num_classes = i+1+1
print("num_classes:", num_classes)
class DataIterator:
    def __init__(self, data_dir,istrain=True):
        self.image = []
        self.labels = []
        if istrain:
            i=0
            for root, sub_folder, file_list in os.walk(data_dir):

                #print  file_list


                for file_path in file_list:


                    #print  file_path
                    file_path1  = unicode(file_path, 'utf-8')


                    i+=1
                    if 1==1:
                        image_name = os.path.join(root, file_path1)


                        if os.path.exists(image_name):
                            try:


                                im = io.imread(image_name,as_grey=True)

                                im = transform.resize(im, (FLAGS.image_height, FLAGS.image_width, FLAGS.image_channel))
                                self.image.append(im)
                                code = file_path1.split('/')[-1].split('.')[-2]
                                #print  code
                                code = [encode_maps[c] for c in list(code)]
                                #print  code
                                #print  '999'
                                #print  code
                                self.labels.append(code)
                            except:
                                continue
                #print  len(self.labels)
                #print  len(self.image)
        else:
            i=0
            for root, sub_folder, file_list in os.walk(data_dir):
                for file_path in file_list:
                    file_path1 = unicode(file_path, 'utf-8')
                    i+=1
                    if  1== 1:

                        image_name = os.path.join(root, file_path)
                        if os.path.exists(image_name):
                            try:
                                im = io.imread(image_name,as_grey=True)
                                im = transform.resize(im, (FLAGS.image_height, FLAGS.image_width, FLAGS.image_channel))
                                self.image.append(im)
                                code = file_path1.split('/')[-1].split('.')[-2]
                                code = [encode_maps[c] for c in list(code)]
                                self.labels.append(code)
                            except :
                                continue

    @property
    def size(self):

        return len(self.labels)


    def the_label(self, indexs):
        labels = []
        for i in indexs:
            labels.append(self.labels[i])

        return labels

    def input_index_generate_batch(self, index=None):
        if index:
            image_batch = [self.image[i] for i in index]
            label_batch = [self.labels[i] for i in index]
        else:
            image_batch = self.image
            label_batch = self.labels

        def get_input_lens(sequences):
            # 64 is the output channels of the last layer of CNN
            lengths = np.asarray([FLAGS.max_stepsize for _ in sequences], dtype=np.int64)

            return sequences, lengths

        batch_inputs, batch_seq_len = get_input_lens(np.array(image_batch))
        batch_labels = sparse_tuple_from_label(label_batch)

        return batch_inputs, batch_seq_len, batch_labels


def accuracy_calculation(original_seq, decoded_seq, ignore_value=-1, isPrint=False):
    if len(original_seq) != len(decoded_seq):
        print('original lengths is different from the decoded_seq, please check again')
        return 0
    count = 0
    for i, origin_label in enumerate(original_seq):
        decoded_label = [j for j in decoded_seq[i] if j != ignore_value]
        if isPrint and i < maxPrintLen:
            # print('seq{0:4d}: origin: {1} decoded:{2}'.format(i, origin_label, decoded_label))

            with open('./test.csv', 'a+') as f:
                f.write(str(origin_label) + '\t' + str(decoded_label))
                f.write('\n')

        if origin_label == decoded_label:
            count += 1

    return count * 1.0 / len(original_seq)


def sparse_tuple_from_label(sequences, dtype=np.int32):
    """Create a sparse representention of x.
    Args:
        sequences: a list of lists of type dtype where each element is a sequence
    Returns:
        A tuple with (indices, values, shape)
    """
    indices = []
    values = []

    for n, seq in enumerate(sequences):
        indices.extend(zip([n] * len(seq), range(len(seq))))
        values.extend(seq)

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1] + 1], dtype=np.int64)

    return indices, values, shape


def eval_expression(encoded_list):
    """
    :param encoded_list:
    :return:
    """

    eval_rs = []
    for item in encoded_list:
        try:
            rs = str(eval(item))
            eval_rs.append(rs)
        except:
            eval_rs.append(item)
            continue

    with open('./result.txt') as f:
        for ith in range(len(encoded_list)):
            f.write(encoded_list[ith] + ' ' + eval_rs[ith] + '\n')

    return eval_rs