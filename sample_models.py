from keras import backend as K
from keras.models import Model
from keras.layers import (BatchNormalization, Conv1D, Dense, Input, 
    TimeDistributed, Activation, Bidirectional, SimpleRNN, GRU, LSTM, MaxPooling1D, Dropout)

def simple_rnn_model(input_dim, output_dim=29):
    """ Build a recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add recurrent layer
    simp_rnn = GRU(output_dim, return_sequences=True, 
                 implementation=2, name='rnn')(input_data)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(simp_rnn)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def rnn_model(input_dim, units, activation, output_dim=29):
    """ Build a recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add recurrent layer
    simp_rnn = GRU(units, activation=activation,
        return_sequences=True, implementation=2, name='rnn')(input_data)
    # TODO: Add batch normalization 
    bn_rnn = BatchNormalization(name='bn_simp_rnn')(simp_rnn)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model


def cnn_rnn_model(input_dim, filters, kernel_size, conv_stride,
    conv_border_mode, units, output_dim=29):
    """ Build a recurrent + convolutional network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add convolutional layer
    conv_1d = Conv1D(filters, kernel_size, 
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv1d')(input_data)
    # Add batch normalization
    bn_cnn = BatchNormalization(name='bn_conv_1d')(conv_1d)
    # Add a recurrent layer
    simp_rnn = SimpleRNN(units, activation='relu',
        return_sequences=True, implementation=2, name='rnn')(bn_cnn)
    # TODO: Add batch normalization
    bn_rnn = BatchNormalization(name='bn_simp_rnn')(simp_rnn)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride)
    print(model.summary())
    return model

def cnn_output_length(input_length, filter_size, border_mode, stride,
                       cnn_layers=0, pool_size=1, pool_stride=None, dilation=1):
    """ Compute the length of the output sequence after 1D convolution along
        time. Note that this function is in line with the function used in
        Convolution1D class from Keras.
    Params:
        input_length (int): Length of the input sequence.
        filter_size (int): Width of the convolution kernel.
        border_mode (str): Only support `same` or `valid`.
        stride (int): Stride size used in 1D convolution.
        dilation (int)
    """
    if input_length is None:
        return None
    assert border_mode in {'same', 'valid'}
    dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
    if border_mode == 'same':
        output_length = input_length
    elif border_mode == 'valid':
        output_length = input_length - dilated_filter_size + 1
    output_length = (output_length + stride - 1) // stride
    for i in range(cnn_layers):
        output_length = output_length - dilated_filter_size + 1
        output_length = (output_length + stride - 1) // stride
    #if pool_stride is None:
    #    output_length = (output_length + pool_size - 1) // pool_size
    #else:
    #    output_length = (output_length + pool_stride - 1) // pool_stride
    return output_length

def deep_rnn_model(input_dim, units, recur_layers, output_dim=29):
    """ Build a deep recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Add recurrent layers, each with batch normalization
    simp_rnn = input_data
    for i in range(recur_layers):
        simp_rnn = GRU(units, return_sequences=True, 
                       implementation=2, name='rnn_' + str(i + 1))(simp_rnn)
        simp_rnn = BatchNormalization(name='bn_simp_rnn_' + str(i + 1))(simp_rnn)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(simp_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def bidirectional_rnn_model(input_dim, units, output_dim=29):
    """ Build a bidirectional recurrent network for speech
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Add bidirectional recurrent layer
    bidir_rnn = Bidirectional(GRU(units, return_sequences=True,
                                  implementation=2, name='rnn'))(input_data)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bidir_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def final_model(input_dim, filters, conv, kernel_size, conv_stride,
    conv_border_mode, cnn_layers, pool_size, pool_stride, dropout, units,
    activation, recur_layers, dropout_W, dropout_U, output_dim=29):
    """ Build a deep network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Specify the layers in your network
    if conv:
        # Add the first convoluational layer and max pooling as in this paper:
        # https://arxiv.org/pdf/1701.02720.pdf
        conv_1d = Conv1D(filters, kernel_size, 
                         strides=conv_stride, 
                         padding=conv_border_mode,
                         activation='relu',
                         name='conv1d_1')(input_data)
        # Add pooling layer
        #max_pool_1d = MaxPooling1D(pool_size, pool_stride, name='maxpool1d')(conv_1d)
        # Add batch normalization
        #bn_max_pool_1d = BatchNormalization(name='bn_max_pool_1d')(max_pool_1d)
        bn_conv_1d = BatchNormalization(name='bn_conv_1d')(conv_1d)
        # Add dropout
        #dropout_1 = Dropout(rate=dropout, name='dropout_1')(bn_max_pool_1d)
        dropout_1 = Dropout(rate=dropout, name='dropout_1')(bn_conv_1d)
        # Add more convolutional layers
        cnn = dropout_1
        if cnn_layers > 0: 
            for i in range(cnn_layers):
                cnn = Conv1D(filters, kernel_size, 
                             strides=conv_stride, 
                             padding=conv_border_mode,
                             activation='relu',
                             name='conv1d_' + str(i + 2))(cnn)
            # Add batch normalization
            cnn = BatchNormalization(name='bn_cnn')(cnn)
            # Add dropout
            cnn = Dropout(rate=dropout, name='dropout_2')(cnn)
    else:
        cnn = input_data
    # Add recurrent layers
    rnn = cnn
    if recur_layers > 0:
        for i in range(recur_layers):
            rnn = Bidirectional(GRU(units, activation=activation,
                      return_sequences=True, implementation=2, name='rnn',
                      dropout=dropout_W, recurrent_dropout=dropout_U),
                      name='birnn_' + str(i + 1))(rnn)
        # TODO: Add batch normalization
        rnn = BatchNormalization(name='bn_rnn')(rnn)
        # Add dropout
        rnn = Dropout(rate=dropout, name='dropout_3')(rnn)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim),
                                name='time_distributed_dense')(rnn)
    # TODO: Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    # TODO: Specify model.output_length
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride, cnn_layers, pool_size, pool_stride)
    print(model.summary())
    return model