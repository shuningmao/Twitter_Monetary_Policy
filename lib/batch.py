

class Batch(object):

    @staticmethod
    def iterate(batch_size, input_data, input_len, output_data=None):
        data_size = len(input_data)
        if output_data is not None:
            assert data_size == len(output_data), '[Batch] encoder data size must match decoder data size.'
            assert data_size == len(input_len), '[Batch] encoder data size must match length data size.'

        for batch in xrange(data_size // batch_size):
            start = batch * batch_size
            end = start + batch_size
            output_batch = None
            if output_data is not None:
                output_batch = output_data[start:end]
            yield(input_data[start:end], input_len[start:end], output_batch)

    @staticmethod
    def count(batch_size, input_data, input_len, output_data=None):
        data_size = len(input_data)
        if output_data is not None:
            assert data_size == len(output_data), '[Batch] encoder data ' +\
                'size must match decoder data size.'
            assert data_size == len(input_len), '[Batch] encoder data ' +\
                'size must match length data size.'
        return data_size // batch_size
